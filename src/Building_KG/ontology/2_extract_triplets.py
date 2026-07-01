"""
Phase 2: Direct LLM triplet extraction using finalized ontology as anchor schema.

Replaces the VnCoreNLP pipeline (steps 3 + 5) with a single LLM call per batch.

Input:  ../material_for_triplets/2_sections_rewritten_*.json  +  ontology.json
Output: output/triplets_{document_slug}.json

How grounding works:
  - LLM receives the full ontology (concepts + keyphrases, relations + keyphrases + concept_s/concept_o)
  - LLM MUST use exact concept/relation names from ontology — no new names allowed
  - Post-filter hard-rejects any output not in ontology
  - Keyphrases in ontology help LLM recognize surface variants in text

Output format per triplet:
  {concept_s, relation, concept_o, section_id, address: {article, clause, point, document}}
"""

import os
import re
import json
import time
import glob
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
MATERIALS_DIR  = os.path.join(BASE_DIR, "..", "material_for_triplets")
DEFAULT_ONTO   = os.path.join(BASE_DIR, "..", "ontology_168.json")
OUTPUT_DIR     = os.path.join(BASE_DIR, "output")

BATCH_SIZE = 15   # sections per LLM call (each section may have multiple propositions)


# ── Ontology helpers ──────────────────────────────────────────────────────────

def load_ontology(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_ontology_context(ontology: dict) -> tuple[str, set, set]:
    """
    Returns (context_string, concept_name_set, relation_name_set).
    Context string is embedded in every LLM system prompt.
    """
    lines = ["=== ONTOLOGY — chỉ dùng tên CHÍNH XÁC từ danh sách này ===", "", "CONCEPTS:"]
    concept_names = set()
    for c in ontology["concepts"]:
        name = c["name"]
        concept_names.add(name)
        kps = ", ".join(f'"{k}"' for k in c.get("keyphrases", [])[:8])
        lines.append(f'  [{name}]: {kps}')

    lines.append("")
    lines.append("RELATIONS:")
    relation_names = set()
    for r in ontology["relations"]:
        name = r["name"]
        relation_names.add(name)
        kps  = ", ".join(f'"{k}"' for k in r.get("keyphrases", [])[:6])
        cs   = r.get("concept_s", "")
        co   = ", ".join(r.get("concept_o", [])[:6])
        lines.append(f'  [{name}] keyphrases: {kps} | S: {cs} | O: {co}')

    return "\n".join(lines), concept_names, relation_names


# ── Address parsing ───────────────────────────────────────────────────────────

def parse_address(path: str, document: str) -> dict:
    art = re.search(r"Điều\s+(\d+)", path or "")
    cl  = re.search(r"Khoản\s+(\d+)", path or "")
    pt  = re.search(r"Điểm\s+([a-zđA-ZĐ])", path or "")
    return {
        "article":  int(art.group(1)) if art else None,
        "clause":   int(cl.group(1))  if cl  else None,
        "point":    pt.group(1).lower() if pt else None,
        "document": document,
    }


# ── LLM ──────────────────────────────────────────────────────────────────────

SYSTEM_TEMPLATE = """\
{onto_context}

=== NHIỆM VỤ ===
Với mỗi phần gồm [section_id] và các mệnh đề pháp lý:
Trích xuất các triplet (ConceptS, Relation, ConceptO) thể hiện hành vi vi phạm cốt lõi.

Quy tắc bắt buộc:
1. concept_s, relation, concept_o PHẢI là tên CHÍNH XÁC từ ONTOLOGY ở trên.
2. Nếu một mệnh đề không khớp bất kỳ concept/relation nào → "triplets": []
3. Không tự đặt tên mới. Không dùng tên gần giống.
4. Một mệnh đề có thể sinh nhiều triplet nếu diễn đạt nhiều hành vi.

Trả về JSON:
{{
  "results": [
    {{
      "section_id": "s117",
      "triplets": [
        {{"concept_s": "Người điều khiển phương tiện", "relation": "Điều_khiển", "concept_o": "Xe ô tô"}},
        {{"concept_s": "Người điều khiển phương tiện", "relation": "Sử_dụng", "concept_o": "Cồn"}}
      ]
    }},
    {{
      "section_id": "s118",
      "triplets": []
    }}
  ]
}}"""


def llm_call(messages: list, max_retries: int = 4) -> dict:
    backoff = 2
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"  retry {attempt+1}: {e}")
            time.sleep(backoff)
            backoff *= 2


def extract_batch(
    sections: list[dict],
    system_prompt: str,
    concept_names: set,
    relation_names: set,
) -> list[dict]:
    """
    sections: list of {id, path, document_name, rewritten_propositions}
    Returns validated triplet records with address attached.
    """
    # Build user message
    parts = []
    for sec in sections:
        props = sec.get("rewritten_propositions") or []
        prop_lines = "\n".join(f"  - {p}" for p in props if p and p.strip())
        if prop_lines:
            parts.append(f"[{sec['id']}]\n{prop_lines}")

    if not parts:
        return []

    result = llm_call([
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": "Các phần cần trích xuất:\n\n" + "\n\n".join(parts)},
    ])

    # Index sections by id for address lookup
    sec_by_id = {s["id"]: s for s in sections}
    output = []

    for item in result.get("results", []):
        sid = item.get("section_id", "")
        sec = sec_by_id.get(sid)
        if not sec:
            continue
        address = parse_address(sec.get("path", ""), sec.get("document_name", ""))

        for t in item.get("triplets", []):
            cs  = (t.get("concept_s") or "").strip()
            rel = (t.get("relation")  or "").strip()
            co  = (t.get("concept_o") or "").strip()

            # Hard validation: all three must be exact ontology names
            if cs in concept_names and rel in relation_names and co in concept_names:
                output.append({
                    "section_id": sid,
                    "address":    address,
                    "concept_s":  cs,
                    "relation":   rel,
                    "concept_o":  co,
                })

    return output


# ── Save + dedup ──────────────────────────────────────────────────────────────

def save(path: str, data: list) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def deduplicate_triplets(triplets: list[dict]) -> list[dict]:
    """
    Merge identical (concept_s, relation, concept_o) triplets.
    Accumulate unique addresses into a list per triplet edge.
    """
    index: dict[tuple, dict] = {}
    for t in triplets:
        key = (t["concept_s"], t["relation"], t["concept_o"])
        if key not in index:
            index[key] = {
                "concept_s": t["concept_s"],
                "relation":  t["relation"],
                "concept_o": t["concept_o"],
                "addresses": [],
            }
        addr = t["address"]
        if addr not in index[key]["addresses"]:
            index[key]["addresses"].append(addr)

    return list(index.values())


# ── Main ──────────────────────────────────────────────────────────────────────

def load_sections(input_glob: str, limit: int = 0) -> list[dict]:
    files = sorted(glob.glob(input_glob))
    if not files:
        raise FileNotFoundError(f"No files matched: {input_glob}")
    sections = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            sections.extend(json.load(f))
    if limit:
        sections = sections[:limit]
    return sections


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Extract triplets using ontology")
    parser.add_argument("--input",      default=os.path.join(MATERIALS_DIR, "2_sections_rewritten_*.json"),
                        help="Glob pattern for rewritten sections JSON files")
    parser.add_argument("--ontology",   default=DEFAULT_ONTO)
    parser.add_argument("--output",     default=os.path.join(OUTPUT_DIR, "triplets.json"),
                        help="Output file for raw triplets (before dedup)")
    parser.add_argument("--output-kg",  default=os.path.join(OUTPUT_DIR, "kg_triplets.json"),
                        help="Output file after dedup (one edge, multiple addresses)")
    parser.add_argument("--threads",    type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--limit",      type=int, default=0, help="Limit sections (0 = all)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading ontology: {args.ontology}")
    ontology = load_ontology(args.ontology)
    onto_ctx, concept_names, relation_names = build_ontology_context(ontology)
    system_prompt = SYSTEM_TEMPLATE.format(onto_context=onto_ctx)
    print(f"  {len(concept_names)} concepts, {len(relation_names)} relations")

    print(f"\nLoading sections: {args.input}")
    sections = load_sections(args.input, args.limit)
    # Filter sections that have propositions and a meaningful address
    sections = [s for s in sections if s.get("rewritten_propositions") and s.get("path")]
    print(f"  {len(sections)} sections with propositions + address")

    # Resume from existing output
    done_ids: set[str] = set()
    all_raw: list[dict] = []
    if os.path.exists(args.output):
        with open(args.output, encoding="utf-8") as f:
            existing = json.load(f)
        all_raw.extend(existing)
        done_ids = {t["section_id"] for t in existing}
        print(f"  Resuming: {len(done_ids)} sections already done, {len(all_raw)} raw triplets")

    to_process = [s for s in sections if s["id"] not in done_ids]
    print(f"  Remaining: {len(to_process)} sections")

    if not to_process:
        print("Nothing to process.")
    else:
        batches = [to_process[i:i+args.batch_size] for i in range(0, len(to_process), args.batch_size)]
        lock    = threading.Lock()
        counts  = {"done": 0, "triplets": 0, "fail": 0}

        def process(batch: list[dict]) -> list[dict]:
            return extract_batch(batch, system_prompt, concept_names, relation_names)

        with ThreadPoolExecutor(max_workers=args.threads) as ex:
            futures = {ex.submit(process, b): b for b in batches}
            for fut in as_completed(futures):
                counts["done"] += 1
                try:
                    results = fut.result()
                    with lock:
                        all_raw.extend(results)
                        counts["triplets"] += len(results)
                        if counts["done"] % 5 == 0 or counts["done"] == len(batches):
                            save(args.output, all_raw)
                    print(
                        f"\r[{counts['done']}/{len(batches)}] "
                        f"triplets={counts['triplets']} fail={counts['fail']}",
                        end=""
                    )
                except Exception as e:
                    counts["fail"] += 1
                    print(f"\n[FAIL] {e}")

        save(args.output, all_raw)
        print(f"\n\nRaw triplets: {len(all_raw)} → {args.output}")

    # Deduplicate → KG edges
    kg = deduplicate_triplets(all_raw)
    save(args.output_kg, kg)
    print(f"KG edges (after dedup): {len(kg)} → {args.output_kg}")

    # Quick summary
    from collections import Counter
    rel_counts = Counter(t["relation"] for t in kg)
    print("\nTop relations:")
    for rel, cnt in rel_counts.most_common(10):
        print(f"  {rel}: {cnt} edges")


if __name__ == "__main__":
    main()
