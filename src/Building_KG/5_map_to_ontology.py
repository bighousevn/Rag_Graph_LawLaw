"""
Bước 5: Ánh xạ raw triplets → ontology triplets + gắn address (Điều/Khoản/Điểm).
LLM nhận raw triplet (subject_text, relation_text, object_text) + ontology context
→ trả về (ConceptS, Relation, ConceptO) chuẩn từ ontology.
Input:  material_for_triplets/3_triplets_extracted_*.json + ontology_168.json
Output: material_for_triplets/4_mapped_triplets_nghi_dinh_168.json
"""

import os
import re
import json
import time
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT  = os.path.join(BASE_DIR, "material_for_triplets/3_triplets_extracted_nghi_dinh_168_2024_1.json")
DEFAULT_ONTO   = os.path.join(BASE_DIR, "ontology_168.json")
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "material_for_triplets/4_mapped_triplets_nghi_dinh_168.json")


# ── Helpers ──────────────────────────────────────────────────────────────────

def parse_address(path: str, document: str) -> dict:
    """Trích Điều, Khoản, Điểm từ path string."""
    art   = re.search(r"Điều\s+(\d+)", path or "")
    cl    = re.search(r"Khoản\s+(\d+)", path or "")
    pt    = re.search(r"Điểm\s+([a-zđA-ZĐ])", path or "")
    return {
        "article":  int(art.group(1)) if art else None,
        "clause":   int(cl.group(1))  if cl  else None,
        "point":    pt.group(1).lower() if pt else None,
        "document": document,
    }


def build_ontology_context(ontology: dict) -> str:
    """Chuyển ontology JSON → chuỗi compact để đưa vào LLM prompt."""
    lines = ["=== ONTOLOGY ===", "CONCEPTS:"]
    for c in ontology["concepts"]:
        kps = ", ".join(c["keyphrases"][:8])  # giới hạn để tiết kiệm token
        lines.append(f"  [{c['name']}]: {kps}")
    lines.append("\nRELATIONS:")
    for r in ontology["relations"]:
        kps = ", ".join(r["keyphrases"][:6])
        lines.append(f"  [{r['name']}] keyphrases: {kps} | S: {r['concept_s']} | O: {', '.join(r['concept_o'][:4])}")
    return "\n".join(lines)


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
            time.sleep(backoff)
            backoff *= 2


# ── Mapping logic ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """{onto_context}

=== NHIỆM VỤ ===
Bạn nhận một danh sách raw triplets (subject, relation, object) từ văn bản pháp lý.
Với MỖI raw triplet, ánh xạ về Concept/Relation trong ontology ở trên:
- subject/object → tên CONCEPT gần nhất (phải là tên chính xác trong danh sách CONCEPTS)
- relation → tên RELATION gần nhất (phải là tên chính xác trong danh sách RELATIONS)
- Nếu subject, relation, hoặc object KHÔNG khớp với bất kỳ concept/relation nào → để null.
  Một triplet chỉ hợp lệ khi CẢ 3 đều khớp.

Trả về JSON:
{{
  "results": [
    {{"idx": 0, "concept_s": "tên concept", "relation": "tên relation", "concept_o": "tên concept"}},
    {{"idx": 1, "concept_s": null, "relation": null, "concept_o": null}}
  ]
}}
Giữ nguyên idx để ánh xạ lại với input. Luôn trả đủ tất cả idx."""


def map_batch(raw_triplets_with_meta: list[dict], onto_context: str) -> list[dict]:
    """
    raw_triplets_with_meta: list of {idx, section_id, document, path, subject, relation, object}
    Returns: list of {idx, concept_s, relation, concept_o} (null nếu không khớp)
    """
    lines = []
    for item in raw_triplets_with_meta:
        lines.append(
            f"[{item['idx']}] subject=\"{item['subject']}\" | relation=\"{item['relation']}\" | object=\"{item['object']}\""
        )

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(onto_context=onto_context)
    result = llm_call([
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": "Raw triplets cần ánh xạ:\n" + "\n".join(lines)},
    ])
    return result.get("results", [])


def save(path: str, data: list) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      default=DEFAULT_INPUT)
    parser.add_argument("--ontology",   default=DEFAULT_ONTO)
    parser.add_argument("--output",     default=DEFAULT_OUTPUT)
    parser.add_argument("--threads",    type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--limit",      type=int, default=0)
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        sections = json.load(f)
    with open(args.ontology, encoding="utf-8") as f:
        ontology = json.load(f)

    if args.limit:
        sections = sections[:args.limit]

    onto_context = build_ontology_context(ontology)
    concept_names  = {c["name"] for c in ontology["concepts"]}
    relation_names = {r["name"] for r in ontology["relations"]}

    # Phẳng hóa: mỗi raw triplet kèm section metadata
    flat: list[dict] = []
    for sec in sections:
        if not sec.get("raw_triplets"):
            continue
        address = parse_address(sec.get("path", ""), sec.get("document_name", ""))
        for t in sec["raw_triplets"]:
            subj = (t.get("subject") or "").strip()
            rel  = (t.get("relation") or "").strip()
            obj  = (t.get("object")  or "").strip()
            if subj and rel and obj:
                flat.append({
                    "section_id": sec["id"],
                    "document":   sec.get("document_name", ""),
                    "path":       sec.get("path", ""),
                    "address":    address,
                    "subject":    subj,
                    "relation":   rel,
                    "object":     obj,
                })

    print(f"Total raw triplets to map: {len(flat)}")

    # Resume: load already-mapped
    done_keys: set[tuple] = set()
    mapped_results: list[dict] = []
    if os.path.exists(args.output):
        with open(args.output, encoding="utf-8") as f:
            existing = json.load(f)
        for item in existing:
            mapped_results.append(item)
            done_keys.add((item["section_id"], item["subject_raw"], item["relation_raw"], item["object_raw"]))
        print(f"Resuming: {len(mapped_results)} already mapped.")

    to_map = [
        item for item in flat
        if (item["section_id"], item["subject"], item["relation"], item["object"]) not in done_keys
    ]
    print(f"Remaining: {len(to_map)}")
    if not to_map:
        return

    # Batch với idx toàn cục để track
    batches = []
    for i in range(0, len(to_map), args.batch_size):
        batch = to_map[i:i + args.batch_size]
        indexed = [{**item, "idx": j} for j, item in enumerate(batch)]
        batches.append(indexed)

    lock = threading.Lock()
    counts = {"ok": 0, "skip": 0, "fail": 0, "done": 0}

    def process(indexed_batch: list[dict]) -> list[dict]:
        llm_results = map_batch(indexed_batch, onto_context)
        idx_to_llm = {r["idx"]: r for r in llm_results if isinstance(r, dict)}
        output = []
        for item in indexed_batch:
            llm = idx_to_llm.get(item["idx"], {})
            cs  = llm.get("concept_s")
            rel = llm.get("relation")
            co  = llm.get("concept_o")
            # Chỉ giữ triplet hợp lệ (cả 3 đều khớp với ontology)
            if cs in concept_names and rel in relation_names and co in concept_names:
                output.append({
                    "section_id":   item["section_id"],
                    "document":     item["document"],
                    "path":         item["path"],
                    "address":      item["address"],
                    "concept_s":    cs,
                    "relation":     rel,
                    "concept_o":    co,
                    "subject_raw":  item["subject"],
                    "relation_raw": item["relation"],
                    "object_raw":   item["object"],
                })
        return output

    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futures = {ex.submit(process, b): b for b in batches}
        for fut in as_completed(futures):
            counts["done"] += 1
            try:
                results = fut.result()
                with lock:
                    mapped_results.extend(results)
                    counts["ok"] += len(results)
                    counts["skip"] += len(futures[fut]) - len(results)
                    save(args.output, mapped_results)
                print(
                    f"[{counts['done']}/{len(batches)}] "
                    f"matched={counts['ok']} skipped={counts['skip']}",
                    end="\r"
                )
            except Exception as e:
                counts["fail"] += len(futures[fut])
                print(f"\n[FAIL] batch: {e}")

    print(f"\nDone. matched={counts['ok']} skipped={counts['skip']} fail={counts['fail']} → {args.output}")


if __name__ == "__main__":
    main()
