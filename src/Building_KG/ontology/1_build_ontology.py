"""
Phase 1: Build draft ontology from law text using LLM.

Run ONCE per new law document. Expert must review output before using in Phase 2.

Input:  ../material_for_triplets/2_sections_rewritten_*.json
Output:
  output/discovery_raw.json   ← accumulated per-batch results (auto-saved, resumable)
  output/ontology_draft.json  ← final merged ontology

Two-pass strategy:
  Pass 1 (discovery): Each batch of propositions → LLM extracts candidate entities + relations
                      Results saved after EVERY batch → resumable on crash
  Pass 2 (merge):     All candidates → LLM merges synonyms, assigns concept_s/concept_o

Grounding: LLM is asked to group synonyms and name canonically — expert reviews before use.
"""

import os
import json
import time
import argparse
import glob
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
MATERIALS_DIR     = os.path.join(BASE_DIR, "..", "material_for_triplets")
OUTPUT_DIR        = os.path.join(BASE_DIR, "output")
DISCOVERY_RAW     = os.path.join(OUTPUT_DIR, "discovery_raw.json")
DEFAULT_OUTPUT    = os.path.join(OUTPUT_DIR, "ontology_draft.json")

DISCOVERY_BATCH   = 30    # propositions per discovery call
MAX_CANDIDATES    = 350   # max candidates fed to merge (stay within token budget)
MERGE_CHUNK_SIZE  = 150   # if candidates > this, merge in chunks first


# ── Prompts ───────────────────────────────────────────────────────────────────

DISCOVERY_SYSTEM = """\
Bạn là chuyên gia xây dựng ontology cho lĩnh vực pháp lý giao thông đường bộ Việt Nam.

Từ các mệnh đề pháp lý dưới đây, hãy xác định:

1. ENTITIES — Danh từ/cụm danh từ đóng vai trò CHỦ THỂ hoặc ĐỐI TƯỢNG trong hành vi pháp lý.
   Nhóm các cách diễn đạt ĐỒNG NGHĨA về cùng một loại thực thể → một entry duy nhất.
   Ví dụ: "xe ô tô", "ô tô", "xe chở người bốn bánh có gắn động cơ", "xe bốn bánh", "xe hơi"
           → TẤT CẢ là keyphrases của entity canonical "Ô tô"
   Bỏ qua: con số tiền phạt, số ngày, số lần cụ thể (không phải loại thực thể).

   QUAN TRỌNG — Subject entity phải là danh từ NGUYÊN TỬ, không chứa động từ:
   - "Người điều khiển xe ô tô" → KHÔNG phải 1 entity. Tách thành:
       entity "Người" (subject) + relation "Điều_khiển" + entity "Ô tô" (object)
   - "Người điều khiển xe mô tô" → tương tự: "Người" + "Điều_khiển" + "Xe máy"
   - Subject đúng: "Người", "Người đi bộ", "Cá nhân", "Tổ chức"
   - "tài xế", "lái xe", "người điều khiển" là keyphrases của entity "Người", KHÔNG phải tên entity riêng

2. RELATIONS — Động từ/cụm động từ thể hiện HÀNH VI cốt lõi.
   Nhóm các động từ đồng nghĩa vào cùng một relation.
   Bỏ qua: trạng từ điều kiện, mức độ cụ thể.

Quy tắc về keyphrases:
- Thu thập ĐẦY ĐỦ mọi cách viết xuất hiện trong văn bản, dù dài hay ngắn
- KHÔNG lọc bỏ keyphrase vì "dài dòng" — keyphrase dài vẫn cần để nhận diện văn bản luật
- Mỗi entity/relation phải có ít nhất 2-3 keyphrases nếu có nhiều cách gọi trong đoạn văn

Trả về JSON:
{
  "entities": [
    {
      "canonical": "Ô tô",
      "keyphrases": ["xe ô tô", "ô tô", "xe hơi", "xe chở người bốn bánh có gắn động cơ",
                     "xe chở hàng bốn bánh có gắn động cơ", "xe bốn bánh", "ô tô tải", "xe khách"],
      "role": "object"
    },
    {
      "canonical": "Người",
      "keyphrases": ["người", "người tham gia giao thông", "tài xế", "lái xe", "người điều khiển",
                     "người điều khiển phương tiện", "người điều khiển xe ô tô",
                     "người điều khiển xe mô tô"],
      "role": "subject"
    }
  ],
  "relations": [
    {
      "canonical": "Điều_khiển",
      "keyphrases": ["điều khiển", "lái", "vận hành", "cầm lái", "lái xe", "chạy xe"],
      "subject_examples": ["người"],
      "object_examples": ["xe ô tô", "xe mô tô", "xe gắn máy"]
    }
  ]
}"""

MERGE_ENTITY_SYSTEM = """\
Bạn là chuyên gia xây dựng ontology pháp lý.

Dưới đây là entity candidates từ nhiều batch phân tích, có thể có trùng lặp và đồng nghĩa.

Nhiệm vụ: Hợp nhất các entities đồng nghĩa/trùng ý → một entry với tên chuẩn.
Gộp TẤT CẢ keyphrases từ mọi entry đồng nghĩa vào một danh sách, loại trùng lặp.
Tên canonical: danh từ chuẩn tiếng Việt, KHÔNG dùng dấu gạch dưới.

QUAN TRỌNG — Tên entity phải là danh từ NGUYÊN TỬ, không chứa động từ:
- Sai: "Người điều khiển phương tiện"  →  Đúng: "Người"
- Sai: "Người điều khiển xe ô tô"      →  Đúng: "Người"
- Sai: "Người lái xe mô tô"            →  Đúng: "Người"
- "tài xế", "lái xe", "người điều khiển", "người điều khiển phương tiện",
  "người điều khiển xe ô tô", "người điều khiển xe mô tô" đều là keyphrases của "Người"
- Khi gặp entity dạng "Người điều khiển [loại xe]": gộp vào "Người", loại xe giữ riêng.

Trả về JSON:
{"concepts": [
  {"name": "Người", "keyphrases": ["người", "tài xế", "lái xe", "người điều khiển",
                                   "người điều khiển phương tiện", "người tham gia giao thông"]},
  {"name": "Xe ô tô", "keyphrases": ["xe ô tô", "ô tô", "xe hơi", "xe bốn bánh"]},
  {"name": "Xe máy",  "keyphrases": ["xe mô tô", "xe gắn máy", "xe máy", "mô tô"]}
]}"""

MERGE_CHUNK_ENTITY_SYSTEM = """\
Bạn là chuyên gia xây dựng ontology pháp lý.
Hợp nhất các entity entries đồng nghĩa/trùng ý. Gộp keyphrases, giữ canonical tốt nhất.

QUAN TRỌNG — Tên entity phải là danh từ NGUYÊN TỬ, không chứa động từ:
- Sai: "Người điều khiển phương tiện"  →  Đúng: "Người"
- Sai: "Người điều khiển xe ô tô"      →  Đúng: "Người"
- "tài xế", "lái xe", "người điều khiển" đều là keyphrases của "Người"
- Khi gặp entity dạng "Người điều khiển [loại xe]": gộp vào "Người", loại xe giữ riêng.

Trả về JSON: {"entities": [{"canonical": "...", "keyphrases": [...], "role": "..."}, ...]}"""

MERGE_CHUNK_RELATION_SYSTEM = """\
Bạn là chuyên gia xây dựng ontology pháp lý.
Hợp nhất các relation entries đồng nghĩa/trùng ý. Gộp keyphrases, subject_examples, object_examples.
CHƯA cần xác định concept_s/concept_o — chỉ gộp synonyms và keyphrases.

Trả về JSON: {"relations": [{"canonical": "...", "keyphrases": [...], "subject_examples": [...], "object_examples": [...]}, ...]}"""


def _build_relation_merge_system(concept_names: list[str]) -> str:
    """Build relation merge prompt with injected concept names so LLM picks from real list."""
    names_str = "\n".join(f"  - {n}" for n in sorted(concept_names))
    return f"""\
Bạn là chuyên gia xây dựng ontology pháp lý.

CONCEPTS đã hợp nhất (chỉ dùng tên CHÍNH XÁC từ danh sách này làm concept_s/concept_o):
{names_str}

Dưới đây là relation candidates từ nhiều batch, có thể có trùng lặp và đồng nghĩa.

Nhiệm vụ:
1. Hợp nhất các relations đồng nghĩa/trùng ý → một entry với tên chuẩn.
   Gộp keyphrases, subject_examples, object_examples.
2. Với mỗi relation đã hợp nhất, xác định:
   - concept_s: MỘT TÊN từ danh sách CONCEPTS ở trên (dựa vào subject_examples)
   - concept_o: DANH SÁCH TÊN từ danh sách CONCEPTS ở trên (dựa vào object_examples)
   → CHỈ dùng tên trong danh sách CONCEPTS, KHÔNG đặt tên ngoài danh sách

Nguyên tắc:
- Mỗi relation có ĐÚNG MỘT concept_s
- Nếu cùng hành vi có nhiều chủ thể rất khác nhau → tạo 2 relations riêng (tên khác nhau)
- Tên canonical relation: dùng dấu gạch dưới (ví dụ "Điều_khiển", "Sử_dụng")

QUAN TRỌNG — concept_s phải là danh từ NGUYÊN TỬ từ danh sách CONCEPTS:
- subject_examples "người điều khiển xe ô tô", "tài xế", "lái xe" → concept_s = "Người"
- "người điều khiển phương tiện" → concept_s = "Người"
- KHÔNG dùng tên không có trong danh sách CONCEPTS ở trên

Trả về JSON:
{{"relations": [
  {{"name": "Điều_khiển", "keyphrases": ["điều khiển", "lái", "vận hành", "cầm lái"],
    "concept_s": "Người", "concept_o": ["Xe ô tô", "Xe máy"]}}
]}}"""


# ── LLM helpers ───────────────────────────────────────────────────────────────

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
            print(f"  [retry {attempt+1}] {e}")
            time.sleep(backoff)
            backoff *= 2
    return {}


def save_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# ── Pass 1: Discovery ─────────────────────────────────────────────────────────

def discovery_pass(
    propositions: list[str],
    batch_size: int = DISCOVERY_BATCH,
    raw_path: str = DISCOVERY_RAW,
) -> tuple[list, list]:
    """
    Feed propositions in batches → collect candidate entities + relations.
    Saves progress after each batch → resumable on crash.
    """
    # Load existing progress
    all_entities: list[dict] = []
    all_relations: list[dict] = []
    props_done = 0

    if os.path.exists(raw_path):
        with open(raw_path, encoding="utf-8") as f:
            saved = json.load(f)
        all_entities  = saved.get("entities", [])
        all_relations = saved.get("relations", [])
        props_done    = saved.get("props_done", 0)
        print(f"  [resume] loaded {len(all_entities)} entities, {len(all_relations)} relations"
              f" ({props_done}/{len(propositions)} props already done)")

    remaining   = propositions[props_done:]
    batches     = [remaining[i:i+batch_size] for i in range(0, len(remaining), batch_size)]
    total_batches = (len(propositions) + batch_size - 1) // batch_size
    done_batches  = props_done // batch_size

    print(f"Discovery: {len(propositions)} props total | "
          f"{len(remaining)} remaining | {len(batches)} batches left")

    for i, batch in enumerate(batches, 1):
        batch_num = done_batches + i
        numbered  = "\n".join(f"{j+1}. {p}" for j, p in enumerate(batch))

        try:
            result = llm_call([
                {"role": "system", "content": DISCOVERY_SYSTEM},
                {"role": "user",   "content": f"Mệnh đề pháp lý:\n{numbered}"},
            ])
        except Exception as e:
            print(f"  [batch {batch_num}/{total_batches}] LLM FAIL: {e} — skipping")
            continue

        batch_entities  = result.get("entities",  []) or []
        batch_relations = result.get("relations", []) or []
        all_entities.extend(batch_entities)
        all_relations.extend(batch_relations)
        props_done += len(batch)

        # Print batch detail
        print(f"  [{batch_num}/{total_batches}] "
              f"+{len(batch_entities)} entities, +{len(batch_relations)} relations  "
              f"(cumulative: {len(all_entities)}, {len(all_relations)})")
        if batch_entities:
            names = [e.get("canonical", "?") for e in batch_entities]
            print(f"    entities : {', '.join(names)}")
        if batch_relations:
            names = [r.get("canonical", "?") for r in batch_relations]
            print(f"    relations: {', '.join(names)}")

        # Save after every batch
        save_json(raw_path, {
            "props_done": props_done,
            "total_props": len(propositions),
            "entities":   all_entities,
            "relations":  all_relations,
        })

    return all_entities, all_relations


# ── Pass 2: Merge ─────────────────────────────────────────────────────────────

def _pre_merge_entities(items: list[dict]) -> list[dict]:
    """Pre-merge entity candidates in chunks (no concept_s/concept_o at this stage)."""
    chunks = [items[i:i+MERGE_CHUNK_SIZE] for i in range(0, len(items), MERGE_CHUNK_SIZE)]
    merged = []
    for ci, chunk in enumerate(chunks, 1):
        payload = json.dumps({"entities": chunk}, ensure_ascii=False)
        result = llm_call([
            {"role": "system", "content": MERGE_CHUNK_ENTITY_SYSTEM},
            {"role": "user",   "content": f"Hợp nhất entities:\n{payload}"},
        ])
        chunk_out = result.get("entities", []) or []
        merged.extend(chunk_out)
        print(f"    entity chunk [{ci}/{len(chunks)}]: {len(chunk)} → {len(chunk_out)}")
    return merged


def _pre_merge_relations(items: list[dict]) -> list[dict]:
    """Pre-merge relation candidates in chunks (no concept_s/concept_o assignment yet)."""
    chunks = [items[i:i+MERGE_CHUNK_SIZE] for i in range(0, len(items), MERGE_CHUNK_SIZE)]
    merged = []
    for ci, chunk in enumerate(chunks, 1):
        payload = json.dumps({"relations": chunk}, ensure_ascii=False)
        result = llm_call([
            {"role": "system", "content": MERGE_CHUNK_RELATION_SYSTEM},
            {"role": "user",   "content": f"Hợp nhất relations:\n{payload}"},
        ])
        chunk_out = result.get("relations", []) or []
        merged.extend(chunk_out)
        print(f"    relation chunk [{ci}/{len(chunks)}]: {len(chunk)} → {len(chunk_out)}")
    return merged


def _post_validate_relations(relations: list[dict], concept_set: set) -> list[dict]:
    """
    Post-validate after relation merge:
    - concept_s not in concept_set → auto-fix compound NP to "Người", else drop
    - concept_o items not in concept_set → remove invalid entries
    """
    _NP_PREFIXES = ("người điều khiển", "người lái", "tài xế")
    validated = []
    for r in relations:
        cs = r.get("concept_s", "")
        if cs not in concept_set:
            if any(cs.lower().startswith(p) for p in _NP_PREFIXES):
                print(f"    [auto-fix] '{r.get('name')}': concept_s '{cs}' → 'Người'")
                r = dict(r, concept_s="Người")
            else:
                print(f"    [drop] '{r.get('name')}': concept_s '{cs}' not in concepts")
                continue
        co_raw   = r.get("concept_o", [])
        co_valid = [c for c in co_raw if c in concept_set]
        co_bad   = [c for c in co_raw if c not in concept_set]
        if co_bad:
            print(f"    [warn] '{r.get('name')}': removed invalid concept_o: {co_bad}")
        validated.append(dict(r, concept_o=co_valid))
    return validated


def merge_pass(entities: list[dict], relations: list[dict]) -> dict:
    """
    Two-step merge following Hướng 2 (atomic noun ConceptS):
      Step 1: Merge entities independently → final concepts list
      Step 2: Merge relations with injected concept names → concept_s/concept_o from real names
      Step 3: Post-validate relations (auto-fix compound NPs, remove invalid references)
    """
    print(f"\n{'─'*60}")
    print(f"Merge pass: {len(entities)} entity candidates, {len(relations)} relation candidates")

    # ── Step 1: Merge entities ────────────────────────────────────────────────
    if len(entities) > MERGE_CHUNK_SIZE:
        print(f"\n  [Step 1a] pre-merging {len(entities)} entities in chunks...")
        entities = _pre_merge_entities(entities)
        print(f"  after pre-merge: {len(entities)} entity candidates")

    if len(entities) > MAX_CANDIDATES:
        print(f"  ⚠ truncating entities {len(entities)} → {MAX_CANDIDATES}")
        entities = entities[:MAX_CANDIDATES]

    entity_payload = json.dumps({"entities": entities}, ensure_ascii=False)
    print(f"\n  [Step 1] merging {len(entities)} entities ({len(entity_payload.encode())/1024:.1f} KB)...")

    entity_result = llm_call([
        {"role": "system", "content": MERGE_ENTITY_SYSTEM},
        {"role": "user",   "content": f"Entity candidates:\n{entity_payload}"},
    ])
    concepts = entity_result.get("concepts", []) or []
    if not concepts:
        print("  ⚠ WARNING: entity merge returned 0 concepts")
    concept_set = {c["name"] for c in concepts}
    print(f"  ✓ {len(concepts)} concepts: {', '.join(sorted(concept_set))}")

    # ── Step 2: Merge relations with concept context ──────────────────────────
    if len(relations) > MERGE_CHUNK_SIZE:
        print(f"\n  [Step 2a] pre-merging {len(relations)} relations in chunks...")
        relations = _pre_merge_relations(relations)
        print(f"  after pre-merge: {len(relations)} relation candidates")

    if len(relations) > MAX_CANDIDATES:
        print(f"  ⚠ truncating relations {len(relations)} → {MAX_CANDIDATES}")
        relations = relations[:MAX_CANDIDATES]

    relation_system  = _build_relation_merge_system(list(concept_set))
    relation_payload = json.dumps({"relations": relations}, ensure_ascii=False)
    print(f"\n  [Step 2] merging {len(relations)} relations ({len(relation_payload.encode())/1024:.1f} KB)...")

    relation_result = llm_call([
        {"role": "system", "content": relation_system},
        {"role": "user",   "content": f"Relation candidates:\n{relation_payload}"},
    ])
    rels_out = relation_result.get("relations", []) or []
    if not rels_out:
        print("  ⚠ WARNING: relation merge returned 0 relations")

    # ── Step 3: Post-validate ─────────────────────────────────────────────────
    print(f"\n  [Step 3] validating {len(rels_out)} relations against {len(concept_set)} concepts...")
    rels_out = _post_validate_relations(rels_out, concept_set)

    # Print results
    print(f"\n  ✓ Final: {len(concepts)} concepts, {len(rels_out)} relations")
    print(f"\n  CONCEPTS:")
    for c in concepts:
        kps = c.get("keyphrases", [])
        print(f"    [{c.get('name', '?')}]  ({len(kps)} keyphrases)")
        print(f"      {', '.join(kps[:6])}{'...' if len(kps) > 6 else ''}")

    print(f"\n  RELATIONS:")
    for r in rels_out:
        cs  = r.get("concept_s", "?")
        co  = r.get("concept_o", [])
        kps = r.get("keyphrases", [])
        print(f"    [{r.get('name', '?')}]  S={cs}  O=[{', '.join(co[:4])}{'...' if len(co)>4 else ''}]")
        print(f"      keyphrases: {', '.join(kps[:5])}{'...' if len(kps)>5 else ''}")

    return {"concepts": concepts, "relations": rels_out}


# ── Main ──────────────────────────────────────────────────────────────────────

def collect_propositions(input_glob: str) -> list[str]:
    files = sorted(glob.glob(input_glob))
    if not files:
        raise FileNotFoundError(f"No files matched: {input_glob}")
    props = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            sections = json.load(f)
        for sec in sections:
            for p in sec.get("rewritten_propositions", []):
                if p and p.strip():
                    props.append(p.strip())
    return props


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Build draft ontology from law text")
    parser.add_argument("--input",       default=os.path.join(MATERIALS_DIR, "2_sections_rewritten_*.json"),
                        help="Glob pattern for rewritten sections JSON files")
    parser.add_argument("--output",      default=DEFAULT_OUTPUT,
                        help="Final merged ontology output path")
    parser.add_argument("--raw",         default=DISCOVERY_RAW,
                        help="Per-batch discovery cache (auto-saved, used for resume)")
    parser.add_argument("--batch-size",  type=int, default=DISCOVERY_BATCH)
    parser.add_argument("--limit-props", type=int, default=0,
                        help="Limit number of propositions (0 = all, useful for testing)")
    parser.add_argument("--merge-only",  action="store_true",
                        help="Skip discovery, run merge on existing --raw file")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.merge_only:
        if not os.path.exists(args.raw):
            print(f"ERROR: --merge-only requires existing raw file: {args.raw}")
            return
        with open(args.raw, encoding="utf-8") as f:
            saved = json.load(f)
        entities  = saved.get("entities",  [])
        relations = saved.get("relations", [])
        print(f"Loaded from {args.raw}: {len(entities)} entities, {len(relations)} relations")
    else:
        print(f"Collecting propositions from: {args.input}")
        propositions = collect_propositions(args.input)
        if args.limit_props:
            propositions = propositions[:args.limit_props]
        print(f"Total propositions: {len(propositions)}")
        print(f"Discovery raw cache: {args.raw}\n")

        entities, relations = discovery_pass(
            propositions, batch_size=args.batch_size, raw_path=args.raw
        )

    print(f"\n{'─'*60}")
    print("Running merge pass...")
    ontology = merge_pass(entities, relations)

    save_json(args.output, ontology)
    n_c = len(ontology.get("concepts", []))
    n_r = len(ontology.get("relations", []))
    print(f"\n{'─'*60}")
    print(f"Done: {n_c} concepts, {n_r} relations → {args.output}")
    print("⚠  Expert review required before using in Phase 2 (2_extract_triplets.py)")
    print(f"   Discovery raw cache kept at: {args.raw}")


if __name__ == "__main__":
    main()
