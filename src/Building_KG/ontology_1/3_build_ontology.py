"""
Bước 3 (pass 3/3): Xây ontology (Concept/Relation) từ TOÀN BỘ triplet đã rà atomic ở Bước 2,
và xuất triplet cuối cùng kèm địa chỉ điều khoản.

Không còn bước "đối chiếu câu gốc để lấy keyphrase" riêng — Bước 1 đã chuẩn hoá s/v/o về tên
canonical rồi, nên keyphrase của mỗi Concept/Relation trong ontology cuối chính là tập hợp các
tên canonical khác nhau (đã lowercase) mà nhiều section dùng để chỉ cùng 1 thực thể/hành vi, gộp
lại qua bước merge dưới đây. Đơn giản hơn nhưng vẫn giữ đúng phần cốt lõi: merge synonym có
partition theo concept_s (tránh gộp nhầm relation của 2 chủ thể khác nhau) + safety-net không để
concept/relation nào bị LLM âm thầm làm rớt khi merge (từng xảy ra thật với các concept ngưỡng số
liệu như "50mg/100ml máu").

Input:  output/triplets_refined.json   (từ 2_refine_atomicity.py, toàn bộ section)
Output: output/ontology.json      {"concepts": [...], "relations": [...]}  — format CLAUDE.md
        output/triplets.json      mỗi triplet kèm address (article/clause/point/document)
        output/kg_triplets.json   dedup theo (concept_s, relation, concept_o), gộp addresses
"""

import os
import re
import sys
import json
import time
import argparse
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR     = os.path.join(BASE_DIR, "output")
DEFAULT_INPUT  = os.path.join(OUTPUT_DIR, "triplets_refined.json")
OUT_ONTOLOGY   = os.path.join(OUTPUT_DIR, "ontology.json")
OUT_TRIPLETS   = os.path.join(OUTPUT_DIR, "triplets.json")
OUT_KG         = os.path.join(OUTPUT_DIR, "kg_triplets.json")

MERGE_CHUNK_SIZE = 120   # nếu candidates > ngưỡng này, pre-merge theo chunk trước khi merge cuối


def log(msg: str) -> None:
    print(msg, flush=True)


# ── Prompts ───────────────────────────────────────────────────────────────────

CONCEPT_MERGE_SYSTEM = """\
Bạn là chuyên gia xây dựng ontology pháp lý giao thông đường bộ Việt Nam.

Dưới đây là danh sách concept candidates (tên canonical Title Case) được trích ra từ nhiều section
riêng lẻ trên toàn văn bản. Một số tên tuy viết khác nhau nhưng CÙNG chỉ 1 loại thực thể (ví dụ
"Ô tô" và "Xe ô tô" do 2 section khác nhau đặt tên hơi khác nhau).

Nhiệm vụ: hợp nhất các tên ĐỒNG NGHĨA (chỉ cùng 1 thực thể) thành 1 concept — chọn 1 tên chuẩn làm
"name", liệt kê "source_names" = tất cả tên gốc đã gộp (kể cả khi giữ nguyên 1 tên, source_names
vẫn phải chứa tên đó), và "keyphrases" = source_names viết thường (lowercase), loại trùng.

BẮT BUỘC: TUYỆT ĐỐI không được bỏ sót bất kỳ tên nào trong input — mọi tên đầu vào phải xuất hiện
trong "source_names" của đúng 1 entry output. Kể cả tên là GIÁ TRỊ SỐ LIỆU/NGƯỠNG ĐO LƯỜNG cụ thể
(vd "50mg/100ml máu", "0,4mg/1L khí thở", "20 km/h đến 35 km/h") vẫn PHẢI giữ lại làm 1 concept
riêng (KHÔNG gộp các ngưỡng khác nhau vào nhau, KHÔNG bỏ qua vì "chỉ là số liệu").

Tên concept phải là danh từ NGUYÊN TỬ — không chứa động từ.

Trả về JSON:
{"concepts": [
  {"name": "Ô tô", "keyphrases": ["ô tô", "xe ô tô"], "source_names": ["Ô Tô", "Xe Ô Tô"]},
  {"name": "Người", "keyphrases": ["người"], "source_names": ["Người"]}
]}"""


def _build_relation_merge_system(concept_names: list[str], concept_s: str) -> str:
    names_str = "\n".join(f"  - {n}" for n in sorted(concept_names))
    return f"""\
Bạn là chuyên gia xây dựng ontology pháp lý giao thông đường bộ Việt Nam.

CONCEPTS hợp lệ (concept_o CHỈ được dùng tên chính xác từ danh sách này):
{names_str}

Dưới đây là các relation candidate đã có concept_s/concept_o từ nhiều section riêng lẻ. TẤT CẢ
đều có CÙNG concept_s = "{concept_s}". Có thể có relation ĐỒNG NGHĨA bị đặt tên hơi khác nhau
(cùng bản chất hành vi, concept_o trùng hoặc gần trùng).

QUAN TRỌNG: PHỦ ĐỊNH tạo ra relation KHÁC HẲN dạng khẳng định — "Không nhường" và "Nhường" là 2
relation khác nhau, TUYỆT ĐỐI không được gộp chung. Chỉ gộp khi cùng bản chất hành vi VÀ cùng cực
tính (cùng khẳng định hoặc cùng phủ định).

Nhiệm vụ: hợp nhất relation ĐỒNG NGHĨA thành 1 entry — chọn 1 tên chuẩn "name", "keyphrases" =
các tên gốc viết thường (lowercase), "source_names" = TẤT CẢ tên relation gốc đã gộp (kể cả khi
giữ nguyên 1 tên vẫn phải liệt kê), "concept_o" = hợp nhất mọi concept_o (loại trùng). Relation ý
nghĩa khác nhau dù chung concept_o thì GIỮ RIÊNG, không gộp. concept_s trả về PHẢI luôn là
"{concept_s}".

BẮT BUỘC: không được bỏ sót bất kỳ relation nào trong input — mọi "name" đầu vào phải nằm trong
"source_names" của đúng 1 entry output.

Trả về JSON:
{{"relations": [
  {{"name": "Điều khiển", "keyphrases": ["điều khiển"], "source_names": ["Điều Khiển"],
    "concept_s": "{concept_s}", "concept_o": ["Ô tô", "Xe máy"]}}
]}}"""


# ── LLM + IO helpers ────────────────────────────────────────────────────────

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
            content = resp.choices[0].message.content
            if content is None:
                raise ValueError("empty LLM response content")
            return json.loads(content)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            log(f"    [retry {attempt+1}] {e}")
            time.sleep(backoff)
            backoff *= 2
    raise RuntimeError("llm_call: unreachable")


def save_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# ── Step A: gom toàn bộ triplet thành candidate concepts/relations ───────────

def build_candidates(sections: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    concepts:  [{name, keyphrases}]      — mọi tên s/o xuất hiện, keyphrase khởi tạo = [tên lowercase]
    relations: [{name, keyphrases, concept_s, concept_o: [...]}]  — nhóm theo (v, s)
    """
    concept_names: dict[str, None] = {}
    relation_agg: dict[tuple[str, str], dict] = {}

    for sec in sections:
        for t in sec.get("triplets", []):
            s, v, o = (t.get("s") or "").strip(), (t.get("v") or "").strip(), (t.get("o") or "").strip()
            if not (s and v and o):
                continue
            concept_names.setdefault(s, None)
            concept_names.setdefault(o, None)

            entry = relation_agg.setdefault((v, s), {"concept_o": []})
            if o not in entry["concept_o"]:
                entry["concept_o"].append(o)

    concepts = [{"name": n, "keyphrases": [n.lower()]} for n in concept_names]
    relations = [
        {"name": v, "keyphrases": [v.lower()], "concept_s": s, "concept_o": e["concept_o"]}
        for (v, s), e in relation_agg.items()
    ]
    return concepts, relations


# ── Step B: merge synonym toàn văn bản ────────────────────────────────────────

def _merge_concepts_call(items: list[dict]) -> list[dict]:
    payload = json.dumps({"concepts": items}, ensure_ascii=False)
    result = llm_call([
        {"role": "system", "content": CONCEPT_MERGE_SYSTEM},
        {"role": "user",   "content": f"Concepts:\n{payload}"},
    ])
    return result.get("concepts", []) or []


def merge_concepts(concepts: list[dict]) -> tuple[list[dict], dict[str, str]]:
    """Trả về (final_concepts, rename_map: tên_gốc -> tên_final). Không bao giờ làm rớt concept:
    bất kỳ tên gốc nào LLM không echo lại trong source_names sẽ được thêm lại làm entry riêng."""
    items = [{"name": c["name"], "keyphrases": c["keyphrases"]} for c in concepts]

    if len(items) > MERGE_CHUNK_SIZE:
        chunks = [items[i:i+MERGE_CHUNK_SIZE] for i in range(0, len(items), MERGE_CHUNK_SIZE)]
        log(f"  [concepts] {len(items)} candidates → pre-merge {len(chunks)} chunk(s)...")
        pre_merged = []
        for ci, chunk in enumerate(chunks, 1):
            out = _merge_concepts_call(chunk)
            pre_merged.extend(out)
            log(f"    chunk [{ci}/{len(chunks)}]: {len(chunk)} → {len(out)}")
        items = [{"name": c["name"], "keyphrases": c.get("keyphrases", [])} for c in pre_merged]

    log(f"  [concepts] final merge on {len(items)} candidates...")
    final_concepts = _merge_concepts_call(items) if items else []

    rename_map: dict[str, str] = {}
    for c in final_concepts:
        for src in c.get("source_names", [c["name"]]):
            rename_map[src] = c["name"]

    # safety net: không để LLM âm thầm làm rớt 1 concept (đã xảy ra thật với concept ngưỡng số liệu)
    final_by_name = {c["name"]: c for c in final_concepts}
    for c in concepts:
        target = rename_map.setdefault(c["name"], c["name"])
        if target not in final_by_name:
            recovered = {"name": target, "keyphrases": list(c["keyphrases"])}
            final_concepts.append(recovered)
            final_by_name[target] = recovered
        else:
            bucket = final_by_name[target].setdefault("keyphrases", [])
            for kp in c["keyphrases"]:
                if kp not in bucket:
                    bucket.append(kp)

    log(f"  [concepts] ✓ {len(final_concepts)} final concepts")
    return final_concepts, rename_map


def merge_relations(
    relations: list[dict], concept_rename_map: dict[str, str], concept_names: set[str]
) -> tuple[list[dict], dict[tuple[str, str], str]]:
    """Trả về (final_relations, rename_map: (concept_s_final, tên_v_gốc) -> tên_v_final)."""
    renamed = []
    for r in relations:
        cs = concept_rename_map.get(r["concept_s"], r["concept_s"])
        co = list(dict.fromkeys(concept_rename_map.get(o, o) for o in r["concept_o"]))
        renamed.append({"name": r["name"], "keyphrases": r["keyphrases"], "concept_s": cs, "concept_o": co})

    groups: dict[str, list[dict]] = {}
    for r in renamed:
        groups.setdefault(r["concept_s"], []).append(r)

    log(f"  [relations] {len(renamed)} candidates, {len(groups)} concept_s group(s)...")
    final_relations: list[dict] = []
    rename_map: dict[tuple[str, str], str] = {}

    for gi, (cs, group) in enumerate(groups.items(), 1):
        if len(group) <= 1:
            merged = [dict(r, source_names=[r["name"]]) for r in group]
        else:
            system = _build_relation_merge_system(sorted(concept_names), cs)
            if len(group) > MERGE_CHUNK_SIZE:
                chunks = [group[i:i+MERGE_CHUNK_SIZE] for i in range(0, len(group), MERGE_CHUNK_SIZE)]
                pre = []
                for ci, chunk in enumerate(chunks, 1):
                    payload = json.dumps({"relations": chunk}, ensure_ascii=False)
                    out = llm_call([
                        {"role": "system", "content": system},
                        {"role": "user",   "content": f"Relations:\n{payload}"},
                    ]).get("relations", []) or chunk
                    pre.extend(out)
                    log(f"    [{cs}] chunk [{ci}/{len(chunks)}]: {len(chunk)} → {len(out)}")
                group = pre
            payload = json.dumps({"relations": group}, ensure_ascii=False)
            merged = llm_call([
                {"role": "system", "content": system},
                {"role": "user",   "content": f"Relations:\n{payload}"},
            ]).get("relations", []) or group
            merged = [dict(r, concept_s=cs, source_names=r.get("source_names") or [r["name"]]) for r in merged]

        # safety net: mọi tên v gốc trong group phải map được sang 1 final relation name
        merged_by_name = {r["name"]: r for r in merged}
        for r in group:
            target = None
            for m in merged:
                if r["name"] in m.get("source_names", [m["name"]]):
                    target = m["name"]
                    break
            if target is None:
                merged.append(dict(r, source_names=[r["name"]]))
                merged_by_name[r["name"]] = merged[-1]
                target = r["name"]
            rename_map[(cs, r["name"])] = target

        final_relations.extend(merged)
        log(f"    [{gi}/{len(groups)}] {cs}: {len(group)} → {len(merged)} relations")

    return final_relations, rename_map


# ── Step C: dựng triplet cuối + address ───────────────────────────────────────

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


def reconstruct_triplets(
    sections: list[dict],
    concept_rename_map: dict[str, str],
    relation_rename_map: dict[tuple[str, str], str],
) -> list[dict]:
    out = []
    for sec in sections:
        address = parse_address(sec.get("path", ""), sec.get("document_name", ""))
        for t in sec.get("triplets", []):
            s, v, o = (t.get("s") or "").strip(), (t.get("v") or "").strip(), (t.get("o") or "").strip()
            if not (s and v and o):
                continue
            s_final = concept_rename_map.get(s, s)
            o_final = concept_rename_map.get(o, o)
            v_final = relation_rename_map.get((s_final, v), v)
            out.append({
                "section_id": sec["id"],
                "address":    address,
                "concept_s":  s_final,
                "relation":   v_final,
                "concept_o":  o_final,
            })
    return out


def deduplicate_triplets(triplets: list[dict]) -> list[dict]:
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
        if t["address"] not in index[key]["addresses"]:
            index[key]["addresses"].append(t["address"])
    return list(index.values())


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Bước 3: xây ontology từ toàn bộ triplet + xuất triplet cuối")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--limit", type=int, default=0, help="0 = xử lý tất cả section")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        sections = json.load(f)
    if args.limit:
        sections = sections[:args.limit]
    log(f"Loaded {len(sections)} sections from {args.input}")

    concepts, relations = build_candidates(sections)
    log(f"Raw candidates: {len(concepts)} concepts, {len(relations)} relations")

    final_concepts, concept_rename_map = merge_concepts(concepts)
    concept_names = {c["name"] for c in final_concepts}

    final_relations, relation_rename_map = merge_relations(relations, concept_rename_map, concept_names)

    ontology = {
        "concepts":  [{"name": c["name"], "keyphrases": c.get("keyphrases", [])} for c in final_concepts],
        "relations": [
            {"name": r["name"], "keyphrases": r.get("keyphrases", []),
             "concept_s": r["concept_s"], "concept_o": r.get("concept_o", [])}
            for r in final_relations
        ],
    }
    save_json(OUT_ONTOLOGY, ontology)
    log(f"\nOntology → {OUT_ONTOLOGY} ({len(ontology['concepts'])} concepts, {len(ontology['relations'])} relations)")

    triplets = reconstruct_triplets(sections, concept_rename_map, relation_rename_map)
    save_json(OUT_TRIPLETS, triplets)
    log(f"Triplets (per-section) → {OUT_TRIPLETS} ({len(triplets)} triplets)")

    kg = deduplicate_triplets(triplets)
    save_json(OUT_KG, kg)
    log(f"KG edges (deduped) → {OUT_KG} ({len(kg)} edges)")

    from collections import Counter
    rel_counts = Counter(t["relation"] for t in kg)
    log("\nTop relations:")
    for rel, cnt in rel_counts.most_common(10):
        log(f"  {rel}: {cnt} edges")


if __name__ == "__main__":
    main()
