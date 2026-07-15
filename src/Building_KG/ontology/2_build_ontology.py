"""
Gộp kết quả trích triplet do SUBAGENT tạo ra (output/subagent_output/*.json) thành
triplets_raw.json, rồi xây ontology (Concept/Relation) và xuất triplet cuối kèm địa chỉ
điều khoản.

Bước trích triplet không còn do script gọi OpenAI API đảm nhiệm nữa — mỗi subagent (Claude,
qua Agent tool) tự đọc 1 file input (output/subagent_input/batch_XX.json, 10 section/file) và
tự ghi kết quả ra output/subagent_output/batch_XX.json theo đúng schema:
    [{"id", "path", "document_name", "propositions", "triplets": [{s, v, o,
      s_keyphrases, v_keyphrases, o_keyphrases}]}]
Script này CHỈ làm 2 việc, thuần code, KHÔNG gọi LLM:
    1. Gộp mọi file trong output/subagent_output/ vào output/triplets_raw.json (cộng dồn với
       lần chạy trước nếu có — ghi đè theo id để chạy lại 1 batch không bị nhân đôi), rồi chạy
       inject_missing_dieu_khien (bổ sung triplet nền (Người, Điều khiển, X) còn thiếu, suy ra
       bằng majority-vote theo Điều — an toàn vì không cần LLM).
    2. Build ontology bằng exact-match theo tên (không phán đoán đồng nghĩa — lý do: LLM merge
       từng gộp SAI hoàn toàn không liên quan, vd "Xe gắn máy" gộp lẫn vào "Người"). Exact-match
       không bao giờ gộp sai, đổi lại có thể còn sót concept trùng nghĩa bị đặt tên khác nhau ở
       batch khác nhau — chấp nhận được, rà tay dễ hơn nhiều so với gộp sai không kiểm soát.

Concept và Relation trong ontology.json đều có "id" ổn định (C0001.., R0001..), gán 1 lần theo
thứ tự tên đã sort — cùng 1 tên luôn ra cùng 1 id giữa các lần chạy, miễn tập concept/relation
không đổi. KG (kg_triplets.json) là đồ thị thật: node = Concept (tham chiếu bằng "id" ở trên),
edge tham chiếu node nguồn/đích qua "source"/"target" (id, không phải tên) + "relation_id" trỏ
về Relation trong ontology.json.

Input:  output/subagent_output/*.json   (mỗi file 1 batch do 1 subagent xử lý)
        output/triplets_raw.json        (nếu đã có từ lần chạy trước — merge cộng dồn theo id)
Output: output/triplets_raw.json        (merged, ghi đè)
        output/ontology.json            {"concepts": [{id, name, keyphrases}],
                                          "relations": [{id, name, keyphrases, concept_s, concept_o}]}
        output/triplets.json            mỗi triplet kèm address (article/clause/point/document)
        output/kg_triplets.json         {"nodes": [{id, name}],
                                          "edges": [{id, relation_id, relation, source, target, addresses}]}
        output/atomicity_warnings.json  cảnh báo (không tự sửa) nếu 1 concept nghi ngờ còn lẫn
                                         quan hệ ẩn — để chuyên gia rà tay
"""

import os
import re
import json
import glob
import argparse

BASE_DIR            = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR          = os.path.join(BASE_DIR, "output")
SUBAGENT_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "subagent_output")
TRIPLETS_RAW        = os.path.join(OUTPUT_DIR, "triplets_raw.json")
OUT_ONTOLOGY        = os.path.join(OUTPUT_DIR, "ontology.json")
OUT_TRIPLETS        = os.path.join(OUTPUT_DIR, "triplets.json")
OUT_KG              = os.path.join(OUTPUT_DIR, "kg_triplets.json")
OUT_WARNINGS        = os.path.join(OUTPUT_DIR, "atomicity_warnings.json")


def log(msg: str) -> None:
    print(msg, flush=True)


def save_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


_ID_NUM_RE = re.compile(r"\d+")


def _id_sort_key(sid: str):
    m = _ID_NUM_RE.search(sid or "")
    return int(m.group()) if m else float("inf")


# ── Bước 0: gộp output/subagent_output/*.json + triplets_raw.json cũ (nếu có) ────────────────

def merge_subagent_outputs() -> list[dict]:
    by_id: dict[str, dict] = {}

    if os.path.exists(TRIPLETS_RAW):
        with open(TRIPLETS_RAW, encoding="utf-8") as f:
            for item in json.load(f):
                if item.get("id"):
                    by_id[item["id"]] = item

    files = sorted(glob.glob(os.path.join(SUBAGENT_OUTPUT_DIR, "*.json")))
    overwritten = 0
    for path in files:
        with open(path, encoding="utf-8") as f:
            items = json.load(f)
        for item in items:
            sid = item.get("id")
            if not sid:
                continue
            if sid in by_id and by_id[sid] != item:
                overwritten += 1
            by_id[sid] = item

    log(f"Gộp {len(files)} file subagent_output + triplets_raw.json cũ → {len(by_id)} section"
        + (f" ({overwritten} section bị ghi đè bởi bản mới hơn)" if overwritten else ""))

    return sorted(by_id.values(), key=lambda x: _id_sort_key(x.get("id", "")))


# ── Bước A: hậu xử lý bằng CODE — bổ sung (Người, Điều khiển, X) còn thiếu ────────────────────

_ARTICLE_RE = re.compile(r"Điều\s+(\d+)")


def inject_missing_dieu_khien(sections: list[dict]) -> int:
    """
    LLM/subagent thỉnh thoảng bỏ sót triplet nền (Người, Điều khiển, <xe>) khi câu có nhiều hành
    vi phức tạp phía sau, dù câu luôn mở đầu bằng "Người điều khiển [loại xe]" và loại xe đó
    THỐNG NHẤT trong toàn bộ 1 Điều. Với mỗi Điều: lấy object PHỔ BIẾN NHẤT mà chính LLM/subagent
    đã trích đúng ở các section khác trong CÙNG Điều (majority vote, cần ≥2 lần) làm căn cứ, rồi
    bổ sung triplet còn thiếu cho các section có propositions mở đầu bằng "Người điều khiển"
    nhưng CHƯA có triplet nào v="Điều khiển". Trả về số triplet đã bổ sung.
    """
    by_article: dict[int, list[dict]] = {}
    for sec in sections:
        m = _ARTICLE_RE.search(sec.get("path") or "")
        if m:
            by_article.setdefault(int(m.group(1)), []).append(sec)

    injected = 0
    for art, secs in by_article.items():
        object_counter: dict[str, int] = {}
        object_keyphrases: dict[str, list[str]] = {}
        for sec in secs:
            for t in sec.get("triplets", []):
                if t.get("v", "").strip().lower() == "điều khiển":
                    o = t["o"]
                    object_counter[o] = object_counter.get(o, 0) + 1
                    bucket = object_keyphrases.setdefault(o, [])
                    for kp in t.get("o_keyphrases", []):
                        if kp not in bucket:
                            bucket.append(kp)

        if not object_counter:
            continue
        majority_o = max(object_counter, key=object_counter.get)
        if object_counter[majority_o] < 2:
            continue  # không đủ căn cứ (chỉ 1 lần) — không suy diễn cho cả Điều

        for sec in secs:
            props = sec.get("propositions") or []
            text = props[0] if props else ""
            if not text.strip().lower().startswith("người điều khiển"):
                continue
            already = any(t.get("v", "").strip().lower() == "điều khiển" for t in sec.get("triplets", []))
            if already:
                continue
            sec.setdefault("triplets", []).insert(0, {
                "s": "Người", "v": "Điều khiển", "o": majority_o,
                "s_keyphrases": ["người"], "v_keyphrases": ["điều khiển"],
                "o_keyphrases": object_keyphrases.get(majority_o) or [majority_o.lower()],
            })
            injected += 1

    return injected


# ── Bước B: gom TOÀN BỘ triplet thành ontology, so khớp CHÍNH XÁC theo tên ────────────────────

def build_ontology(sections: list[dict]) -> tuple[dict, dict[str, str], dict[tuple[str, str], str]]:
    """
    Duyệt mọi triplet {s, v, o, s_keyphrases, v_keyphrases, o_keyphrases} trên toàn bộ section.
    - Concept: khoá = tên s/o CHÍNH XÁC. Gặp lại tên đã có → cộng dồn keyphrases (union), không
      tạo entry mới.
    - Relation: khoá = cặp (s, v) CHÍNH XÁC (1 relation chỉ có đúng 1 concept_s). Gặp lại đúng cặp
      → cộng dồn concept_o (union) + keyphrases (union), không tạo entry mới.

    Trả về (ontology, concept_to_id, relation_to_id) — 2 map id dùng lại ở Bước D để dựng
    node/edge của KG bằng id thay vì tên.
    """
    concept_kps: dict[str, list[str]] = {}
    relation_agg: dict[tuple[str, str], dict] = {}

    def _union(bucket: list[str], values: list[str]) -> None:
        # Lưới an toàn cuối: loại keyphrase chứa dấu phẩy (dấu hiệu là 1 danh sách/mệnh đề bị kéo
        # nguyên vào, không phải tên gọi atomic) — dù Bước 1 đã lọc, vẫn lọc lại ở đây phòng dữ liệu
        # cũ/sót. KHÔNG lọc theo độ dài — tên loại xe hợp lệ trong luật vốn có thể dài >6 từ mà vẫn
        # atomic (vd "xe chở người bốn bánh có gắn động cơ"). KHÔNG fallback về tên canonical khi
        # rỗng — mảng keyphrase rỗng là kết quả hợp lệ (câu gốc không có cách viết nào khác), không
        # phải thiếu sót cần lấp đầy (xem EXTRACTION_RULES.md mục E16).
        clean = [v.strip() for v in values if isinstance(v, str) and v.strip() and "," not in v]
        for v in clean:
            if v not in bucket:
                bucket.append(v)

    for sec in sections:
        for t in sec.get("triplets", []):
            s, v, o = (t.get("s") or "").strip(), (t.get("v") or "").strip(), (t.get("o") or "").strip()
            if not (s and v and o):
                continue

            _union(concept_kps.setdefault(s, []), t.get("s_keyphrases", []))
            _union(concept_kps.setdefault(o, []), t.get("o_keyphrases", []))

            entry = relation_agg.setdefault((s, v), {"concept_o": [], "keyphrases": []})
            if o not in entry["concept_o"]:
                entry["concept_o"].append(o)
            _union(entry["keyphrases"], t.get("v_keyphrases", []))

    concept_to_id = {n: f"C{i+1:04d}" for i, n in enumerate(sorted(concept_kps))}
    relation_to_id = {key: f"R{i+1:04d}" for i, key in enumerate(sorted(relation_agg))}

    concepts = [
        {"id": concept_to_id[n], "name": n, "keyphrases": kps}
        for n, kps in sorted(concept_kps.items(), key=lambda kv: concept_to_id[kv[0]])
    ]
    relations = [
        {
            "id": relation_to_id[(s, v)],
            "name": v,
            "keyphrases": e["keyphrases"],
            "concept_s": s,
            "concept_o": e["concept_o"],
        }
        for (s, v), e in sorted(relation_agg.items(), key=lambda kv: relation_to_id[kv[0]])
    ]
    ontology = {"concepts": concepts, "relations": relations}
    return ontology, concept_to_id, relation_to_id


# ── Bước C: rà atomicity của keyphrase cuối (cảnh báo, KHÔNG tự sửa) ──────────────────────────

_WORD_RE = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)


def check_atomicity(ontology: dict) -> list[dict]:
    """
    Cảnh báo nếu TÊN hoặc keyphrase của 1 Concept CHỨA (theo ranh giới từ) 1 keyphrase của Relation
    bất kỳ — dấu hiệu subject/object đó còn lẫn quan hệ ẩn. Không tự sửa — chỉ ghi lại để rà tay.
    """
    relation_kps: set[str] = set()
    for r in ontology.get("relations", []):
        relation_kps.add(r["name"].strip().lower())
        for kp in r.get("keyphrases", []):
            relation_kps.add(kp.strip().lower())

    warnings = []
    for c in ontology.get("concepts", []):
        candidates = [c["name"]] + c.get("keyphrases", [])
        for cand in candidates:
            cand_l = cand.strip().lower()
            words = set(_WORD_RE.findall(cand_l))
            for rel_kp in relation_kps:
                rel_words = _WORD_RE.findall(rel_kp)
                if rel_words and set(rel_words).issubset(words) and rel_kp != cand_l:
                    warnings.append({
                        "concept": c["name"], "flagged_text": cand,
                        "matched_relation_keyphrase": rel_kp,
                    })
    return warnings


# ── Bước D: dựng triplet cuối + address ───────────────────────────────────────────────────────

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


def reconstruct_triplets(sections: list[dict]) -> list[dict]:
    out = []
    for sec in sections:
        address = parse_address(sec.get("path", ""), sec.get("document_name", ""))
        for t in sec.get("triplets", []):
            s, v, o = (t.get("s") or "").strip(), (t.get("v") or "").strip(), (t.get("o") or "").strip()
            if not (s and v and o):
                continue
            out.append({
                "section_id": sec["id"],
                "address":    address,
                "concept_s":  s,
                "relation":   v,
                "concept_o":  o,
            })
    return out


def build_kg(
    triplets: list[dict],
    concept_to_id: dict[str, str],
    relation_to_id: dict[tuple[str, str], str],
) -> dict:
    """
    Dedup theo (concept_s, relation, concept_o), gộp addresses, rồi dựng đồ thị thật:
    - node = Concept, id lấy đúng từ concept_to_id (khớp 1-1 với id trong ontology.json).
    - edge = Triplet, "source"/"target" là id node (không phải tên), "relation_id" trỏ về
      Relation tương ứng trong ontology.json (khoá theo (concept_s, relation)).
    """
    index: dict[tuple, dict] = {}
    for t in triplets:
        key = (t["concept_s"], t["relation"], t["concept_o"])
        if key not in index:
            index[key] = {
                "relation_id": relation_to_id[(t["concept_s"], t["relation"])],
                "relation":    t["relation"],
                "source":      concept_to_id[t["concept_s"]],
                "target":      concept_to_id[t["concept_o"]],
                "addresses":   [],
            }
        if t["address"] not in index[key]["addresses"]:
            index[key]["addresses"].append(t["address"])

    edges = [
        {"id": f"E{i+1:04d}", **edge}
        for i, edge in enumerate(index.values())
    ]
    nodes = [
        {"id": cid, "name": name}
        for name, cid in sorted(concept_to_id.items(), key=lambda kv: kv[1])
    ]
    return {"nodes": nodes, "edges": edges}


# ── Main ────────────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Gộp output/subagent_output/*.json → triplets_raw.json → build ontology (không LLM)"
    )
    parser.add_argument("--no-inject", action="store_true",
                        help="Tắt bước hậu xử lý bổ sung triplet (Người, Điều khiển, X) còn thiếu")
    args = parser.parse_args()

    sections = merge_subagent_outputs()
    if not sections:
        log("Không có section nào trong output/subagent_output/ hoặc triplets_raw.json — dừng.")
        return

    if not args.no_inject:
        n = inject_missing_dieu_khien(sections)
        log(f"[inject] bổ sung {n} triplet (Người, Điều khiển, X) còn thiếu")

    save_json(TRIPLETS_RAW, sections)
    log(f"Triplets raw (merged) → {TRIPLETS_RAW} ({len(sections)} section)")

    ontology, concept_to_id, relation_to_id = build_ontology(sections)
    save_json(OUT_ONTOLOGY, ontology)
    log(f"Ontology → {OUT_ONTOLOGY} ({len(ontology['concepts'])} concepts, {len(ontology['relations'])} relations)")

    warnings = check_atomicity(ontology)
    save_json(OUT_WARNINGS, warnings)
    log(f"Atomicity warnings: {len(warnings)} → {OUT_WARNINGS}")

    triplets = reconstruct_triplets(sections)
    save_json(OUT_TRIPLETS, triplets)
    log(f"Triplets (per-section) → {OUT_TRIPLETS} ({len(triplets)} triplets)")

    kg = build_kg(triplets, concept_to_id, relation_to_id)
    save_json(OUT_KG, kg)
    log(f"KG → {OUT_KG} ({len(kg['nodes'])} nodes, {len(kg['edges'])} edges)")

    from collections import Counter
    rel_counts = Counter(e["relation"] for e in kg["edges"])
    log("\nTop relations:")
    for rel, cnt in rel_counts.most_common(10):
        log(f"  {rel}: {cnt} edges")


if __name__ == "__main__":
    main()
