"""
Bước 3: Truy vấn đồ thị bằng các sub-triplet đã vector hoá (output/question_triplets_vectorized.json).

Đọc index vector + đồ thị từ output CỦA PIPELINE BUILD (src/Building_KG/ontology/) qua đường dẫn
tương đối — không copy dữ liệu sang đây, luôn dùng bản mới nhất mà pipeline build sinh ra.

Với MỖI sub-triplet (s, v, o):
    1. So cosine vector s/o với các row "type=concept" trong keyphrase_vectors.npy, vector v với
       row "type=relation" — lấy top-K trên ngưỡng, union "ids" của các row khớp
       → listS (node_id), listV (edge_id), listO (node_id).
       Thành phần "*" (name rỗng/vector None) → khớp với TẤT CẢ id cùng loại (không thu hẹp).
    2. Duyệt candidate edge trong listV, giữ edge nào thoả CẢ 3: edge.id ∈ listV (hiển nhiên vì
       duyệt từ listV) VÀ edge.source ∈ listS VÀ edge.target ∈ listO.
    3. Chấm điểm mỗi edge khớp = min(sim_S, sim_V, sim_O) — bảo thủ, tránh 1 điểm cao che 2 điểm
       yếu. Giữ mọi edge trên ngưỡng (không chỉ top-1), union "section_ids" của chúng.

Sau khi có 1 tập section_ids cho MỖI sub-triplet, kết hợp giữa các sub-triplet:
    - Giao (intersection) trước — đúng khi câu hỏi mô tả 1 hành vi ghép (nhiều sub-triplet cùng
      nằm chung 1 điều khoản).
    - Giao rỗng → fallback UNION (kèm nhãn sub-triplet nào góp phần) — tránh trả về rỗng khi câu
      hỏi thực chất hỏi nhiều hành vi độc lập không chung điều khoản nào.

Input:  output/question_triplets_vectorized.json
        ../Building_KG/ontology/output/keyphrase_vectors.npy
        ../Building_KG/ontology/output/keyphrase_rows.json
        ../Building_KG/ontology/output/kg_triplets.json
Output: in ra kết quả + ghi output/query_result.json
"""

import os
import json
import numpy as np

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(BASE_DIR, "output")
IN_TRIPLETS = os.path.join(OUTPUT_DIR, "question_triplets_vectorized.json")
OUT_RESULT  = os.path.join(OUTPUT_DIR, "query_result.json")

BUILD_OUTPUT_DIR  = os.path.join(os.path.dirname(BASE_DIR), "Building_KG", "ontology", "output")
KEYPHRASE_VECTORS = os.path.join(BUILD_OUTPUT_DIR, "keyphrase_vectors.npy")
KEYPHRASE_ROWS    = os.path.join(BUILD_OUTPUT_DIR, "keyphrase_rows.json")
KG_FILE           = os.path.join(BUILD_OUTPUT_DIR, "kg_triplets.json")
TRIPLETS_RAW_FILE = os.path.join(BUILD_OUTPUT_DIR, "triplets_raw.json")

TOP_K     = 5
THRESHOLD = 0.5


def log(msg: str) -> None:
    print(msg, flush=True)


def load_index():
    vectors = np.load(KEYPHRASE_VECTORS)
    with open(KEYPHRASE_ROWS, encoding="utf-8") as f:
        rows = json.load(f)
    with open(KG_FILE, encoding="utf-8") as f:
        kg = json.load(f)
    edges = {e["id"]: e for e in kg["edges"]}
    node_names = {n["id"]: n["name"] for n in kg["nodes"]}

    idxs_by_type: dict[str, list[int]] = {"concept": [], "relation": []}
    for i, r in enumerate(rows):
        idxs_by_type[r["type"]].append(i)

    all_node_ids = set(node_names)
    all_edge_ids = set(edges)

    return vectors, rows, edges, node_names, idxs_by_type, all_node_ids, all_edge_ids


def load_section_texts() -> dict[str, dict]:
    """section_id -> {path, document_name, propositions} — lấy từ triplets_raw.json (nguồn gốc
    của cả pipeline build), để gắn văn bản đã viết lại (rewritten) vào kết quả truy vấn cuối."""
    with open(TRIPLETS_RAW_FILE, encoding="utf-8") as f:
        sections = json.load(f)
    return {
        s["id"]: {
            "path": s.get("path"),
            "document_name": s.get("document_name"),
            "propositions": s.get("propositions", []),
        }
        for s in sections
    }


def search(vector, row_type: str, vectors, rows, idxs_by_type, all_ids: set,
           top_k: int = TOP_K, threshold: float = THRESHOLD) -> dict[str, float]:
    """Trả về {id: điểm cao nhất} — "*" (vector=None) khớp với TOÀN BỘ id cùng loại, điểm 1.0."""
    if vector is None:
        return {i: 1.0 for i in all_ids}

    qv = np.asarray(vector, dtype=np.float32)
    idxs = idxs_by_type[row_type]
    sims = [(i, float(np.dot(qv, vectors[i]))) for i in idxs]
    sims.sort(key=lambda x: -x[1])

    result: dict[str, float] = {}
    for i, sim in sims[:top_k]:
        if sim < threshold:
            break
        for id_ in rows[i]["ids"]:
            if id_ not in result or sim > result[id_]:
                result[id_] = sim
    return result


def match_triplet(triplet: dict, vectors, rows, edges, idxs_by_type, all_node_ids, all_edge_ids) -> dict:
    s_scores = search(triplet["s"]["vector"], "concept", vectors, rows, idxs_by_type, all_node_ids)
    v_scores = search(triplet["v"]["vector"], "relation", vectors, rows, idxs_by_type, all_edge_ids)
    o_scores = search(triplet["o"]["vector"], "concept", vectors, rows, idxs_by_type, all_node_ids)

    listS, listV, listO = set(s_scores), set(v_scores), set(o_scores)

    matched_edges = []
    for eid in listV:
        e = edges.get(eid)
        if not e or e["source"] not in listS or e["target"] not in listO:
            continue
        score = min(s_scores[e["source"]], v_scores[eid], o_scores[e["target"]])
        matched_edges.append({"edge_id": eid, "score": score,
                               "source": e["source"], "relation": e["relation"], "target": e["target"],
                               "section_ids": e["section_ids"]})
    matched_edges.sort(key=lambda m: -m["score"])

    section_ids: set = set()
    for m in matched_edges:
        section_ids.update(m["section_ids"])

    return {
        "query": {"s": triplet["s"]["name"], "v": triplet["v"]["name"], "o": triplet["o"]["name"]},
        "matched_edges": matched_edges,
        "section_ids": sorted(section_ids),
    }


def combine(per_triplet_results: list[dict]) -> dict:
    sets = [set(r["section_ids"]) for r in per_triplet_results]
    non_empty = [s for s in sets if s]
    if not non_empty:
        return {"strategy": "none", "section_ids": []}

    inter = set.intersection(*non_empty) if len(non_empty) > 1 else non_empty[0]
    if inter:
        return {"strategy": "intersection", "section_ids": sorted(inter)}

    union = set.union(*non_empty)
    return {"strategy": "union_fallback", "section_ids": sorted(union)}


def main():
    with open(IN_TRIPLETS, encoding="utf-8") as f:
        triplets = json.load(f)
    log(f"Đã tải {len(triplets)} sub-triplet đã vector hoá từ {IN_TRIPLETS}")

    vectors, rows, edges, node_names, idxs_by_type, all_node_ids, all_edge_ids = load_index()
    log(f"Đã tải index: {len(rows)} keyphrase row, {len(edges)} edge (từ {BUILD_OUTPUT_DIR})")

    per_triplet_results = []
    for t in triplets:
        r = match_triplet(t, vectors, rows, edges, idxs_by_type, all_node_ids, all_edge_ids)
        per_triplet_results.append(r)
        log(f"\nSub-triplet: ({r['query']['s']}, {r['query']['v']}, {r['query']['o']})")
        if not r["matched_edges"]:
            log("  => Không khớp edge nào trong đồ thị")
        for m in r["matched_edges"][:5]:
            log(f"  => ({node_names.get(m['source'])}, {m['relation']}, {node_names.get(m['target'])}) "
                f"score={m['score']:.3f}")
        log(f"  section_ids: {r['section_ids']}")

    final = combine(per_triplet_results)
    section_texts = load_section_texts()
    final["sections"] = [
        {"section_id": sid, **section_texts.get(sid, {"path": None, "document_name": None, "propositions": []})}
        for sid in final["section_ids"]
    ]

    log(f"\n=== KẾT QUẢ CUỐI ({final['strategy']}) ===")
    for sec in final["sections"]:
        log(f"  [{sec['section_id']}] {sec['path']}")
        for p in sec["propositions"]:
            log(f"      {p}")

    result = {"per_triplet": per_triplet_results, "final": final}
    tmp = OUT_RESULT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    os.replace(tmp, OUT_RESULT)
    log(f"\nĐã lưu kết quả đầy đủ → {OUT_RESULT}")


if __name__ == "__main__":
    main()
