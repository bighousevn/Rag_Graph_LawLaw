"""
Bước 3: Vector hoá keyphrase (Concept + Relation) từ ontology.json, map sẵn về node_id/edge_id
thật trong kg_triplets.json — phục vụ truy vấn bằng vector ở bước sau (chưa viết).

Input:  output/ontology.json      {"concepts": [{id,name,keyphrases}],
                                    "relations": [{id,name,keyphrases,concept_s,concept_o}]}
        output/kg_triplets.json   {"nodes": [{id,name}],
                                    "edges": [{id,relation_id,relation,source,target,section_ids}]}
Output: output/keyphrase_vectors.npy   ma trận N×D (float32), đã L2-normalize — hàng i ứng với
                                        phần tử thứ i trong keyphrase_rows.json (căn theo index).
        output/keyphrase_rows.json     N phần tử song song: {"text", "type": "concept"|"relation",
                                        "ids": [...]}
                                        - type=concept  → "ids" là node_id (thường 1 phần tử)
                                        - type=relation → "ids" là edge_id (MỌI edge cùng chia sẻ
                                          relation_id đó — để lúc truy vấn có thẳng listV ra edge_id,
                                          không cần tra lại concept_s/concept_o trong ontology.json)

Mỗi Concept/Relation luôn có ÍT NHẤT 1 row — dùng chính tên canonical khi "keyphrases" rỗng (rất
phổ biến với rule keyphrase hiện tại, ~93% theo thống kê thực tế). Không có row nào thì concept/
relation đó không bao giờ match được lúc search, kể cả khi người dùng gõ đúng tên canonical.

Dedup theo (type, text) CHÍNH XÁC — nếu cùng 1 chuỗi text xuất hiện ở nhiều concept/relation khác
nhau (hiếm nhưng có thể, do nhiều subagent độc lập sinh dữ liệu không đối chiếu chéo), gộp "ids"
vào chung 1 row thay vì tạo row trùng lặp — vừa tránh embed lại đúng 1 chuỗi nhiều lần, vừa phản
ánh đúng bản chất mơ hồ (1 keyphrase trỏ về nhiều thực thể) khi truy vấn.
"""

import os
import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR    = os.path.join(BASE_DIR, "output")
ONTOLOGY_FILE = os.path.join(OUTPUT_DIR, "ontology.json")
KG_FILE       = os.path.join(OUTPUT_DIR, "kg_triplets.json")
OUT_VECTORS   = os.path.join(OUTPUT_DIR, "keyphrase_vectors.npy")
OUT_ROWS      = os.path.join(OUTPUT_DIR, "keyphrase_rows.json")

EMBEDDING_MODEL = "AITeamVN/Vietnamese_Embedding_v2"


def log(msg: str) -> None:
    print(msg, flush=True)


def build_rows(ontology: dict, kg: dict) -> list[dict]:
    # Precompute 1 lần: relation_id -> danh sách edge_id (1 relation có thể ứng nhiều edge
    # khác concept_o).
    edges_by_relation: dict[str, list[str]] = {}
    for e in kg.get("edges", []):
        edges_by_relation.setdefault(e["relation_id"], []).append(e["id"])

    merged: dict[tuple[str, str], dict] = {}  # (type, text) -> row

    def add(text: str, row_type: str, ids: list[str]) -> None:
        text = (text or "").strip()
        if not text or not ids:
            return
        key = (row_type, text)
        entry = merged.setdefault(key, {"text": text, "type": row_type, "ids": []})
        for i in ids:
            if i not in entry["ids"]:
                entry["ids"].append(i)

    # LUÔN gồm cả tên canonical, cộng thêm keyphrase (nếu có) — không phải "or" (fallback khi
    # rỗng). Theo EXTRACTION_RULES.md, keyphrase KHÔNG BAO GIỜ chứa lại tên canonical, nên nếu
    # dùng "or" thì mọi concept/relation ĐÃ có keyphrase sẽ mất luôn row cho chính tên canonical
    # của nó — bug thực tế đã gặp với concept "Người" (có keyphrase nên "Người" không được vector
    # hoá, không bao giờ match được dù người dùng gõ đúng từ "người").
    for c in ontology.get("concepts", []):
        texts = list(c.get("keyphrases") or []) + [c["name"]]
        for t in texts:
            add(t, "concept", [c["id"]])

    for r in ontology.get("relations", []):
        edge_ids = edges_by_relation.get(r["id"], [])
        texts = list(r.get("keyphrases") or []) + [r["name"]]
        for t in texts:
            add(t, "relation", edge_ids)

    return list(merged.values())


def main():
    with open(ONTOLOGY_FILE, encoding="utf-8") as f:
        ontology = json.load(f)
    with open(KG_FILE, encoding="utf-8") as f:
        kg = json.load(f)

    rows = build_rows(ontology, kg)
    n_concept  = sum(1 for r in rows if r["type"] == "concept")
    n_relation = sum(1 for r in rows if r["type"] == "relation")
    log(f"Tổng {len(rows)} row keyphrase cần vector hoá ({n_concept} concept, {n_relation} relation)")

    # CỐ Ý không dùng MPS: backend MPS của PyTorch cho model này trả về vector SAI khi batch
    # nhiều câu cùng lúc (đã kiểm chứng — cùng 1 chữ encode riêng lẻ vs trong batch ra 2 vector
    # khác hẳn nhau, cosine chỉ ~0.43 thay vì 1.0). CPU cho kết quả đúng tuyệt đối bất kể batch,
    # và ở quy mô ~1000 keyphrase thì CPU vẫn đủ nhanh (vài giây). CUDA chưa kiểm chứng nên vẫn
    # cho phép nếu có, nhưng ưu tiên CPU trên máy Apple Silicon để tránh đúng bug đã gặp.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Thiết bị: {device}")
    log(f"Đang tải model {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)

    texts = [r["text"] for r in rows]
    log("Đang sinh vector...")
    vectors = model.encode(
        texts,
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2-normalize ngay ở đây -> cosine = dot product lúc query
    )
    vectors = np.asarray(vectors, dtype=np.float32)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(OUT_VECTORS, vectors)

    tmp = OUT_ROWS + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    os.replace(tmp, OUT_ROWS)

    log(f"Vectors → {OUT_VECTORS}  shape={vectors.shape}")
    log(f"Rows    → {OUT_ROWS}  ({len(rows)} phần tử)")


if __name__ == "__main__":
    main()
