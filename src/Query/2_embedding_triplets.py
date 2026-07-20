"""
Bước 2: Vector hoá các sub-triplet đã tách từ câu hỏi người dùng.

Bước 1 (tách câu hỏi tự do → sub-triplet (s, v, o)) CHƯA làm — file input hiện tại
(`output/question_triplets.json`) là VÍ DỤ viết tay, đóng vai trò định nghĩa FORMAT mà Bước 1
(sẽ làm sau, có thể bằng LLM) phải sinh ra.

Format input — 1 JSON array, mỗi phần tử là 1 sub-triplet của câu hỏi:
    [{"s": "người", "v": "uống", "o": "rượu bia"}, {"s": "người", "v": "lái", "o": "xe ô tô"}]
- Một câu hỏi có thể tách ra NHIỀU sub-triplet độc lập (xem Bước 3 — kết quả các sub-triplet sẽ
  được giao/hợp lại với nhau).
- Thành phần nào bị khuyết (câu hỏi không nêu, hoặc chính là ẩn số cần tìm) → dùng "*" thay vì bỏ
  trống. "*" nghĩa là "khớp với bất kỳ" ở đúng vị trí đó khi truy vấn (xử lý ở Bước 3).

Input:  output/question_triplets.json
Output: output/question_triplets_vectorized.json
        [{"s": {"name":.., "vector":[...]|null}, "v": {...}, "o": {...}}, ...]
        vector = null khi name == "*" (không embed thành phần khuyết).

QUAN TRỌNG: LUÔN embed trên CPU (device="cpu"), TUYỆT ĐỐI không dùng MPS — đã kiểm chứng thực tế
backend MPS của PyTorch trả SAI vector khi batch nhiều câu cùng lúc với model này (cùng 1 chữ
encode riêng lẻ vs trong batch ra 2 vector khác hẳn nhau, cosine chỉ ~0.43 thay vì phải là 1.0).
CPU cho kết quả đúng tuyệt đối bất kể batch, và ở quy mô vài triplet/câu hỏi thì cực nhanh.
Model dùng CHUNG với bước build index (src/Building_KG/ontology/3_build_keyphrase_index.py) —
bắt buộc trùng model, nếu không vector 2 bên không nằm chung không gian, so cosine vô nghĩa.
"""

import os
import json
from sentence_transformers import SentenceTransformer

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
IN_FILE    = os.path.join(OUTPUT_DIR, "question_triplets.json")
OUT_FILE   = os.path.join(OUTPUT_DIR, "question_triplets_vectorized.json")

EMBEDDING_MODEL = "AITeamVN/Vietnamese_Embedding_v2"


def log(msg: str) -> None:
    print(msg, flush=True)


def main():
    with open(IN_FILE, encoding="utf-8") as f:
        triplets = json.load(f)
    log(f"Đã tải {len(triplets)} sub-triplet từ {IN_FILE}")

    # Gom mọi text cần embed (bỏ qua "*" và rỗng), embed 1 lượt duy nhất cho nhanh, rồi rải
    # ngược kết quả lại đúng vị trí s/v/o của từng triplet.
    texts_to_embed: list[str] = []
    for t in triplets:
        for role in ("s", "v", "o"):
            val = (t.get(role) or "").strip()
            if val and val != "*":
                texts_to_embed.append(val)

    log(f"Đang embed {len(texts_to_embed)} cụm từ (CPU)...")
    model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    embeddings = model.encode(texts_to_embed, normalize_embeddings=True, show_progress_bar=True) \
        if texts_to_embed else []
    vector_by_text: dict[str, list[float]] = {
        text: vec.tolist() for text, vec in zip(texts_to_embed, embeddings)
    }

    output = []
    for t in triplets:
        entry = {}
        for role in ("s", "v", "o"):
            val = (t.get(role) or "").strip()
            entry[role] = {
                "name": val,
                "vector": vector_by_text.get(val),  # None nếu val == "*" hoặc rỗng
            }
        output.append(entry)

    tmp = OUT_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    os.replace(tmp, OUT_FILE)
    log(f"Đã lưu {len(output)} sub-triplet có vector → {OUT_FILE}")


if __name__ == "__main__":
    main()
