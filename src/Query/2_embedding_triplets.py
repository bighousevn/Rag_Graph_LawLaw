import json
import os
import torch
from sentence_transformers import SentenceTransformer


def main():
    # 1. Khởi tạo model AITeamVN/Vietnamese_Embedding_v2
    print("Đang kiểm tra thiết bị phần cứng...")
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    print("Đang tải model từ Hugging Face...")
    model = SentenceTransformer("AITeamVN/Vietnamese_Embedding_v2", device=device)

    # Đường dẫn file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_filepath = os.path.join(base_dir, "output", "1.2_triplets.json")
    output_filepath = os.path.join(base_dir, "output", "1.3_triplets_with_nested_vectors.json")

    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    # 2. Đọc file data đầu vào
    try:
        with open(input_filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Không tìm thấy file {input_filepath}. Vui lòng kiểm tra lại đường dẫn.")
        return

    if not isinstance(data, list):
        print(f"File {input_filepath} không đúng định dạng danh sách triplet.")
        return

    print(f"Đã tải {len(data)} triplet. Đang tiến hành vector hóa...")

    # 3. Trích xuất toàn bộ văn bản cần vector hóa theo thứ tự s, v, o
    texts_to_encode = []
    for item in data:
        texts_to_encode.extend([
            item.get("s", ""),
            item.get("v", ""),
            item.get("o", "")
        ])

    # 4. Dùng model để sinh vector (Embedding)
    # Hàm encode hỗ trợ xử lý hàng loạt cực kỳ tối ưu, show_progress_bar giúp xem tiến độ
    embeddings = model.encode(texts_to_encode, show_progress_bar=True)

    # 5. Gắn vector trở lại vào từng triplet với cấu trúc lồng nhau
    output_data = []
    embedding_index = 0

    for item in data:
        s_name = item.get("s", "")
        v_name = item.get("v", "")
        o_name = item.get("o", "")

        output_data.append({
            "s": {
                "name": s_name,
                "vector": embeddings[embedding_index].tolist()
            },
            "v": {
                "name": v_name,
                "vector": embeddings[embedding_index + 1].tolist()
            },
            "o": {
                "name": o_name,
                "vector": embeddings[embedding_index + 2].tolist()
            }
        })
        embedding_index += 3

    # 6. Lưu kết quả ra file mới
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(f"\nHoàn tất! Đã lưu file có chứa vector tại: {output_filepath}")

    # In thử chiều dài của vector để kiểm tra (thường model này trả về vector 768 chiều)
    if output_data:
        vector_dimension = len(output_data[0]["s"]["vector"])
        print(f"Kích thước (số chiều) của mỗi vector là: {vector_dimension}")


if __name__ == "__main__":
    main()
