import json
import os
import torch
from sentence_transformers import SentenceTransformer

def generate_embeddings(input_filepath, output_filepath, model):
    print(f"\nĐang xử lý: {input_filepath}")

    # Đọc file data đầu vào
    try:
        with open(input_filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Không tìm thấy file {input_filepath}. Vui lòng kiểm tra lại đường dẫn.")
        return

    if not data:
        print("File JSON rỗng.")
        return

    print(f"Đã tải {len(data)} bản ghi. Đang tiến hành vector hóa...")

    # Trích xuất văn bản cần vector hóa từ trường 'name'
    texts_to_encode = [item["name"] for item in data]

    # Dùng model để sinh vector (Embedding)
    # Hàm encode hỗ trợ xử lý hàng loạt cực kỳ tối ưu, show_progress_bar giúp xem tiến độ
    embeddings = model.encode(texts_to_encode, show_progress_bar=True)

    # Gắn vector trở lại vào từng object của JSON
    for i, item in enumerate(data):
        # Lưu ý QUAN TRỌNG: model.encode trả về mảng Numpy (Numpy Array).
        # JSON không hiểu được định dạng Numpy, nên bắt buộc phải ép kiểu về list() chuẩn của Python.
        item["vector"] = embeddings[i].tolist()

    # Lưu kết quả ra file mới
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Hoàn tất! Đã lưu file có chứa vector tại: {output_filepath}")

    # In thử chiều dài của vector để kiểm tra (thường model này trả về vector 768 chiều)
    vector_dimension = len(data[0]['vector'])
    print(f"Kích thước (số chiều) của mỗi vector là: {vector_dimension}")

def main():
    # Khởi tạo model AITeamVN/Vietnamese_Embedding_v2
    print("Đang kiểm tra thiết bị phần cứng...")
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    print("Đang tải model từ Hugging Face...")
    model = SentenceTransformer('AITeamVN/Vietnamese_Embedding_v2', device=device)

    # Đường dẫn đến thư mục chứa file python
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Danh sách các file cần xử lý (input_name, output_name)
    files_to_process = [
        # ("3_split_entities_vphc.json", "4_entities_with_vectors_vphc.json"),
        ("3_split_entities_dat_dai_1.json", "4_entities_with_vectors_dat_dai_1.json")
    ]

    for input_name, output_name in files_to_process:
        input_path = os.path.join(base_dir, input_name)
        output_path = os.path.join(base_dir, output_name)
        generate_embeddings(input_path, output_path, model)

if __name__ == "__main__":
    main()