import json
from sentence_transformers import SentenceTransformer

def main():
    # 1. Khởi tạo model AITeamVN/Vietnamese_Embedding_v2
    # Quá trình này sẽ mất một chút thời gian ở lần chạy đầu tiên để tải model (~vài trăm MB)
    print("Đang tải model từ Hugging Face...")
    model = SentenceTransformer('AITeamVN/Vietnamese_Embedding_v2', device='mps')

    # Đường dẫn file
    input_filepath = "./version_3/query/2_entities_question.json"
    output_filepath = "./version_3/query/3_entities_with_vectors_question.json"

    # 2. Đọc file data đầu vào
    try:
        with open(input_filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Không tìm thấy file {input_filepath}. Vui lòng kiểm tra lại đường dẫn.")
        return

    print(f"Đã tải {len(data)} bản ghi. Đang tiến hành vector hóa...")

    # 3. Trích xuất văn bản cần vector hóa
    # Chúng ta ưu tiên vector hóa trường 'name' vì đây là tên của thực thể
    texts_to_encode = [item["name"] for item in data]

    # 4. Dùng model để sinh vector (Embedding)
    # Hàm encode hỗ trợ xử lý hàng loạt cực kỳ tối ưu, show_progress_bar giúp xem tiến độ
    embeddings = model.encode(texts_to_encode, show_progress_bar=True)

    # 5. Gắn vector trở lại vào từng object của JSON
    for i, item in enumerate(data):
        # Lưu ý QUAN TRỌNG: model.encode trả về mảng Numpy (Numpy Array).
        # JSON không hiểu được định dạng Numpy, nên bắt buộc phải ép kiểu về list() chuẩn của Python.
        item["vector"] = embeddings[i].tolist()

    # 6. Lưu kết quả ra file mới
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"\nHoàn tất! Đã lưu file có chứa vector tại: {output_filepath}")

    # In thử chiều dài của vector để kiểm tra (thường model này trả về vector 768 chiều)
    vector_dimension = len(data[0]['vector'])
    print(f"Kích thước (số chiều) của mỗi vector là: {vector_dimension}")

if __name__ == "__main__":
    main()