import json
import os
import torch
import pymongo
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load biến môi trường
load_dotenv()

def generate_embeddings_and_load_to_mongo(input_filepath, model):
    print(f"\nĐang xử lý file: {input_filepath}")

    # Đọc file data đầu vào
    try:
        with open(input_filepath, "r", encoding="utf-8") as f:
            json_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file {input_filepath}. Vui lòng kiểm tra lại đường dẫn.")
        return

    if not json_data:
        print("❌ File JSON rỗng.")
        return

    # Lấy thông tin nodes và relationships
    graph_data = json_data[0] if isinstance(json_data, list) and json_data else json_data
    nodes = graph_data.get("nodes", [])
    relationships = graph_data.get("relationships", [])

    print(f"Đã tải dữ liệu từ JSON. Có {len(nodes)} nodes và {len(relationships)} relationships.")

    documents_to_insert = []

    # 1. Chuẩn bị dữ liệu cho Nodes (Thực thể)
    for node in nodes:
        documents_to_insert.append({
            "entityId": node["id"],
            "entityName": node["name"],
            "synonym": node["name"],
            "type": "node"
        })

    # 2. Chuẩn bị dữ liệu cho Relationships (Cạnh/Quan hệ)
    for rel in relationships:
        documents_to_insert.append({
            "entityId": rel["id"],
            "entityName": rel["name"],
            "synonym": rel["name"],
            "type": "relationship"
        })

    if not documents_to_insert:
        print("⚠️ Không tìm thấy thực thể hoặc quan hệ nào để vector hóa.")
        return

    print(f"Bắt đầu vector hóa {len(documents_to_insert)} bản ghi...")
    texts_to_encode = [doc["entityName"] for doc in documents_to_insert]

    # Tiến hành sinh vector (Embedding)
    embeddings = model.encode(texts_to_encode, show_progress_bar=True)

    # Gắn vector vào tài liệu
    for i, doc in enumerate(documents_to_insert):
        doc["vector"] = embeddings[i].tolist()

    # 3. Kết nối MongoDB Local từ .env
    MONGO_URI = os.getenv("MONGO_URI", "").strip()
    DB_NAME = os.getenv("MONGO_DB_NAME", "vectorDB").strip()
    COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "vector_entities").strip()

    if not MONGO_URI:
        raise ValueError("❌ Thiếu cấu hình MONGO_URI trong file .env.")

    print(f"Đang kết nối tới MongoDB Local: {MONGO_URI}...")
    try:
        client = pymongo.MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]

        # Test kết nối
        client.admin.command('ping')
        print("✅ Kết nối tới MongoDB thành công!")
    except Exception as e:
        print(f"❌ Lỗi kết nối MongoDB: {e}")
        return

    # Dọn dẹp dữ liệu cũ trước khi nạp
    print(f"Đang xóa dữ liệu cũ trong collection '{COLLECTION_NAME}'...")
    collection.delete_many({})

    # Nạp dữ liệu mới
    print(f"Đang nạp {len(documents_to_insert)} bản ghi vào collection '{COLLECTION_NAME}'...")
    try:
        collection.insert_many(documents_to_insert)
        print("✅ Nạp dữ liệu vào MongoDB Local thành công!")
    except Exception as e:
        print(f"❌ Lỗi nạp dữ liệu vào MongoDB: {e}")
        return

    # In thử thông tin vector của phần tử đầu tiên kiểm tra kích thước
    vector_dimension = len(documents_to_insert[0]['vector'])
    print(f"Kích thước (số chiều) của mỗi vector là: {vector_dimension}")

def main():
    # Khởi tạo model AITeamVN/Vietnamese_Embedding_v2
    print("Đang kiểm tra thiết bị phần cứng...")
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    print("Đang tải model từ Hugging Face...")
    model = SentenceTransformer('AITeamVN/Vietnamese_Embedding_v2', device=device)

    # Đường dẫn đến thư mục chứa file python hiện tại
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # File đầu vào chứa đồ thị (nodes và relationships)
    input_file = os.path.join(base_dir, "triplets", "triplets_dat_dai_1.json")

    generate_embeddings_and_load_to_mongo(input_file, model)

if __name__ == "__main__":
    main()