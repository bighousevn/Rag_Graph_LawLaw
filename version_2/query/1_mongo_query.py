import pymongo
import json
import os
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. CẤU HÌNH MONGODB
# ==========================================
MONGO_URI = os.getenv("MONGO_URI", "").strip()
DB_NAME = os.getenv("MONGO_DB_NAME", "vectorDB").strip()
COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "vector_entities").strip()
VECTOR_INDEX_NAME = os.getenv("MONGO_VECTOR_INDEX_NAME", "vector_index").strip()
VECTOR_FIELD_PATH = os.getenv("MONGO_VECTOR_FIELD_PATH", "vector").strip()

def main():
    if not MONGO_URI:
        print("❌ Thiếu biến môi trường MONGO_URI trong file .env")
        return

    # Kết nối MongoDB
    try:
        client = pymongo.MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        # Thử ping DB để kiểm tra kết nối
        client.admin.command('ping')
        print("✅ Đã kết nối thành công tới MongoDB!")
    except Exception as e:
        print(f"❌ Lỗi kết nối MongoDB: {e}")
        return

    # Đường dẫn file
    input_file = "./version_2/query/1_entities_with_vectors.json"
    output_file = "./version_2/query/2_output.json"

    # ==========================================
    # 2. ĐỌC DỮ LIỆU CÂU HỎI ĐẦU VÀO
    # ==========================================
    if not os.path.exists(input_file):
        print(f"❌ Lỗi: Không tìm thấy file {input_file}")
        return

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            query_items = json.load(f)
    except Exception as e:
        print(f"❌ Lỗi đọc file {input_file}: {e}")
        return

    all_results = []
    print(f"\nĐang xử lý tìm kiếm cho {len(query_items)} thực thể trong câu hỏi...\n" + "="*50)

    # ==========================================
    # 3. LẶP QUÉT TỪNG ENTITY VÀ TÌM KIẾM VECTOR
    # ==========================================
    for item in query_items:
        query_id = item.get("id")
        query_name = item.get("name")
        query_vector = item.get("vector")

        if not query_vector:
            print(f"[!] Bỏ qua '{query_name}' vì không có vector.")
            continue

        # Xây dựng Aggregation Pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "index": VECTOR_INDEX_NAME,
                    "path": VECTOR_FIELD_PATH,
                    "queryVector": query_vector,
                    "numCandidates": 100,  # Số lượng ứng viên quét qua ban đầu
                    "limit": 20            # Giới hạn số kết quả trả về tối đa trước khi lọc
                }
            },
            {
                # Lấy ra các trường cần thiết và tính toán điểm tương đồng (score)
                "$project": {
                    "_id": 0,
                    "entityId": 1,         # Trả về ID của thực thể chuẩn
                    "entityName": 1,       # Trả về Tên thực thể chuẩn
                    "synonym": 1,          # Trả về Từ khóa thực tế trong DB bị khớp
                    "score": { "$meta": "vectorSearchScore" }
                }
            },
            {
                # BỘ LỌC QUAN TRỌNG: Chỉ lấy kết quả có độ tương đồng > 0.85
                "$match": {
                    "score": { "$gt": 0.85 }
                }
            }
        ]

        # Thực thi truy vấn với MongoDB
        try:
            matches = list(collection.aggregate(pipeline))

            # Nếu có kết quả > 0.85, lưu vào mảng tổng
            if matches:
                all_results.append({
                    "query_id": query_id,
                    "query_name": query_name,
                    "total_matches": len(matches),
                    "matched_items": matches
                })
                print(f" 🟢 '{query_name}': Tìm thấy {len(matches)} kết quả khớp (> 0.85)")
            else:
                print(f" 🔴 '{query_name}': Không có kết quả nào đạt điểm > 0.85")

        except Exception as e:
            print(f" ❌ Lỗi khi truy vấn cho '{query_name}': {e}")

    # ==========================================
    # 4. XUẤT RA FILE OUTPUT.JSON
    # ==========================================
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

    print("\n" + "="*50)
    print(f"🎉 Hoàn tất! Đã lưu toàn bộ kết quả tra cứu tại: {output_file}")

if __name__ == "__main__":
    main()
