import json
import time
from neo4j import GraphDatabase

# --- CẤU HÌNH KẾT NỐI ---
# ⚠️ LƯU Ý BẢO MẬT: Nhớ thay mật khẩu và giấu vào biến môi trường (.env) nhé!
URI = "neo4j+s://9a156fb4.databases.neo4j.io"
USER = "9a156fb4"
PASSWORD = "lSIr4nirHZX4oPudtOO0bQiwUUm3XulJgYuSC5NPNfE" 
FILE_JSON = 'data/2_synonyms_with_vectors.json'

class Neo4jUpdater:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_vector_nodes(self, tx, batch_data):
        # Cypher cực kỳ tinh gọn: Lặp qua từng row và MERGE tạo Node Vector
        query = """
        UNWIND $batch AS row
        
        // Dùng MERGE với synonymId để an toàn, không tạo Node trùng lặp nếu chạy lại script
        MERGE (v:Vector {id: row.synonymId})
        SET v.entityId = row.entityId,           // Lưu ID gốc để liên kết/truy vết
            v.entityName = row.entityName,       // Tên gốc (chỉ để dễ nhìn khi debug)
            v.text_value = row.synonym,          // Từ đồng nghĩa (text)
            v.embedding = row.vector             // Mảng vector 768 chiều
            
        RETURN count(v) AS created_count
        """
        
        result = tx.run(query, batch=batch_data)
        return sum([record["created_count"] for record in result])

# --- LUỒNG THỰC THI ---
if __name__ == "__main__":
    try:
        print(f"Đang đọc dữ liệu từ {FILE_JSON}...")
        with open(FILE_JSON, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        updater = Neo4jUpdater(URI, USER, PASSWORD)
        print(f"Bắt đầu xử lý {len(data)} bản ghi vector...")
        
        # CHIA BATCH KHI LÀM VIỆC VỚI VECTOR (rất quan trọng để tránh tràn RAM)
        batch_size = 200
        start_time_total = time.time()
        
        with updater.driver.session() as session:
            for i in range(0, len(data), batch_size):
                batch = data[i : i + batch_size]
                
                print(f"--- Đang xử lý lô từ {i} đến {i + len(batch)} ---")
                start_time_batch = time.time()
                
                # Gọi hàm thông qua execute_write để đảm bảo an toàn Transaction
                total_created = session.execute_write(updater.create_vector_nodes, batch)
                
                end_time_batch = time.time()
                print(f">>> Đã import {total_created} Node Vector.")
                print(f">>> Thời gian xử lý lô: {end_time_batch - start_time_batch:.2f} giây.\n")
        
        updater.close()
        end_time_total = time.time()
        print(f"--- HOÀN TẤT! Tổng thời gian chạy: {end_time_total - start_time_total:.2f} giây. ---")
        
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {FILE_JSON}")
    except Exception as e:
        print(f"Đã xảy ra lỗi hệ thống: {e}")