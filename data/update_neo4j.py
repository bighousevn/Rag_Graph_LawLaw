import json
import time
from neo4j import GraphDatabase

# --- CẤU HÌNH KẾT NỐI ---
URI = "neo4j+s://9a156fb4.databases.neo4j.io"
USER = "9a156fb4"  # Username mặc định của Aura thường là 'neo4j'
PASSWORD = "lSIr4nirHZX4oPudtOO0bQiwUUm3XulJgYuSC5NPNfE"
FILE_JSON = 'data/b1_vectors.json'

class Neo4jUpdater:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def update_hidden_vectors(self, data):
        # Query tạo các Node Vector độc lập, không dùng quan hệ để "ẩn" đồ thị
        query = """
        UNWIND $batch AS row
        
        // 1. Cập nhật synonyms (text) vào Node hoặc Edge gốc để search từ khóa
        CALL {
            WITH row
            MATCH (n) WHERE elementId(n) = row.elementId 
            SET n.synonyms = row.synonym
            RETURN 'node' AS kind
            UNION
            WITH row
            MATCH ()-[r]->() WHERE elementId(r) = row.elementId 
            SET r.synonyms = row.synonym
            RETURN 'edge' AS kind
        }
        
        // 2. Với mỗi vector trong mảng lồng nhau, tạo một Node HiddenVector riêng
        WITH row, kind
        UNWIND range(0, size(row.vectors) - 1) AS idx
        CREATE (v:HiddenVector {
            embedding: row.vectors[idx],
            origin_id: row.elementId,
            origin_kind: kind,
            text_value: CASE WHEN idx = 0 THEN row.name ELSE row.synonym[idx-1] END,
            vector_type: CASE WHEN idx = 0 THEN 'primary' ELSE 'synonym' END
        })
        RETURN count(v) AS created_count
        """
        
        start_time = time.time()
        try:
            with self.driver.session() as session:
                result = session.run(query, batch=data)
                # Tính tổng số node vector đã tạo
                total_v = sum([record["created_count"] for record in result])
                
                end_time = time.time()
                print(f">>> Đã tạo {total_v} Node Vector ẩn cho {len(data)} thực thể.")
                print(f">>> Thời gian thực hiện: {end_time - start_time:.2f} giây.")
        except Exception as e:
            print(f"Lỗi khi thực thi query: {e}")

# --- LUỒNG THỰC THI ---
if __name__ == "__main__":
    try:
        print(f"Đang đọc dữ liệu từ {FILE_JSON}...")
        with open(FILE_JSON, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        updater = Neo4jUpdater(URI, USER, PASSWORD)
        print(f"Bắt đầu xử lý {len(data)} đối tượng pháp lý...")
        
        updater.update_hidden_vectors(data)
        
        updater.close()
        print("--- Hoàn tất cập nhật cấu trúc Vector ẩn! ---")
        
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {FILE_JSON}")
    except Exception as e:
        print(f"Đã xảy ra lỗi hệ thống: {e}")