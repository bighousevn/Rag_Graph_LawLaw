import json
import time
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

# --- CẤU HÌNH KẾT NỐI ---
URI = "neo4j+s://78e068a6.databases.neo4j.io"  # Hoặc địa chỉ Neo4j Aura của bạn
USER = "78e068a6"
PASSWORD = "wbZSEmNb6V54rlRhl2VhlSPq-V7p0p1K1EhRQb9GZ2o"
FILE_JSON = 'duc/nodesData_embedded.json' # File chứa [ {"id": "...", "embedding": [...]}, ... ]

class Neo4jUpdater:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def update_batch(self, nodes):
        with self.driver.session() as session:
            # Sử dụng UNWIND để biến mảng JSON thành các dòng dữ liệu trong Cypher
            query = """
            UNWIND $batch AS row
            MATCH (n)
            WHERE elementId(n) = row.elementId
            SET n.embedding = row.embedding
            RETURN count(n) as updated
            """
            start_time = time.time()
            result = session.run(query, batch=nodes)
            count = result.single()["updated"]
            end_time = time.time()
            print(f">>> Đã cập nhật {count} nodes trong {end_time - start_time:.2f} giây.")

# --- LUỒNG THỰC THI ---
if __name__ == "__main__":
    try:
        # 1. Đọc dữ liệu từ file JSON
        with open(FILE_JSON, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 2. Kết nối và cập nhật
        updater = Neo4jUpdater(URI, USER, PASSWORD)
        updater.update_batch(data)
        updater.close()
        
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {FILE_JSON}")
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")