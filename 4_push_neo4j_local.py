import json
import os
from neo4j import GraphDatabase

# === CẬP NHẬT THÔNG TIN ĐĂNG NHẬP LOCAL NEO4J CỦA BẠN ===
URI = "neo4j://127.0.0.1:7687"
AUTH = ("neo4j", "123123123") # <<< HÃY THAY "password" BẰNG MẬT KHẨU CỦA DBMS LOCAL MÀ BẠN ĐÃ TẠO

JSON_PATH = "output/3_graphdata.json"

def import_graph_data():
    print(f"Reading data from {JSON_PATH}...")
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)[0] # Dữ liệu đang được bọc trong 1 list []
        
    nodes = data.get("nodes", [])
    relationships = data.get("relationships", [])
    
    print(f"Loaded {len(nodes)} nodes and {len(relationships)} relationships.")
    
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        # Kiểm tra kết nối
        try:
            driver.verify_connectivity()
            print("Connected to Neo4j successfully!")
        except Exception as e:
            print("\n[LỖI KẾT NỐI] Không thể kết nối Neo4j.")
            print("1. Hãy chắc chắn bạn đã BẤT 'Start' cái database trong Neo4j Desktop chưa?")
            print("2. Đảm bảo Username/Password của Database là đúng ở dòng 6 của file này.")
            print(f"Chi tiết lỗi: {e}")
            return
            
        with driver.session(database="test") as session:
            # 1. Tạo Constraint/Index để lệnh MATCH siêu nhanh (O(1)) khi import cạnh
            print("Creating index on Entity id...")
            session.run("CREATE CONSTRAINT If NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
            
            # 2. Xóa Graph cũ (Bỏ comment 2 dòng dưới nếu muốn reset Graph trắng xóa trước khi nạp)
            # print("Clearing old database...")
            # session.run("MATCH (n) DETACH DELETE n")
            
            # 3. Nạp Nodes
            print("Importing nodes (Pha 1/2)...")
            node_query = """
            UNWIND $batch AS node
            MERGE (n:Entity {id: node.id})
            SET n.name = node.name,
                n.list_section_id = node.list_section_id
            """
            session.run(node_query, batch=nodes)
            
            # 4. Nạp Relationships
            # Lưu ý quan trọng: Lệnh Cypher mặc định không cho biến truyền thay thế vào tên Relationship type (như -[r:$type]->)
            # Thay vì bắt bạn phải cài plugin APOC như đoạn code cũ trên máy local rất mệt mỏi,
            # Ta dùng thủ thuật Render thẳng relation name trong vòng lặp Python.
            print("Importing relationships (Pha 2/2)...")
            count = 0
            for rel in relationships:
                # Xóa dấu backtick (`) phòng trường hợp chuỗi lỗi trong tên node
                rel_type = rel['name'].replace('`', '') 
                if not rel_type: 
                     rel_type = "UNKNOWN_REL"
                
                query = f"""
                MATCH (src:Entity {{id: $source_id}})
                MATCH (tgt:Entity {{id: $target_id}})
                MERGE (src)-[r:`{rel_type}` {{id: $rel_id}}]->(tgt)
                SET r.name = $rel_name,
                    r.list_section_id = $list_section_id
                """
                session.run(query, 
                            source_id=rel['source'], 
                            target_id=rel['target'], 
                            rel_id=rel['id'],
                            rel_name=rel['name'],
                            list_section_id=rel.get('list_section_id', []))
                count += 1
                if count % 500 == 0:
                     print(f"  -> Processed {count}/{len(relationships)} edges...")
            
            print(f"XONG! Đã import thành công đồ thị vào Neo4j Local.")

if __name__ == "__main__":
    import_graph_data()
