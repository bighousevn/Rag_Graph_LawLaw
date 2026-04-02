import json
from neo4j import GraphDatabase

# 1. Load dữ liệu từ file JSON thành List of Dictionaries trong Python
with open("./data/1_data.json", "r", encoding="utf-8") as file:
    json_data = json.load(file)

# 2. Câu lệnh Cypher xử lý toàn bộ logic tạo Graph
# Lưu ý: Truyền toàn bộ json_data qua tham số $batch
cypher_query = """
UNWIND $batch AS triplet

// --- 1. TẠO SOURCE NODE ---
MERGE (src:Entity {id: triplet.source.id})
SET src.name = triplet.source.name,
    src.synonym = triplet.source.synonym,
    src.label = triplet.source.label,
    src.type = triplet.source.type

// --- 2. TẠO TARGET NODE ---
MERGE (tgt:Entity {id: triplet.target.id})
SET tgt.name = triplet.target.name,
    tgt.synonym = triplet.target.synonym,
    tgt.label = triplet.target.label,
    tgt.type = triplet.target.type

// --- 3. TẠO CẠNH (EDGE) VÀ GẮN THUỘC TÍNH ---
WITH src, tgt, triplet.edge AS edgeData

// Dùng apoc.merge.relationship để tạo cạnh với tên linh hoạt (lấy từ edgeData.label)
// Đồng thời chống trùng lặp cạnh bằng thuộc tính id
CALL apoc.merge.relationship(
  src, 
  edgeData.label, 
  {id: edgeData.id}, // Tiêu chí gộp (MERGE) để không tạo cạnh trùng
  {
    name: edgeData.name,
    synonym: edgeData.synonym,
    type: edgeData.type
  }, 
  tgt, 
  {}
) YIELD rel

RETURN count(rel)
"""

# 3. Kết nối và thực thi
URI = "neo4j+s://9a156fb4.databases.neo4j.io"
AUTH = ("9a156fb4", "lSIr4nirHZX4oPudtOO0bQiwUUm3XulJgYuSC5NPNfE")

def build_graph(tx, batch_data):
    tx.run(cypher_query, batch=batch_data)

with GraphDatabase.driver(URI, auth=AUTH) as driver:
    with driver.session() as session:
        # Nếu data quá lớn (hàng triệu records), bạn nên cắt json_data thành các chunks (ví dụ 5000 items/lần) 
        # rồi gọi session.execute_write nhiều lần.
        session.execute_write(build_graph, json_data)
        print("Import dữ liệu đồ thị thành công!")