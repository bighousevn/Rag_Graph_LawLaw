import json
import os
from neo4j import GraphDatabase

# 1. Load dữ liệu từ file JSON thành List of Dictionaries trong Python
with open("./data/1_dataForGraph.json", "r", encoding="utf-8") as file:
    json_data = json.load(file)

# 2. Câu lệnh Cypher xử lý toàn bộ logic tạo Graph
cypher_query = """
// ==========================================
// PHA 1: IMPORT TẤT CẢ CÁC NODE
// ==========================================
UNWIND $data AS batch
UNWIND batch.nodes AS node

MERGE (n:Entity {id: node.id})
SET n.name = node.name,
    n.listSectionId = node.listSectionId

// ==========================================
// PHA 2: IMPORT TẤT CẢ RELATIONSHIPS (CẠNH)
// ==========================================
WITH $data AS data 
UNWIND data AS batch
UNWIND batch.relationships AS rel

MATCH (src:Entity {id: rel.source})
MATCH (tgt:Entity {id: rel.target})

CALL apoc.merge.relationship(
  src, 
  rel.name,          
  {id: rel.id},      
  {
    name: rel.name,  
    listSectionId: rel.listSectionId
  }, 
  tgt, 
  {}
) YIELD rel AS r

RETURN count(r) AS TotalEdgesCreated
"""

# 3. Kết nối và thực thi
# LỜI KHUYÊN: Nên đưa thông tin nhạy cảm vào biến môi trường (Environment Variables)
URI = "neo4j+s://9a156fb4.databases.neo4j.io"
AUTH = ("9a156fb4", "lSIr4nirHZX4oPudtOO0bQiwUUm3XulJgYuSC5NPNfE") # <-- Đừng quên đổi mật khẩu trên Aura!

def build_graph(tx, batch_data):
    # Đã sửa 'batch' thành 'data' để khớp với biến $data trong Cypher
    tx.run(cypher_query, data=batch_data)

with GraphDatabase.driver(URI, auth=AUTH) as driver:
    with driver.session() as session:
        session.execute_write(build_graph, json_data)
        print("Import dữ liệu đồ thị thành công!")