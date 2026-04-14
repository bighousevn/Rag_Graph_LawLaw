import json
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

# 1. Load dữ liệu từ file JSON
with open("./version_3/2_entitites_per_chunk.json", "r", encoding="utf-8") as file:
    json_data = json.load(file)

# --- XỬ LÝ DATA TRONG PYTHON TRƯỚC ---
# Gom toàn bộ nodes và rels thành 2 mảng phẳng (flat lists)
all_nodes = []
all_relationships = []

for batch in json_data:
    all_nodes.extend(batch.get("nodes", []))
    all_relationships.extend(batch.get("relationships", []))

print(f"Tổng số Nodes cần import: {len(all_nodes)}")
print(f"Tổng số Rels cần import: {len(all_relationships)}")

# 2. Câu lệnh Cypher tách biệt
cypher_import_nodes = """
UNWIND $batch_data AS node
MERGE (n:Entity {id: node.id})
SET n.name = node.name,
    n.listSectionId = node.listSectionId
"""

cypher_import_rels = """
UNWIND $batch_data AS rel
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

# 3. Kết nối
URI = os.getenv("NEO4J_URI", "").strip()
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "").strip()
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "").strip()

if not URI or not NEO4J_USERNAME or not NEO4J_PASSWORD:
    raise ValueError(
        "Thiếu biến môi trường Neo4j. Hãy thiết lập NEO4J_URI, "
        "NEO4J_USERNAME và NEO4J_PASSWORD trong file .env."
    )

AUTH = (NEO4J_USERNAME, NEO4J_PASSWORD)
# 4. Hàm xử lý Batching (Chia nhỏ dữ liệu)
def process_in_batches(driver, query, data_list, batch_size=2000):
    total_batches = (len(data_list) + batch_size - 1) // batch_size
    with driver.session() as session:
        for i in range(total_batches):
            start_index = i * batch_size
            end_index = start_index + batch_size
            batch_data = data_list[start_index:end_index]

            # Gửi và commit từng chunk nhỏ
            session.execute_write(lambda tx: tx.run(query, batch_data=batch_data))
            print(f"  -> Đã xử lý batch {i+1}/{total_batches} ({len(batch_data)} records)")

# 5. Thực thi toàn bộ quy trình
with GraphDatabase.driver(URI, auth=AUTH) as driver:
    with driver.session() as session:
        # Bước cực kỳ quan trọng: Tạo Constraint/Index để MATCH node siêu tốc
        print("Thiết lập Constraint/Index...")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE")

    print("\n--- BẮT ĐẦU IMPORT NODES ---")
    # Batch size cho Node có thể để lớn vì nó nhẹ (5000)
    process_in_batches(driver, cypher_import_nodes, all_nodes, batch_size=5000)

    print("\n--- BẮT ĐẦU IMPORT RELATIONSHIPS ---")
    # Batch size cho Rel nên để nhỏ hơn vì APOC consume nhiều memory hơn (2000)
    process_in_batches(driver, cypher_import_rels, all_relationships, batch_size=2000)

    print("\n✅ Import dữ liệu đồ thị thành công!")
