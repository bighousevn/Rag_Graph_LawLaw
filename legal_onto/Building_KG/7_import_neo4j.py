"""
Phase 6a — Nạp graph vào Neo4j (KHÔNG tự chạy — bấm khi DB sẵn sàng)
====================================================================
Input: data/master_graph_optimized.json
.env : NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD   (cần APOC trên Neo4j)

Khác bản v4:
  - Node mang nhãn :Concept + thêm nhãn theo category (PHUONG_TIEN, CHU_THE...).
  - Cạnh dùng TYPE đã ASCII-hóa từ tên quan hệ (vd "Điều khiển" -> DIEU_KHIEN),
    giữ tên gốc + relId + weight + listSectionId làm thuộc tính.
"""

import os
import re
import json
import unicodedata
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()
BASE = os.path.dirname(os.path.abspath(__file__))
GRAPH_IN = os.path.join(BASE, "data", "master_graph_optimized.json")


def rel_type(name):
    """'Điều khiển' -> 'DIEU_KHIEN' ; 'Chở / Vận chuyển' -> 'CHO_VAN_CHUYEN'."""
    s = unicodedata.normalize("NFD", name)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.replace("đ", "d").replace("Đ", "D")
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").upper()
    return s or "REL"


CYPHER_NODES = """
UNWIND $batch AS node
MERGE (n:Concept {id: node.id})
SET n.name = node.name,
    n.category = node.category,
    n.listSectionId = node.listSectionId
WITH n, node
CALL apoc.create.addLabels(n, [node.category]) YIELD node AS _ignored
RETURN count(n) AS c
"""

CYPHER_RELS = """
UNWIND $batch AS rel
MATCH (s:Concept {id: rel.source})
MATCH (t:Concept {id: rel.target})
CALL apoc.merge.relationship(
  s, rel.relType, {id: rel.id},
  {name: rel.name, relId: rel.relId, weight: rel.weight,
   count: rel.count, listSectionId: rel.listSectionId},
  t, {}
) YIELD rel AS r
RETURN count(r) AS c
"""


def batches(data, size):
    for i in range(0, len(data), size):
        yield data[i:i + size]


def main():
    with open(GRAPH_IN, "r", encoding="utf-8") as f:
        graph = json.load(f)
    nodes = graph["nodes"]
    edges = graph["relationships"]
    for e in edges:
        e["relType"] = rel_type(e["name"])

    print(f"Nodes: {len(nodes)} | Edges: {len(edges)}")

    uri = os.getenv("NEO4J_URI", "").strip()
    user = os.getenv("NEO4J_USERNAME", "").strip()
    pwd = os.getenv("NEO4J_PASSWORD", "").strip()
    if not (uri and user and pwd):
        raise ValueError("Thiếu NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD trong .env")

    with GraphDatabase.driver(uri, auth=(user, pwd)) as driver:
        with driver.session() as s:
            s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Concept) REQUIRE n.id IS UNIQUE")
        print("--- Import NODES ---")
        with driver.session() as s:
            for i, b in enumerate(batches(nodes, 1000), 1):
                s.execute_write(lambda tx: tx.run(CYPHER_NODES, batch=b).consume())
                print(f"  node batch {i} ({len(b)})")
        print("--- Import RELATIONSHIPS ---")
        with driver.session() as s:
            for i, b in enumerate(batches(edges, 1000), 1):
                s.execute_write(lambda tx: tx.run(CYPHER_RELS, batch=b).consume())
                print(f"  rel batch {i} ({len(b)})")
    print("✅ Import Neo4j xong.")


if __name__ == "__main__":
    main()
