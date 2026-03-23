import json
from pathlib import Path

from neo4j import GraphDatabase


# 1. CẤU HÌNH KẾT NỐI NEO4J (Thay bằng thông tin thật của bạn)
URI = "neo4j+s://78e068a6.databases.neo4j.io"  
USER = "78e068a6"
PASSWORD = "wbZSEmNb6V54rlRhl2VhlSPq-V7p0p1K1EhRQb9GZ2o"
driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

QUESTION_NODE_VECTOR_FILE = Path(__file__).with_name("question_node_vector.json")


def doc_danh_sach_vector_tu_file(json_path: Path):
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    danh_sach_vector = []
    for item in data:
        embedding = item.get("embedding")
        if isinstance(embedding, list) and embedding:
            danh_sach_vector.append(embedding)

    if not danh_sach_vector:
        raise ValueError(f"Khong tim thay embedding hop le trong file: {json_path}")

    return danh_sach_vector


# 2. VIẾT CÂU LỆNH CYPHER CÓ CHỨA BIẾN THAM SỐ $vector_list
cypher_query = """
UNWIND $vector_list AS single_vector
CALL db.index.vector.queryNodes('all_nodes_index', 10, single_vector)
YIELD node AS seed_node, score
WITH DISTINCT seed_node, MAX(score) AS max_score
RETURN seed_node.name AS Ten_Nut, max_score AS Diem_Tuong_Dong
ORDER BY max_score DESC
LIMIT 10
"""


# 3. HÀM THỰC THI TRUY VẤN
def lay_top_10_nut_hat_giong(danh_sach_vector):
    with driver.session() as session:
        ket_qua = session.run(cypher_query, vector_list=danh_sach_vector)

        print("--- TOP 10 NUT HAT GIONG TIM DUOC ---")
        for record in ket_qua:
            print(f"- Nut: '{record['Ten_Nut']}' | Do tuong dong: {record['Diem_Tuong_Dong']:.4f}")


# 4. CHẠY CHƯƠNG TRÌNH
if __name__ == "__main__":
    try:
        danh_sach_vector_cau_hoi = doc_danh_sach_vector_tu_file(QUESTION_NODE_VECTOR_FILE)
        lay_top_10_nut_hat_giong(danh_sach_vector_cau_hoi)
    except Exception as e:
        print(f"Loi truy van: {e}")
    finally:
        driver.close()
