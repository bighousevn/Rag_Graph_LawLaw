import json
import math
from pathlib import Path

from neo4j import GraphDatabase


URI = "neo4j+s://78e068a6.databases.neo4j.io"
USER = "78e068a6"
PASSWORD = "wbZSEmNb6V54rlRhl2VhlSPq-V7p0p1K1EhRQb9GZ2o"
driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

BASE_DIR = Path(__file__).resolve().parent
QUESTION_NODE_VECTOR_FILE = BASE_DIR / "question_node_vector.json"
QUESTION_EMBEDDING_FILE = BASE_DIR / "embedding_data.json"
LAW_VECTOR_FILE = BASE_DIR / "law_vector.json"
VECTOR_INDEX_NAME = "all_nodes_index"
SCORE_THRESHOLD = 0.82
TOP_K_PER_VECTOR = 100  # Tăng lượng ứng viên ban đầu để đảm bảo tìm đủ node > 0.82
TOP_SECTION_MATCHES = 5


def doc_json(json_path: Path):
    with json_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def doc_danh_sach_vector_tu_file(json_path: Path):
    data = doc_json(json_path)
    danh_sach_vector = []

    if not isinstance(data, list):
        raise ValueError(f"Dinh dang file khong ho tro de doc danh sach vector: {json_path}")

    for item in data:
        embedding = item.get("embedding")
        if isinstance(embedding, list) and embedding:
            danh_sach_vector.append(embedding)

    if not danh_sach_vector:
        raise ValueError(f"Khong tim thay embedding hop le trong file: {json_path}")

    return danh_sach_vector


def doc_vector_don_tu_file(json_path: Path):
    data = doc_json(json_path)
    if not isinstance(data, dict):
        raise ValueError(f"Dinh dang file khong ho tro de doc vector don: {json_path}")

    embedding = data.get("vector_embedding") or data.get("embedding")
    if not isinstance(embedding, list) or not embedding:
        raise ValueError(f"Khong tim thay vector hop le trong file: {json_path}")

    return embedding


SEED_QUERY = f"""
UNWIND $vector_list AS single_vector
CALL db.index.vector.queryNodes('{VECTOR_INDEX_NAME}', {TOP_K_PER_VECTOR}, single_vector)
YIELD node AS seed_node, score
WITH seed_node, MAX(score) AS max_score
WHERE max_score > $score_threshold
RETURN DISTINCT
    elementId(seed_node) AS seed_id,
    coalesce(seed_node.name, seed_node.title, seed_node.entity_name, elementId(seed_node)) AS seed_name,
    labels(seed_node) AS seed_labels,
    max_score AS vector_score
ORDER BY vector_score DESC
"""


SECTION_ONE_HOP_QUERY = """
UNWIND $seed_node_ids AS seed_id
MATCH (seed)-[rel]-(section_node)
WHERE elementId(seed) = seed_id
  AND section_node.sectionId IS NOT NULL
RETURN DISTINCT
    elementId(seed) AS seed_id,
    coalesce(seed.name, seed.title, seed.entity_name, elementId(seed)) AS seed_name,
    labels(seed) AS seed_labels,
    type(rel) AS relation_type,
    elementId(section_node) AS section_id,
    coalesce(section_node.name, section_node.title, section_node.entity_name, elementId(section_node)) AS section_name,
    section_node.sectionId AS section_value,
    labels(section_node) AS section_labels
"""


def lay_top_seed_nodes(danh_sach_vector):
    with driver.session() as session:
        result = session.run(
            SEED_QUERY,
            vector_list=danh_sach_vector,
            score_threshold=SCORE_THRESHOLD,
        )
        return [record.data() for record in result]


def lay_section_nodes_one_hop(seed_nodes):
    if not seed_nodes:
        return []

    seed_node_ids = [node["seed_id"] for node in seed_nodes]

    with driver.session() as session:
        result = session.run(SECTION_ONE_HOP_QUERY, seed_node_ids=seed_node_ids)
        return [record.data() for record in result]


def lay_danh_sach_section_id(section_nodes):
    return sorted({item["section_value"] for item in section_nodes if item.get("section_value")})


def tinh_cosine_similarity(vector_a, vector_b):
    if len(vector_a) != len(vector_b):
        return None

    tich_vo_huong = sum(a * b for a, b in zip(vector_a, vector_b))
    do_lon_a = math.sqrt(sum(a * a for a in vector_a))
    do_lon_b = math.sqrt(sum(b * b for b in vector_b))

    if do_lon_a == 0 or do_lon_b == 0:
        return None

    return tich_vo_huong / (do_lon_a * do_lon_b)


def lay_top_section_theo_vector(question_vector, section_ids, law_vector_path: Path, top_k=TOP_SECTION_MATCHES):
    if not question_vector or not section_ids:
        return []

    law_vectors = doc_json(law_vector_path)
    section_id_set = set(section_ids)
    ung_vien = []

    for item in law_vectors:
        section_id = item.get("node_id")
        section_vector = item.get("embedding")

        if section_id not in section_id_set:
            continue
        if not isinstance(section_vector, list) or not section_vector:
            continue

        similarity = tinh_cosine_similarity(question_vector, section_vector)
        if similarity is None:
            continue

        ung_vien.append(
            {
                "section_id": section_id,
                "level": item.get("level"),
                "text_content": item.get("text_content"),
                "similarity_score": similarity,
            }
        )

    return sorted(ung_vien, key=lambda item: item["similarity_score"], reverse=True)[:top_k]


def in_seed_nodes(seed_nodes):
    print(f"--- SEED NODES (SCORE > {SCORE_THRESHOLD}) ---")
    if not seed_nodes:
        print("Khong tim thay seed node nao.")
        return

    for node in seed_nodes:
        print(
            f"- Seed: '{node['seed_name']}' | labels={node['seed_labels']} "
            f"| vector_score={node['vector_score']:.4f}"
        )


def in_section_nodes(section_nodes):
    print("\n--- ONE-HOP SECTION NODE (CHI LAY NODE CO sectionId) ---")
    if not section_nodes:
        print("Khong tim thay node one-hop nao co sectionId.")
        return

    for item in section_nodes:
        print(
            f"- ({item['seed_name']}) -[{item['relation_type']}]-> ({item['section_name']}) "
            f"| sectionId={item['section_value']} | labels={item['section_labels']}"
        )


def in_danh_sach_section_id(section_ids):
    print("\n--- DANH SACH sectionId TRONG SUBGRAPH ---")
    if not section_ids:
        print("Khong tim thay sectionId nao.")
        return

    for section_id in section_ids:
        print(f"- {section_id}")


def tao_cypher_do_thi_con(seed_nodes, section_nodes):
    if not section_nodes:
        return None

    node_ids = sorted(
        {node["seed_id"] for node in seed_nodes}
        | {item["section_id"] for item in section_nodes}
    )
    edge_keys = sorted(
        [
            [item["seed_id"], item["relation_type"], item["section_id"]]
            for item in section_nodes
        ],
        key=lambda item: (item[0], item[1], item[2]),
    )

    node_ids_cypher = json.dumps(node_ids, ensure_ascii=True)
    edge_keys_cypher = json.dumps(edge_keys, ensure_ascii=True)

    return f"""WITH {node_ids_cypher} AS kept_node_ids, {edge_keys_cypher} AS kept_edges
MATCH (a)-[r]-(b)
WHERE elementId(a) IN kept_node_ids
  AND elementId(b) IN kept_node_ids
  AND ANY(edge IN kept_edges WHERE
      (elementId(a) = edge[0] AND type(r) = edge[1] AND elementId(b) = edge[2])
      OR
      (elementId(a) = edge[2] AND type(r) = edge[1] AND elementId(b) = edge[0])
  )
RETURN a, r, b"""


def in_cypher_do_thi_con(seed_nodes, section_nodes):
    cypher_subgraph = tao_cypher_do_thi_con(seed_nodes, section_nodes)

    print("\n--- CYPHER LAY DO THI CON ---")
    if not cypher_subgraph:
        print("Khong co Cypher vi khong tim thay section node nao.")
        return

    print(cypher_subgraph)


def in_top_section_matches(top_sections):
    print(f"\n--- TOP {TOP_SECTION_MATCHES} SECTION TUONG DONG NHAT ---")
    if not top_sections:
        print("Khong tim thay section nao phu hop de so khop vector.")
        return

    for index, item in enumerate(top_sections, start=1):
        print(
            f"{index}. sectionId={item['section_id']} | score={item['similarity_score']:.4f} "
            f"| level={item['level']}"
        )
        print(f"   text: {item['text_content']}")


if __name__ == "__main__":
    try:
        danh_sach_vector_cau_hoi = doc_danh_sach_vector_tu_file(QUESTION_NODE_VECTOR_FILE)
        vector_cau_hoi = doc_vector_don_tu_file(QUESTION_EMBEDDING_FILE)

        seed_nodes = lay_top_seed_nodes(danh_sach_vector_cau_hoi)
        section_nodes = lay_section_nodes_one_hop(seed_nodes)
        section_ids = lay_danh_sach_section_id(section_nodes)
        top_sections = lay_top_section_theo_vector(
            question_vector=vector_cau_hoi,
            section_ids=section_ids,
            law_vector_path=LAW_VECTOR_FILE,
        )

        in_seed_nodes(seed_nodes)
        in_section_nodes(section_nodes)
        in_danh_sach_section_id(section_ids)
        
        # Lọc các section_nodes chỉ giữ lại những node có trong top_sections
        top_section_ids = {item["section_id"] for item in top_sections}
        filtered_section_nodes = [node for node in section_nodes if node["section_value"] in top_section_ids]
        
        in_cypher_do_thi_con(seed_nodes, filtered_section_nodes)
        in_top_section_matches(top_sections)
    except Exception as e:
        print(f"Loi truy van: {e}")
    finally:
        driver.close()
