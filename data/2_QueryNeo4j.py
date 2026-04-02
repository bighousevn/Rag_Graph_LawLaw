import json
from pathlib import Path
from neo4j import GraphDatabase

# --- CẤU HÌNH ---
URI = "neo4j+s://9a156fb4.databases.neo4j.io"
AUTH = ("9a156fb4", "lSIr4nirHZX4oPudtOO0bQiwUUm3XulJgYuSC5NPNfE")

BASE_DIR = Path(__file__).resolve().parent
FILE_ENTITIES = BASE_DIR / '11_vector.json'
OUTPUT_FILE = BASE_DIR / '3_results.json'

MIN_SCORE = 0.85  # Ngưỡng an toàn để lọc nhiễu
TOP_K_PER_ENTITY = 1000

def search_by_entities():
    try:
        with open(FILE_ENTITIES, 'r', encoding='utf-8') as f:
            entities_list = json.load(f)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {FILE_ENTITIES}")
        return

    driver = GraphDatabase.driver(URI, auth=AUTH)

    # --- BƯỚC 1: TRUY VẤN VECTOR VÀ SO KHỚP ---
    matched_nodes = {}
    matched_edges = {}

    print(f"--- ĐANG SO KHỚP VECTOR CHO {len(entities_list)} THỰC THỂ ---")

    with driver.session() as session:
        cypher_query = """
        CALL db.index.vector.queryNodes('hidden_vector_index', $top_k, $vector)
        YIELD node AS v, score
        CALL (v) {
            MATCH (n) WHERE elementId(n) = v.origin_id
            RETURN 'node' AS entity_type, elementId(n) AS e_id, labels(n)[0] AS label_or_type, properties(n) AS properties, null AS start_node, null AS end_node
            UNION
            MATCH (start)-[r]->(end) WHERE elementId(r) = v.origin_id
            RETURN 'relationship' AS entity_type, elementId(r) AS e_id, type(r) AS label_or_type, properties(r) AS properties,
                { id: elementId(start), labels: labels(start), properties: properties(start) } AS start_node,
                { id: elementId(end), labels: labels(end), properties: properties(end) } AS end_node
        }
        RETURN entity_type, e_id, label_or_type, properties, start_node, end_node, v.text_value AS matched_text, score
        """

        for item in entities_list:
            word = str(item.get('name', '')).strip()
            query_vector = item.get('vector')
            if not word or not query_vector: continue

            records = session.run(cypher_query, vector=query_vector, top_k=TOP_K_PER_ENTITY)

            for rec in records:
                score = rec['score']
                if score <= MIN_SCORE: continue

                entity_type = rec['entity_type']
                e_id = rec['e_id']
                result_data = {
                    "graph_entity_id": e_id,
                    "entity_type": entity_type,
                    "label_or_type": rec['label_or_type'],
                    "properties": rec['properties'] or {},
                    "matched_from": word,
                    "matched_text": rec['matched_text'],
                    "score": score
                }

                if entity_type == 'node':
                    if e_id not in matched_nodes or score > matched_nodes[e_id]['score']:
                        matched_nodes[e_id] = result_data
                elif entity_type == 'relationship':
                    if e_id not in matched_edges or score > matched_edges[e_id]['score']:
                        result_data["start_node"] = rec['start_node']
                        result_data["end_node"] = rec['end_node']
                        matched_edges[e_id] = result_data

    driver.close()

    # --- BƯỚC 2: LỌC BỘ 3 TRIPLET VÀ LÀM PHẲNG KẾT QUẢ ---
    valid_edges = {}
    valid_node_ids = set()

    for e_id, edge in matched_edges.items():
        start_id = edge['start_node']['id']
        end_id = edge['end_node']['id']

        # 1. Chỉ giữ lại Edge nếu CẢ start_node và end_node đều trúng vector (nằm trong matched_nodes)
        if start_id in matched_nodes and end_id in matched_nodes:
            # Đánh dấu 2 Node này là hợp lệ (thuộc bộ 3)
            valid_node_ids.add(start_id)
            valid_node_ids.add(end_id)

            # 2. Làm phẳng dữ liệu Edge: Giữ lại ID để tham chiếu, xóa object lồng nhau
            edge['start_node_id'] = start_id
            edge['end_node_id'] = end_id
            del edge['start_node']
            del edge['end_node']

            valid_edges[e_id] = edge

    # 3. Lọc Node: Loại bỏ các "Node mồ côi" không thuộc bất kỳ Cạnh hợp lệ nào
    valid_nodes = {n_id: matched_nodes[n_id] for n_id in valid_node_ids}

    # Gộp Node và Edge đã làm sạch để xuất JSON
    final_results = list(valid_nodes.values()) + list(valid_edges.values())
    sorted_results = sorted(final_results, key=lambda x: x['score'], reverse=True)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(sorted_results, f, ensure_ascii=False, indent=2)

    # --- IN KẾT QUẢ RA TERMINAL ---
    print("\n" + "="*80)
    print(f" TÌM THẤY {len(valid_edges)} BỘ 3 QUAN HỆ HỢP LỆ (Lọc từ {len(matched_nodes)} Nodes thô)")
    print("="*80)

    # Sắp xếp các Cạnh hợp lệ theo Score để in ra Terminal
    sorted_edges = sorted(valid_edges.values(), key=lambda x: x['score'], reverse=True)

    for i, edge in enumerate(sorted_edges, 1):
        # Truy xuất tên Node từ valid_nodes để log ra Terminal cho dễ đọc
        s_node = valid_nodes[edge['start_node_id']]['properties']
        e_node = valid_nodes[edge['end_node_id']]['properties']

        s_name = s_node.get('name') or s_node.get('id') or s_node.get('title') or 'Start'
        e_name = e_node.get('name') or e_node.get('id') or e_node.get('title') or 'End'

        print(f"{i}. [EDGE - {edge['score']:.4f}] ({s_name}) -[{edge['label_or_type']}]-> ({e_name})")
        print(f"   -> Cạnh khớp với từ khóa: '{edge['matched_from']}'")
        print("-" * 80)

    print(f"\n Đã xuất {len(valid_nodes)} Nodes và {len(valid_edges)} Edges (đã làm phẳng) ra file: {OUTPUT_FILE}")

if __name__ == "__main__":
    search_by_entities()