import json
from pathlib import Path
from neo4j import GraphDatabase

# --- CẤU HÌNH ---
URI = "neo4j+s://9a156fb4.databases.neo4j.io"
AUTH = ("9a156fb4", "lSIr4nirHZX4oPudtOO0bQiwUUm3XulJgYuSC5NPNfE")

BASE_DIR = Path(__file__).resolve().parent
FILE_ENTITIES = BASE_DIR / '2_question_entities_vectorized.json'
OUTPUT_FILE = BASE_DIR / '3_results.json'

MIN_SCORE = 0.9  # Ngưỡng an toàn để lọc nhiễu
TOP_K_PER_ENTITY = 1000  # Lấy top 30 cho mỗi từ khóa để đảm bảo độ phủ

def search_by_entities():
    try:
        with open(FILE_ENTITIES, 'r', encoding='utf-8') as f:
            entities_list = json.load(f)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {FILE_ENTITIES}")
        return

    driver = GraphDatabase.driver(URI, auth=AUTH)
    all_results = {}

    print(f"--- Bắt đầu tìm kiếm vector cho {len(entities_list)} thực thể (Sử dụng vector có sẵn) ---")

    with driver.session() as session:
        for item in entities_list:
            word = str(item.get('name', '')).strip()
            query_vector = item.get('vector') # Lấy trực tiếp vector từ file JSON

            # Bỏ qua nếu không có tên hoặc không có vector
            if not word or not query_vector:
                continue

            # Query Neo4j sử dụng vector truyền vào
            cypher_query = """
            CALL db.index.vector.queryNodes('hidden_vector_index', $top_k, $vector)
            YIELD node AS v, score
            CALL {
                WITH v
                MATCH (n) WHERE elementId(n) = v.origin_id RETURN n AS result, labels(n)[0] as label
                UNION
                WITH v
                MATCH ()-[r]->() WHERE elementId(r) = v.origin_id RETURN r AS result, type(r) as label
            }
            RETURN elementId(result) as e_id, result, label, v.text_value as matched_text, score
            """

            records = session.run(
                cypher_query,
                vector=query_vector,
                top_k=TOP_K_PER_ENTITY
            )

            for rec in records:
                e_id = rec['e_id']
                score = rec['score']

                # Chỉ giữ lại kết quả tốt nhất cho mỗi thực thể trong đồ thị
                if e_id not in all_results or score > all_results[e_id]['score']:
                    result = rec['result']
                    all_results[e_id] = {
                        "graph_entity_id": e_id,
                        "name": result.get('name') or rec['label'],
                        "label": rec['label'],
                        "matched_from": word,
                        "matched_text": rec['matched_text'],
                        "score": score,
                        "listSectionId": result.get('listSectionId') or []
                    }

    driver.close()

    # Lọc theo ngưỡng điểm và sắp xếp
    filtered_results = [res for res in all_results.values() if res['score'] > MIN_SCORE]
    sorted_results = sorted(filtered_results, key=lambda x: x['score'], reverse=True)

    # Lưu kết quả
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(sorted_results, f, ensure_ascii=False, indent=2)

    # Hiển thị kết quả ra Terminal
    print("\n" + "="*60)
    print(f"TOP {len(sorted_results)} THỰC THỂ PHÁP LÝ (SCORE > {MIN_SCORE})")
    print("="*60)

    for i, res in enumerate(sorted_results, 1):
        print(f"{i}. [{res['score']:.4f}] {res['name']} ({res['label']})")
        print(f"   -> Khớp: '{res['matched_text']}' (từ từ khóa: '{res['matched_from']}')")
        if res['listSectionId']:
            print(f"   -> Các điều luật liên quan: {res['listSectionId']}")
        print("-" * 60)

if __name__ == "__main__":
    search_by_entities()