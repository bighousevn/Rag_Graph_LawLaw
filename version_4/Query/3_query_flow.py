import json
import os
import pymongo
from dotenv import load_dotenv

load_dotenv()

# ── CONFIG PATHS ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRIPLETS_IN = os.path.join(BASE_DIR, "output", "1.3_triplets_with_nested_vectors.json")
FILTERED_OUT = os.path.join(BASE_DIR, "output", "5_filtered_triplets.json")

# Đọc file đồ thị tương ứng với cơ sở dữ liệu đã nạp trong MongoDB
GRAPH_FILE = os.path.join(os.path.dirname(BASE_DIR), "Building_KG", "triplets", "triplets_dat_dai_1.json")

# ── MONGODB CONFIG ───────────────────────────────────────────────────────────
MONGO_URI = os.getenv("MONGO_URI", "").strip()
DB_NAME = os.getenv("MONGO_DB_NAME", "vectorDB").strip()
COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "vector_entities").strip()
VECTOR_INDEX_NAME = os.getenv("MONGO_VECTOR_INDEX_NAME", "vector_index").strip()
VECTOR_FIELD_PATH = os.getenv("MONGO_VECTOR_FIELD_PATH", "vector").strip()
SCORE_THRESHOLD = float(os.getenv("MONGO_VECTOR_SCORE_THRESHOLD", "0.85"))
SEARCH_LIMIT = int(os.getenv("MONGO_VECTOR_SEARCH_LIMIT", "20"))
NUM_CANDIDATES = int(os.getenv("MONGO_VECTOR_NUM_CANDIDATES", "100"))

def load_graph(graph_file):
    with open(graph_file, "r", encoding="utf-8") as f:
        raw_graph = json.load(f)

    graph_data = raw_graph[0] if isinstance(raw_graph, list) and raw_graph else raw_graph
    if not isinstance(graph_data, dict):
        raise ValueError(f"Graph file không đúng định dạng: {graph_file}")

    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("relationships", [])

    nodes_by_id = {node["id"]: node for node in nodes if node.get("id")}
    edges_by_id = {edge["id"]: edge for edge in edges if edge.get("id")}

    return graph_data, nodes_by_id, edges_by_id

def mongo_vector_search(db_collection, query_vector, entity_type=None, limit=SEARCH_LIMIT) -> dict:
    """Tìm kiếm các thực thể/quan hệ tương đồng từ MongoDB dựa trên Vector Search."""
    if not query_vector:
        return {}

    vector_search_stage = {
        "index": VECTOR_INDEX_NAME,
        "path": VECTOR_FIELD_PATH,
        "queryVector": query_vector,
        "numCandidates": NUM_CANDIDATES,
        "limit": limit
    }

    # Sử dụng tính năng filter trường 'type' nếu được chỉ định
    if entity_type:
        vector_search_stage["filter"] = {
            "type": entity_type
        }

    pipeline = [
        {
            "$vectorSearch": vector_search_stage
        },
        {
            "$project": {
                "_id": 0,
                "entityId": 1,
                "entityName": 1,
                "synonym": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        },
        {
            "$match": {"score": {"$gt": SCORE_THRESHOLD}}
        }
    ]

    try:
        matches = list(db_collection.aggregate(pipeline))
        candidates = {}
        for match in matches:
            entity_id = match.get("entityId")
            if not entity_id:
                continue

            current = candidates.get(entity_id)
            if current is None or match.get("score", 0) > current.get("score", 0):
                candidates[entity_id] = match

        return candidates
    except Exception as e:
        print(f"Lỗi truy vấn MongoDB: {e}")
        return {}

def candidate_ids(candidates):
    return set(candidates.keys())

def compact_candidate(candidate, graph_item=None):
    data = {
        "id": candidate.get("entityId"),
        "entityName": candidate.get("entityName"),
        "synonym": candidate.get("synonym"),
        "score": candidate.get("score"),
    }
    if graph_item:
        data["graphName"] = graph_item.get("name")
    return data

def build_matched_triplet_record(query_triplet, edge, source_node, target_node, s_match, v_match, o_match):
    return {
        "query_triplet": {
            "s": query_triplet.get("s", {}).get("name", ""),
            "v": query_triplet.get("v", {}).get("name", ""),
            "o": query_triplet.get("o", {}).get("name", ""),
        },
        "matched_triplet": {
            "s": {
                "id": source_node.get("id"),
                "name": source_node.get("name"),
            },
            "v": {
                "id": edge.get("id"),
                "name": edge.get("name"),
                "source": edge.get("source"),
                "target": edge.get("target"),
                "listSectionId": edge.get("listSectionId", []),
            },
            "o": {
                "id": target_node.get("id"),
                "name": target_node.get("name"),
            },
        },
        "match_scores": {
            "s": s_match.get("score") if s_match else None,
            "v": v_match.get("score") if v_match else None,
            "o": o_match.get("score") if o_match else None,
        },
        "matched_by": {
            "s": compact_candidate(s_match, source_node) if s_match else None,
            "v": compact_candidate(v_match, edge) if v_match else None,
            "o": compact_candidate(o_match, target_node) if o_match else None,
        },
        "edge": edge,
        "source_node": source_node,
        "target_node": target_node,
    }

def run_pipeline():
    print("="*70)
    print("🚀 QUERY FLOW PIPELINE — SO KHỚP CÂU HỎI VỚI ĐỒ THỊ BẰNG VECTOR & ĐỒ THỊ")
    print("="*70)

    # 1. Kiểm tra file câu hỏi đầu vào đã có vector
    if not os.path.exists(TRIPLETS_IN):
        print(f"❌ Không tìm thấy file input triplets chứa vector: {TRIPLETS_IN}")
        return

    with open(TRIPLETS_IN, "r", encoding="utf-8") as f:
        input_triplets = json.load(f)
    print(f"📥 Đã tải {len(input_triplets)} triplets câu hỏi từ: {TRIPLETS_IN}")

    if not MONGO_URI:
        print("❌ Thiếu cấu hình MONGO_URI trong .env")
        return

    # 2. Kết nối MongoDB
    client = pymongo.MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    try:
        client.admin.command('ping')
        print("✅ Kết nối thành công tới MongoDB Local!")
    except Exception as e:
        print(f"❌ Lỗi kết nối tới MongoDB Local: {e}")
        return

    # 3. Nạp dữ liệu graph local (nodes và edges)
    print(f"\n📌 Đang nạp file đồ thị local từ: {GRAPH_FILE}")
    try:
        graph_data, all_nodes, all_edges_by_id = load_graph(GRAPH_FILE)
    except Exception as e:
        print(f"❌ Không thể nạp đồ thị local: {e}")
        return

    all_edges = graph_data.get("relationships", [])
    print(f"   => Đã nạp thành công: {len(all_nodes)} nodes | {len(all_edges)} edges từ local")

    # 4. Tìm kiếm vector và thực hiện lọc Strict Triplet
    print("\n📌 BẮT ĐẦU VECTOR SEARCH VÀ SO KHỚP ĐỒ THỊ...")

    valid_edges = []
    valid_node_ids = set()
    section_ids = set()
    matched_triplets = []
    matched_triplets_by_query = []

    for i, t in enumerate(input_triplets):
        print(f"\n▶ Xử lý Triplet #{i+1}:")
        s_obj = t.get("s", {})
        v_obj = t.get("v", {})
        o_obj = t.get("o", {})

        s_name = s_obj.get("name", "")
        v_name = v_obj.get("name", "")
        o_name = o_obj.get("name", "")

        # Thực hiện vector search trên MongoDB với phân loại type là node hay relationship
        s_candidates = mongo_vector_search(collection, s_obj.get("vector"), entity_type="node") if s_name != "*" else {}
        v_candidates = mongo_vector_search(collection, v_obj.get("vector"), entity_type="relationship") if v_name != "*" else {}
        o_candidates = mongo_vector_search(collection, o_obj.get("vector"), entity_type="node") if o_name != "*" else {}

        # Nếu thành phần nào là "*" (khuyết), cho phép khớp với mọi ID tương ứng trong graph
        s_ids = candidate_ids(s_candidates) if s_name != "*" else set(all_nodes.keys())
        v_ids = candidate_ids(v_candidates) if v_name != "*" else set(all_edges_by_id.keys())
        o_ids = candidate_ids(o_candidates) if o_name != "*" else set(all_nodes.keys())

        print(f"   [s] '{s_name}' -> {len(s_ids)} candidates")
        print(f"   [v] '{v_name}' -> {len(v_ids)} candidates")
        print(f"   [o] '{o_name}' -> {len(o_ids)} candidates")

        query_matches = []
        # Giao các cạnh hợp lệ (V) giữa kết quả search của Mongo và Edges thực tế trong đồ thị
        graph_edge_ids = v_ids.intersection(all_edges_by_id.keys())
        
        # Sắp xếp các cạnh theo điểm số tìm kiếm nếu có
        if v_name != "*":
            sorted_graph_edge_ids = sorted(
                graph_edge_ids,
                key=lambda edge_id: v_candidates[edge_id].get("score", 0),
                reverse=True,
            )
        else:
            sorted_graph_edge_ids = list(graph_edge_ids)

        for edge_id in sorted_graph_edge_ids:
            edge = all_edges_by_id.get(edge_id)
            if not edge:
                continue

            src_id = edge.get("source")
            tgt_id = edge.get("target")

            # KIỂM TRA ĐẦU ĐUÔI CỦA CẠNH (source/target) CÓ TỒN TẠI TRONG DANH SÁCH MATCH CỦA S/O KHÔNG
            if src_id not in s_ids:
                continue

            if tgt_id not in o_ids:
                continue

            # Thỏa mãn điều kiện khớp: Lấy triplet đó!
            source_node = all_nodes.get(src_id, {"id": src_id, "name": "Unknown"})
            target_node = all_nodes.get(tgt_id, {"id": tgt_id, "name": "Unknown"})
            
            s_match = s_candidates.get(src_id) if s_name != "*" else {"entityId": src_id, "entityName": source_node.get("name"), "synonym": "", "score": 1.0}
            v_match = v_candidates.get(edge_id) if v_name != "*" else {"entityId": edge_id, "entityName": edge.get("name"), "synonym": "", "score": 1.0}
            o_match = o_candidates.get(tgt_id) if o_name != "*" else {"entityId": tgt_id, "entityName": target_node.get("name"), "synonym": "", "score": 1.0}

            match_record = build_matched_triplet_record(
                t,
                edge,
                source_node,
                target_node,
                s_match,
                v_match,
                o_match
            )

            valid_edges.append(edge)
            valid_node_ids.update([src_id, tgt_id])
            section_ids.update(edge.get("listSectionId", []))
            matched_triplets.append(match_record)
            query_matches.append(match_record)

            print(f"   => KHỚP GRAPH: ({source_node.get('name')})-[{edge.get('name')}]->({target_node.get('name')})")

        if not query_matches:
            print("   => ❌ Không tìm thấy quan hệ tương ứng khớp trong đồ thị.")

        matched_triplets_by_query.append({
            "query_index": i,
            "query_triplet": {"s": s_name, "v": v_name, "o": o_name},
            "matches": query_matches,
        })

    # Cộng dồn và sắp xếp danh sách section_id tìm kiếm được
    sorted_sids = sorted(list(section_ids))
    unique_edges = {e["id"]: e for e in valid_edges}.values()
    valid_nodes = [all_nodes[nid] for nid in valid_node_ids if nid in all_nodes]

    print("\n" + "="*50)
    print("🎯 KẾT QUẢ TÌM KIẾM CUỐI CÙNG:")
    print(f"   - Số lượng nodes khớp: {len(valid_nodes)}")
    print(f"   - Số lượng edges khớp: {len(unique_edges)}")
    print(f"   - Danh sách Section ID tìm thấy (Cộng dồn): {sorted_sids}")
    print("="*50 + "\n")

    # 5. Lưu kết quả ra file JSON
    filtered_data = [{
        "nodes": valid_nodes,
        "relationships": list(unique_edges),
        "relevant_section_ids": sorted_sids,
        "matched_triplets": matched_triplets,
        "matched_triplets_by_query": matched_triplets_by_query,
    }]
    
    os.makedirs(os.path.dirname(FILTERED_OUT), exist_ok=True)
    with open(FILTERED_OUT, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=4)
    print(f"💾 Đã lưu kết quả truy vấn tại: {FILTERED_OUT}")

if __name__ == "__main__":
    run_pipeline()
