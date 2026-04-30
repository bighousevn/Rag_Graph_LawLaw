"""
Query Flow Pipeline
-------------------
Input   : ./version_3/query/1.3_triplets_with_nested_vectors.json
          (Danh sách triplet câu hỏi, mỗi S/V/O có "name" và "vector")
Process :
          - Với từng triplet câu hỏi, vector search riêng S, V, O để lấy candidate IDs.
          - Dùng graph local để lấy edge data đầy đủ, gồm source/target/listSectionId.
          - Lọc strict: edge thuộc V candidates, source thuộc S candidates,
            target thuộc O candidates.
Output  : ./version_3/query/5_filtered_triplets.json
"""

import json
import os
import pymongo
import certifi
from dotenv import load_dotenv

load_dotenv()

# ── CẤU HÌNH ĐƯỜNG DẪN ────────────────────────────────────────────────────────
BASE             = "./version_3"
TRIPLETS_IN      = f"{BASE}/query/1.3_triplets_with_nested_vectors.json"  # File User cung cấp
FILTERED_OUT     = f"{BASE}/query/5_filtered_triplets.json"
GRAPH_FILE       = f"{BASE}/2_final_graph.json"

# ── MONGODB CONFIG ───────────────────────────────────────────────────
MONGO_URI            = os.getenv("MONGO_URI", "").strip()
DB_NAME              = os.getenv("MONGO_DB_NAME", "vectorDB").strip()
COLLECTION_NAME      = os.getenv("MONGO_COLLECTION_NAME", "vector_entities").strip()
VECTOR_INDEX_NAME    = os.getenv("MONGO_VECTOR_INDEX_NAME", "vector_index").strip()
VECTOR_FIELD_PATH    = os.getenv("MONGO_VECTOR_FIELD_PATH", "vector").strip()
SCORE_THRESHOLD      = float(os.getenv("MONGO_VECTOR_SCORE_THRESHOLD", "0.85"))
SEARCH_LIMIT         = int(os.getenv("MONGO_VECTOR_SEARCH_LIMIT", "20"))
NUM_CANDIDATES       = int(os.getenv("MONGO_VECTOR_NUM_CANDIDATES", "100"))

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

def mongo_vector_search(db_collection, query_vector, limit=SEARCH_LIMIT) -> dict:
    """Trả về dict entityId -> metadata khớp vector query (> SCORE_THRESHOLD)."""
    if not query_vector:
        return {}

    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX_NAME,
                "path": VECTOR_FIELD_PATH,
                "queryVector": query_vector,
                "numCandidates": NUM_CANDIDATES,
                "limit": limit
            }
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
            "s": s_match.get("score"),
            "v": v_match.get("score"),
            "o": o_match.get("score"),
        },
        "matched_by": {
            "s": compact_candidate(s_match, source_node),
            "v": compact_candidate(v_match, edge),
            "o": compact_candidate(o_match, target_node),
        },
        "edge": edge,
        "source_node": source_node,
        "target_node": target_node,
    }

def run_pipeline():
    print("="*70)
    print("🚀 QUERY FLOW PIPELINE — MATCH QUESTION TRIPLETS TO GRAPH TRIPLETS")
    print("="*70)

    # 1. KIỂM TRA FILE ĐẦU VÀO
    if not os.path.exists(TRIPLETS_IN):
        print(f"❌ Không tìm thấy file input (bộ 3 embedding): {TRIPLETS_IN}")
        print("Định dạng mong muốn:")
        print('[{ "s": {"name": "..", "vector": [...]}, "v": {...}, "o": {...} }]')
        return

    with open(TRIPLETS_IN, "r", encoding="utf-8") as f:
        input_triplets = json.load(f)
    print(f"📥 Đã nạp {len(input_triplets)} tuples từ: {TRIPLETS_IN}")

    if not MONGO_URI:
        print("❌ Thiếu biến môi trường MONGO_URI trong .env")
        return

    # Thêm tlsAllowInvalidCertificates=True để bỏ qua lỗi SSL Handshake khắt khe
    client = pymongo.MongoClient(MONGO_URI, tlsCAFile=certifi.where(), tlsAllowInvalidCertificates=True)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    print("\n📌 Nạp graph local để lấy edge data đầy đủ")
    try:
        graph_data, all_nodes, all_edges_by_id = load_graph(GRAPH_FILE)
    except Exception as e:
        print(f"❌ Không nạp được graph: {e}")
        return

    all_edges = graph_data.get("relationships", [])
    print(f"   => Graph: {len(all_nodes)} nodes | {len(all_edges)} edges")

    # --- VECTOR SEARCH TRÊN TỪNG THÀNH PHẦN & LỌC TRIPLETS ---
    print("\n📌 VECTOR SEARCH VÀ LỌC STRICT TRIPLET")

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

        # Tìm candidate IDs từ Mongo (những ID có semantic tương đồng)
        v_candidates = mongo_vector_search(collection, v_obj.get("vector"))
        s_candidates = mongo_vector_search(collection, s_obj.get("vector"))
        o_candidates = mongo_vector_search(collection, o_obj.get("vector"))

        v_ids = candidate_ids(v_candidates)
        s_ids = candidate_ids(s_candidates)
        o_ids = candidate_ids(o_candidates)

        print(f"   [s] '{s_obj.get('name')}' -> {len(s_ids)} candidate IDs")
        print(f"   [v] '{v_obj.get('name')}' -> {len(v_ids)} candidate IDs")
        print(f"   [o] '{o_obj.get('name')}' -> {len(o_ids)} candidate IDs")

        query_matches = []
        graph_edge_ids = v_ids.intersection(all_edges_by_id.keys())
        sorted_graph_edge_ids = sorted(
            graph_edge_ids,
            key=lambda edge_id: v_candidates[edge_id].get("score", 0),
            reverse=True,
        )
        for edge_id in sorted_graph_edge_ids:
            edge = all_edges_by_id.get(edge_id)
            if not edge:
                continue

            src_id = edge.get("source")
            tgt_id = edge.get("target")

            if src_id not in s_candidates:
                continue

            if tgt_id not in o_candidates:
                continue

            source_node = all_nodes.get(src_id, {"id": src_id, "name": ""})
            target_node = all_nodes.get(tgt_id, {"id": tgt_id, "name": ""})
            match_record = build_matched_triplet_record(
                t,
                edge,
                source_node,
                target_node,
                s_candidates[src_id],
                v_candidates[edge_id],
                o_candidates[tgt_id],
            )

            valid_edges.append(edge)
            valid_node_ids.update([src_id, tgt_id])
            section_ids.update(edge.get("listSectionId", []))
            matched_triplets.append(match_record)
            query_matches.append(match_record)

            print(f"   => KHỚP: ({source_node.get('name')})-[{edge.get('name')}]->({target_node.get('name')})")

        if not query_matches:
            print("   => KHÔNG TÌM THẤY TRIPLETS TƯƠNG ƯỚNG TRONG GRAPH")

        matched_triplets_by_query.append({
            "query_index": i,
            "query_triplet": {
                "s": s_obj.get("name", ""),
                "v": v_obj.get("name", ""),
                "o": o_obj.get("name", ""),
            },
            "candidate_counts": {
                "s": len(s_ids),
                "v": len(v_ids),
                "o": len(o_ids),
                "v_edges_in_graph": len(graph_edge_ids),
            },
            "matches": query_matches,
        })

    # Lưu Step 4 Output
    unique_edges = {e["id"]: e for e in valid_edges}.values()
    valid_nodes = [all_nodes[nid] for nid in valid_node_ids if nid in all_nodes]
    sorted_sids = sorted(list(section_ids))

    print(f"\n✅ Graph Filter: {len(unique_edges)} edges | {len(valid_nodes)} nodes | {len(sorted_sids)} section IDs")

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

if __name__ == "__main__":
    run_pipeline()
