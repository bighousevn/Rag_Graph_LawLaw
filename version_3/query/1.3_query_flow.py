"""
Query Flow Pipeline (Steps 3 → 4 → 5)
--------------------------------------
Input   : ./version_3/query/3_triplets_with_vectors.json
          (Chứa danh sách các triplet, mỗi S, V, O có sẵn trường "name" và "vector")
Process :
          - Step 3: MongoDB Vector Search cho từng thành phần (s, v, o) để tìm candidate IDs.
          - Step 4: Lọc Strict Triplet từ đồ thị (edge thuộc V, source thuộc S, target thuộc O).
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
GRAPH_FILE       = f"{BASE}/2_entities_per_chunk.json"

# ── MONGODB CONFIG ───────────────────────────────────────────────────
MONGO_URI            = os.getenv("MONGO_URI", "").strip()
DB_NAME              = os.getenv("MONGO_DB_NAME", "vectorDB").strip()
COLLECTION_NAME      = os.getenv("MONGO_COLLECTION_NAME", "vector_entities").strip()
VECTOR_INDEX_NAME    = os.getenv("MONGO_VECTOR_INDEX_NAME", "vector_index").strip()
VECTOR_FIELD_PATH    = os.getenv("MONGO_VECTOR_FIELD_PATH", "vector").strip()
SCORE_THRESHOLD      = 0.85

def mongo_vector_search(db_collection, query_vector, limit=20) -> set:
    """Trả về tập hợp các entityId khớp với vector query (> SCORE_THRESHOLD)."""
    if not query_vector:
        return set()

    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX_NAME,
                "path": VECTOR_FIELD_PATH,
                "queryVector": query_vector,
                "numCandidates": 100,
                "limit": limit
            }
        },
        {
            "$project": {
                "entityId": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        },
        {
            "$match": {"score": {"$gt": SCORE_THRESHOLD}}
        }
    ]

    try:
        matches = list(db_collection.aggregate(pipeline))
        return {m["entityId"] for m in matches}
    except Exception as e:
        print(f"Lỗi truy vấn MongoDB: {e}")
        return set()

def run_pipeline():
    print("="*70)
    print("🚀 QUERY FLOW PIPELINE — STEPS 3 & 4 (FROM VECTOR TRIPLETS)")
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

    # --- STEP 3 & 4: MONGODB SEARCH TRÊN TỪNG THÀNH PHẦN & LỌC TRIPLETS ---
    print("\n📌 STEP 3 & 4 — VECTOR SEARCH VÀ LỌC ĐỒ THỊ GỐC")

    with open(GRAPH_FILE, "r", encoding="utf-8") as f:
        raw_graph = json.load(f)
    graph_data = raw_graph[0] if isinstance(raw_graph, list) and raw_graph else raw_graph

    all_edges = graph_data.get("relationships", [])
    all_nodes = {n["id"]: n for n in graph_data.get("nodes", [])}

    valid_edges = []
    valid_node_ids = set()
    section_ids = set()

    for i, t in enumerate(input_triplets):
        print(f"\n▶ Xử lý Triplet #{i+1}:")
        s_obj = t.get("s", {})
        v_obj = t.get("v", {})
        o_obj = t.get("o", {})

        # Step 3: Tìm candidate IDs từ Mongo (những ID có semantic tương đồng)
        v_candidates = mongo_vector_search(collection, v_obj.get("vector"))
        s_candidates = mongo_vector_search(collection, s_obj.get("vector"))
        o_candidates = mongo_vector_search(collection, o_obj.get("vector"))

        print(f"   [v] '{v_obj.get('name')}' -> {len(v_candidates)} matching edges")
        print(f"   [s] '{s_obj.get('name')}' -> {len(s_candidates)} matching nodes")
        print(f"   [o] '{o_obj.get('name')}' -> {len(o_candidates)} matching nodes")

        # Step 4: Map thẳng vào Grid Edge của Graph
        found = False
        for edge in all_edges:
            edge_id = edge.get("id")
            src_id = edge.get("source")
            tgt_id = edge.get("target")

            # Check Vector Relations (v)
            if edge_id not in v_candidates:
                continue

            # Check Subjects (s)
            if src_id not in s_candidates:
                continue

            # Check Objects (o)
            if tgt_id not in o_candidates:
                continue

            # Thoả đủ 3 chiều S-V-O
            valid_edges.append(edge)
            valid_node_ids.update([src_id, tgt_id])
            section_ids.update(edge.get("listSectionId", []))
            found = True

            src_name = all_nodes.get(src_id, {}).get("name", "")
            tgt_name = all_nodes.get(tgt_id, {}).get("name", "")
            edge_name = edge.get("name", "")
            print(f"   => KHỚP: ({src_name})-[{edge_name}]->({tgt_name})")

        if not found:
            print("   => KHÔNG TÌM THẤY TRIPLETS TƯƠNG ƯỚNG TRONG GRAPH")

    # Lưu Step 4 Output
    unique_edges = {e["id"]: e for e in valid_edges}.values()
    valid_nodes = [all_nodes[nid] for nid in valid_node_ids if nid in all_nodes]
    sorted_sids = sorted(list(section_ids))

    print(f"\n✅ Graph Filter: {len(unique_edges)} edges | {len(valid_nodes)} nodes | {len(sorted_sids)} section IDs")

    filtered_data = [{
        "nodes": valid_nodes,
        "relationships": list(unique_edges),
        "relevant_section_ids": sorted_sids,
    }]
    os.makedirs(os.path.dirname(FILTERED_OUT), exist_ok=True)
    with open(FILTERED_OUT, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    run_pipeline()
