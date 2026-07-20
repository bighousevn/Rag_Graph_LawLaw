import os
import json
import torch
from typing import List, Dict
from dotenv import load_dotenv
import pymongo
from sentence_transformers import SentenceTransformer

load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPH_FILE = os.path.join(os.path.dirname(BASE_DIR), "Building_KG", "ontology", "output", "kg_triplets.json")

# Lấy cấu hình DB từ .env
MONGO_URI = os.getenv("MONGO_URI", "").strip()
DB_NAME = os.getenv("MONGO_DB_NAME", "vectorDB").strip()
COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "vector_entities").strip()
VECTOR_INDEX_NAME = os.getenv("MONGO_VECTOR_INDEX_NAME", "vector_index").strip()
VECTOR_FIELD_PATH = os.getenv("MONGO_VECTOR_FIELD_PATH", "vector").strip()
SCORE_THRESHOLD = float(os.getenv("MONGO_VECTOR_SCORE_THRESHOLD", "0.80"))
SEARCH_LIMIT = int(os.getenv("MONGO_VECTOR_SEARCH_LIMIT", "20"))
NUM_CANDIDATES = int(os.getenv("MONGO_VECTOR_NUM_CANDIDATES", "100"))

_embedding_model = None
def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print("  - Tải embedding model (AITeamVN/Vietnamese_Embedding_v2)...")
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        _embedding_model = SentenceTransformer("AITeamVN/Vietnamese_Embedding_v2", device=device)
    return _embedding_model

def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()

def mongo_vector_search(db_collection, query_vector, entity_type=None, limit=SEARCH_LIMIT) -> dict:
    if not query_vector:
        return {}

    vector_search_stage = {
        "index": VECTOR_INDEX_NAME,
        "path": VECTOR_FIELD_PATH,
        "queryVector": query_vector,
        "numCandidates": NUM_CANDIDATES,
        "limit": limit
    }
    if entity_type:
        vector_search_stage["filter"] = {"type": entity_type}

    pipeline = [
        {"$vectorSearch": vector_search_stage},
        {"$project": {
            "_id": 0, "entityId": 1, "entityName": 1, "score": {"$meta": "vectorSearchScore"}
        }},
        {"$match": {"score": {"$gt": SCORE_THRESHOLD}}}
    ]
    try:
        matches = list(db_collection.aggregate(pipeline))
        candidates = {}
        for match in matches:
            entity_id = match.get("entityId")
            if entity_id:
                current = candidates.get(entity_id)
                if current is None or match.get("score", 0) > current.get("score", 0):
                    candidates[entity_id] = match
        return candidates
    except Exception as e:
        print(f"  - Lỗi MongoDB search: {e}")
        return {}

def search_graph_from_triplets(triplets: List[Dict]) -> List[Dict]:
    """
    Nhận danh sách triplet từ Bước 2.
    Query vào MongoDB để tìm node/relation ID, sau đó đối chiếu với kg_triplets.json
    Trả về danh sách node & relation khớp nhất.
    """
    if not MONGO_URI:
        print("❌ MONGO_URI chưa được cấu hình. Bỏ qua bước vector search.")
        return []

    try:
        client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        # Test connection
        client.admin.command('ping')
    except Exception as e:
        print(f"❌ Lỗi kết nối MongoDB: {e}")
        return []

    # Load KG
    if not os.path.exists(GRAPH_FILE):
        print(f"❌ Không tìm thấy file đồ thị: {GRAPH_FILE}")
        return []
    
    with open(GRAPH_FILE, "r", encoding="utf-8") as f:
        graph_data = json.load(f)

    nodes_by_id = {n["id"]: n for n in graph_data.get("nodes", [])}
    edges = graph_data.get("edges", [])

    results = []

    for i, t in enumerate(triplets):
        s_name, v_name, o_name = t.get("s", ""), t.get("v", ""), t.get("o", "")
        print(f"\n  [Query {i+1}] ({s_name}) - [{v_name}] -> ({o_name})")
        
        # 1. Embed query (bỏ qua nếu là ẩn số *)
        s_vec = embed_texts([s_name])[0] if s_name and s_name != "*" else None
        v_vec = embed_texts([v_name])[0] if v_name and v_name != "*" else None
        o_vec = embed_texts([o_name])[0] if o_name and o_name != "*" else None

        # 2. Vector search trên DB lấy candidates
        s_cands = mongo_vector_search(collection, s_vec, entity_type="node") if s_vec else {}
        v_cands = mongo_vector_search(collection, v_vec, entity_type="relationship") if v_vec else {}
        o_cands = mongo_vector_search(collection, o_vec, entity_type="node") if o_vec else {}
        
        # Lấy ID (nếu là * thì lấy tất cả)
        s_ids = set(s_cands.keys()) if s_name != "*" else set(nodes_by_id.keys())
        v_ids = set(v_cands.keys()) if v_name != "*" else set(e["relation_id"] for e in edges) | set(e["id"] for e in edges)
        o_ids = set(o_cands.keys()) if o_name != "*" else set(nodes_by_id.keys())

        print(f"    -> Tìm thấy: {len(s_cands)} Subject, {len(v_cands)} Relation, {len(o_cands)} Object tương đồng trong DB.")

        matched_nodes = []
        matched_relations = []

        # 3. Lấy thông tin Node từ đồ thị
        # Các node map với S
        for s_id, s_data in s_cands.items():
            if s_id in nodes_by_id:
                node_info = dict(nodes_by_id[s_id])
                node_info["role"] = "source"
                node_info["score"] = s_data.get("score")
                matched_nodes.append(node_info)

        # Các node map với O
        for o_id, o_data in o_cands.items():
            if o_id in nodes_by_id:
                node_info = dict(nodes_by_id[o_id])
                node_info["role"] = "target"
                node_info["score"] = o_data.get("score")
                matched_nodes.append(node_info)

        # 4. Lấy thông tin Relation từ đồ thị
        # Trong Vector DB, v_cands có thể là relation_id hoặc edge_id tùy cách bạn seed DB.
        # Ở đây ta sẽ quét các edge trong KG, nếu edge đó có relation_id hoặc id nằm trong v_cands thì lấy.
        for edge in edges:
            rel_id = edge.get("relation_id")
            edge_id = edge.get("id")
            
            match_data = None
            if rel_id in v_cands:
                match_data = v_cands[rel_id]
            elif edge_id in v_cands:
                match_data = v_cands[edge_id]

            if match_data:
                matched_relations.append({
                    "id": edge_id,
                    "relation_id": rel_id,
                    "name": edge.get("relation"),
                    "source_id": edge.get("source"),
                    "target_id": edge.get("target"),
                    "listSectionId": edge.get("listSectionId", []),
                    "score": match_data.get("score")
                })

        results.append({
            "query_triplet": t,
            "matched_nodes": matched_nodes,
            "matched_relations": matched_relations
        })

    return results

if __name__ == "__main__":
    # Test script trực tiếp
    test_triplets = [
        {"s": "Người mua xe", "v": "Giao", "o": "Xe"},
        {"s": "Người", "v": "Đủ", "o": "Điều kiện"}
    ]
    
    print("--- Đang xử lý Step 3: Vector Search ---")
    out_results = search_graph_from_triplets(test_triplets)
    
    print("\n[KẾT QUẢ STEP 3]")
    print(json.dumps(out_results, ensure_ascii=False, indent=4))
