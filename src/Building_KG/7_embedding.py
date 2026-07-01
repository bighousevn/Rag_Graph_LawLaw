"""
Bước 7: Embed nodes và relations từ KG JSON vào MongoDB.
Cải tiến so với phiên bản cũ: embed cả KEYPHRASES từ ontology (không chỉ tên concept/relation)
→ vector search tìm được "xe máy" → khớp concept "Xe mô tô" qua synonym.
Input:  triplets/triplets_nghi_dinh_168_2024_1.json + ontology_168.json
Output: MongoDB collection (xóa cũ, insert mới)
"""

import json
import os
import torch
import pymongo
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
KG_FILE    = os.path.join(BASE_DIR, "triplets/triplets_nghi_dinh_168_2024_1.json")
ONTO_FILE  = os.path.join(BASE_DIR, "ontology_168.json")

MONGO_URI        = os.getenv("MONGO_URI", "").strip()
DB_NAME          = os.getenv("MONGO_DB_NAME", "vectorDB").strip()
COLLECTION_NAME  = os.getenv("MONGO_COLLECTION_NAME", "vector_entities").strip()
EMBEDDING_MODEL  = "AITeamVN/Vietnamese_Embedding_v2"


def build_documents(kg: dict, ontology: dict) -> list[dict]:
    """
    Mỗi document MongoDB: {entityId, entityName, synonym, type, vector}
    - entityName = tên chuẩn của concept/relation trong ontology
    - synonym    = keyphrase cụ thể (hoặc bằng entityName nếu không có keyphrase)
    → Nhiều documents cùng entityId nhưng synonym khác nhau, vector khác nhau.
    """
    # Build lookup: name → keyphrases
    concept_kps  = {c["name"]: c["keyphrases"] for c in ontology["concepts"]}
    relation_kps = {r["name"]: r["keyphrases"] for r in ontology["relations"]}

    docs: list[dict] = []
    seen: set[tuple] = set()  # (entityId, synonym)

    def add(entity_id: str, entity_name: str, synonyms: list[str], doc_type: str):
        all_synonyms = [entity_name] + [s for s in synonyms if s != entity_name]
        for syn in all_synonyms:
            key = (entity_id, syn)
            if key not in seen:
                seen.add(key)
                docs.append({
                    "entityId":   entity_id,
                    "entityName": entity_name,
                    "synonym":    syn,
                    "type":       doc_type,
                })

    nodes = kg.get("nodes", [])
    edges = kg.get("relationships", [])

    for node in nodes:
        kps = concept_kps.get(node["name"], [])
        add(node["id"], node["name"], kps, "node")

    for edge in edges:
        kps = relation_kps.get(edge["name"], [])
        add(edge["id"], edge["name"], kps, "relationship")

    return docs


def main():
    print("Đang kiểm tra thiết bị...")
    device = (
        "mps"  if torch.backends.mps.is_available()  else
        "cuda" if torch.cuda.is_available()           else
        "cpu"
    )
    print(f"Thiết bị: {device}")

    print(f"Đang tải model {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)

    with open(KG_FILE, encoding="utf-8") as f:
        kg_data = json.load(f)
    kg = kg_data[0] if isinstance(kg_data, list) else kg_data

    with open(ONTO_FILE, encoding="utf-8") as f:
        ontology = json.load(f)

    print(f"KG: {len(kg['nodes'])} nodes, {len(kg['relationships'])} edges")

    docs = build_documents(kg, ontology)
    print(f"Tổng documents cần embed (node + relation + synonyms): {len(docs)}")

    texts = [d["synonym"] for d in docs]
    print("Đang sinh embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=128)

    for i, doc in enumerate(docs):
        doc["vector"] = embeddings[i].tolist()

    if not MONGO_URI:
        raise ValueError("Thiếu MONGO_URI trong .env")

    print(f"Kết nối MongoDB: {MONGO_URI}")
    mongo = pymongo.MongoClient(MONGO_URI)
    mongo.admin.command("ping")
    print("Kết nối thành công.")

    col = mongo[DB_NAME][COLLECTION_NAME]
    print(f"Xóa dữ liệu cũ trong '{COLLECTION_NAME}'...")
    col.delete_many({})

    print(f"Insert {len(docs)} documents...")
    col.insert_many(docs)

    dim = len(docs[0]["vector"])
    print(f"Hoàn thành. Vector dimension: {dim}")
    print(f"Collection '{COLLECTION_NAME}' tổng: {col.count_documents({})} documents")


if __name__ == "__main__":
    main()
