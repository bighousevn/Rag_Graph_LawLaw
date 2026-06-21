"""
Phase 6b — Embedding node/relation -> MongoDB (KHÔNG tự chạy)
============================================================
Input: data/master_graph_optimized.json
.env : MONGO_URI, MONGO_DB_NAME, MONGO_COLLECTION_NAME
Model: AITeamVN/Vietnamese_Embedding_v2 (giống pipeline cũ của bạn)

Vector hóa tên Concept (node) và tên Quan hệ (relationship) để bước truy vấn
so khớp ngữ nghĩa (Vector Anchor Matching). Lưu kèm category để lọc theo nhóm.
"""

import os
import json
import torch
import pymongo
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()
BASE = os.path.dirname(os.path.abspath(__file__))
GRAPH_IN = os.path.join(BASE, "data", "master_graph_optimized.json")


def build_documents(graph):
    docs = []
    for n in graph["nodes"]:
        docs.append({
            "entityId": n["id"],
            "entityName": n["name"],
            "synonym": n["name"],
            "category": n.get("label"),
            "type": "node",
        })
    for e in graph["relationships"]:
        docs.append({
            "entityId": e["id"],
            "entityName": e["name"],
            "synonym": e["name"],
            "relId": e.get("relId"),
            "type": "relationship",
        })
    return docs


def main():
    with open(GRAPH_IN, "r", encoding="utf-8") as f:
        graph = json.load(f)

    docs = build_documents(graph)
    if not docs:
        print("⚠️ Không có node/relationship để embedding.")
        return
    print(f"Chuẩn bị vector hóa {len(docs)} bản ghi "
          f"({len(graph['nodes'])} node + {len(graph['relationships'])} relationship)...")

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Thiết bị: {device}")
    model = SentenceTransformer("AITeamVN/Vietnamese_Embedding_v2", device=device)

    vectors = model.encode([d["entityName"] for d in docs], show_progress_bar=True)
    for d, v in zip(docs, vectors):
        d["vector"] = v.tolist()

    uri = os.getenv("MONGO_URI", "").strip()
    db_name = os.getenv("MONGO_DB_NAME", "vectorDB").strip()
    coll_name = os.getenv("MONGO_COLLECTION_NAME", "vector_entities").strip()
    if not uri:
        raise ValueError("Thiếu MONGO_URI trong .env")

    client = pymongo.MongoClient(uri)
    client.admin.command("ping")
    coll = client[db_name][coll_name]
    print(f"Xóa dữ liệu cũ trong '{coll_name}'...")
    coll.delete_many({})
    coll.insert_many(docs)
    print(f"✅ Đã nạp {len(docs)} vector (dim={len(docs[0]['vector'])}) vào MongoDB.")


if __name__ == "__main__":
    main()
