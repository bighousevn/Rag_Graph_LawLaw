#!/usr/bin/env python3
"""Query legal content from persistent Qdrant VectorDB."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Sequence


class SentenceTransformerEncoder:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def encode(self, input: Sequence[str]) -> List[List[float]]:
        vectors = self.model.encode(list(input), normalize_embeddings=True)
        return vectors.tolist()


def build_qdrant_client(qdrant_url: str, qdrant_api_key: str, qdrant_path: Path):
    from qdrant_client import QdrantClient

    if qdrant_url:
        return QdrantClient(url=qdrant_url, api_key=qdrant_api_key or None, timeout=60.0)
    return QdrantClient(path=str(qdrant_path))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Query legal Qdrant VectorDB")
    p.add_argument("--query", type=str, required=True, help="Noi dung can tra cuu")
    p.add_argument("--qdrant-url", type=str, default="", help="Qdrant cloud URL")
    p.add_argument("--qdrant-api-key", type=str, default="", help="Qdrant API key (neu bo trong se doc env QDRANT_API_KEY)")
    p.add_argument("--qdrant-path", type=Path, default=Path("vectordb/qdrant"), help="Path local Qdrant")
    p.add_argument("--collection", type=str, default="legal_articles")
    p.add_argument("--embedding-model", type=str, default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    p.add_argument("--top-k", type=int, default=5)
    return p


def main() -> int:
    args = build_parser().parse_args()

    qdrant_api_key = args.qdrant_api_key or os.getenv("QDRANT_API_KEY", "")
    if not args.qdrant_url and not args.qdrant_path.exists():
        raise FileNotFoundError(f"Khong tim thay Qdrant local path: {args.qdrant_path}")

    encoder = SentenceTransformerEncoder(args.embedding_model)
    query_vec = encoder.encode([args.query])[0]

    client = build_qdrant_client(
        qdrant_url=args.qdrant_url,
        qdrant_api_key=qdrant_api_key,
        qdrant_path=args.qdrant_path,
    )

    hits = client.search(
        collection_name=args.collection,
        query_vector=query_vec,
        limit=max(1, args.top_k),
        with_payload=True,
    )

    rows = []
    for i, hit in enumerate(hits):
        payload = hit.payload or {}
        rows.append(
            {
                "rank": i + 1,
                "score": hit.score,
                "text": payload.get("text", ""),
                "metadata": {k: v for k, v in payload.items() if k != "text"},
            }
        )

    print(json.dumps({"query": args.query, "results": rows}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
