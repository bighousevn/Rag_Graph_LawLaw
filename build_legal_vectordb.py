#!/usr/bin/env python3
"""Build a persistent Qdrant VectorDB from structured legal content."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence


def normalize_text(text: str) -> str:
    return " ".join(str(text).split()).strip()


class SentenceTransformerEncoder:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def encode(self, input: Sequence[str]) -> List[List[float]]:
        vectors = self.model.encode(list(input), normalize_embeddings=True)
        return vectors.tolist()


def load_structured_records(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RuntimeError("Input JSON phai la mot danh sach object.")
    return [x for x in data if isinstance(x, dict)]


def flatten_documents(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []

    for rec in records:
        base_id = str(rec.get("id", "")) or "unknown"
        metadata = rec.get("metadata") if isinstance(rec.get("metadata"), dict) else {}

        chapter = str(metadata.get("chuong") or "")
        article = str(metadata.get("dieu") or "")
        clause = str(metadata.get("khoan") or "")
        point = str(metadata.get("diem") or "")

        sentences_raw = rec.get("llm_processed_text", [])
        if isinstance(sentences_raw, list):
            sentences = [normalize_text(x) for x in sentences_raw if normalize_text(x)]
        else:
            one = normalize_text(str(sentences_raw))
            sentences = [one] if one else []

        original_text = normalize_text(str(rec.get("original_text") or ""))

        for idx, sentence in enumerate(sentences, start=1):
            doc_id = f"{base_id}_{idx:02d}"
            docs.append(
                {
                    "id": doc_id,
                    "text": sentence,
                    "metadata": {
                        "record_id": base_id,
                        "atomic_index": idx,
                        "chuong": chapter,
                        "dieu": article,
                        "khoan": clause,
                        "diem": point,
                        "original_text": original_text,
                    },
                }
            )

    return docs


def build_qdrant_client(qdrant_url: str, qdrant_api_key: str, qdrant_path: Path):
    from qdrant_client import QdrantClient

    if qdrant_url:
        return QdrantClient(url=qdrant_url, api_key=qdrant_api_key or None, timeout=60.0)

    qdrant_path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(qdrant_path))


def upsert_collection(
    docs: Sequence[Dict[str, Any]],
    qdrant_url: str,
    qdrant_api_key: str,
    qdrant_path: Path,
    collection_name: str,
    model_name: str,
    batch_size: int,
    reset: bool,
) -> None:
    from qdrant_client import models

    client = build_qdrant_client(qdrant_url=qdrant_url, qdrant_api_key=qdrant_api_key, qdrant_path=qdrant_path)
    encoder = SentenceTransformerEncoder(model_name)

    sample_vec = encoder.encode([docs[0]["text"]])[0]
    vector_size = len(sample_vec)

    exists = client.collection_exists(collection_name)
    if reset and exists:
        client.delete_collection(collection_name)
        exists = False

    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )

    total = len(docs)
    for start in range(0, total, batch_size):
        chunk = docs[start : start + batch_size]
        texts = [d["text"] for d in chunk]
        vectors = encoder.encode(texts)

        points: List[models.PointStruct] = []
        for i, d in enumerate(chunk):
            payload = {"text": d["text"], **d["metadata"]}
            points.append(models.PointStruct(id=d["id"], vector=vectors[i], payload=payload))

        client.upsert(collection_name=collection_name, points=points, wait=True)

    print(f"Da ghi {total} documents vao collection '{collection_name}'.")
    if qdrant_url:
        print(f"Qdrant cloud URL: {qdrant_url}")
    else:
        print(f"Qdrant local path: {qdrant_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build legal Qdrant VectorDB from structured JSON")
    p.add_argument("--input-json", type=Path, default=Path("2_llm_pre_structured.json"))
    p.add_argument("--qdrant-url", type=str, default="", help="Qdrant cloud URL, vd: https://xxx.aws.cloud.qdrant.io")
    p.add_argument("--qdrant-api-key", type=str, default="", help="Qdrant API key (neu bo trong se doc env QDRANT_API_KEY)")
    p.add_argument("--qdrant-path", type=Path, default=Path("vectordb/qdrant"), help="Path local Qdrant (duoc dung khi khong co --qdrant-url)")
    p.add_argument("--collection", type=str, default="legal_articles")
    p.add_argument("--embedding-model", type=str, default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--reset", action="store_true", help="Xoa collection cu truoc khi ghi moi")
    return p


def main() -> int:
    args = build_parser().parse_args()

    if not args.input_json.exists():
        raise FileNotFoundError(f"Khong tim thay input JSON: {args.input_json}")

    records = load_structured_records(args.input_json)
    docs = flatten_documents(records)
    if not docs:
        raise RuntimeError("Khong co document nao de luu vao VectorDB.")

    qdrant_api_key = args.qdrant_api_key or os.getenv("QDRANT_API_KEY", "")

    upsert_collection(
        docs=docs,
        qdrant_url=args.qdrant_url,
        qdrant_api_key=qdrant_api_key,
        qdrant_path=args.qdrant_path,
        collection_name=args.collection,
        model_name=args.embedding_model,
        batch_size=args.batch_size,
        reset=args.reset,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
