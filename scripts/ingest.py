from __future__ import annotations

import argparse
import json
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EU_ADDITIVES_PATH = PROJECT_ROOT / "data" / "raw" / "eu_food_additives.json"
COT_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "synthetic_cot_dataset.jsonl"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ADDITIVES_COLLECTION = "additives_corpus"
COT_COLLECTION = "cot_corpus"
BATCH_SIZE = 64

# Matches e.g. "E322", "E150d", "e415"
ENUM_RE = re.compile(r"\bE\d{3,4}[a-zA-Z]?\b", re.IGNORECASE)


def _make_client(qdrant_url: Optional[str], in_memory: bool) -> QdrantClient:
    if in_memory:
        print("Using in-memory Qdrant (no Docker required — data lost on exit).")
        return QdrantClient(":memory:")
    url = qdrant_url or "http://localhost:6333"
    print(f"Connecting to Qdrant at {url}")
    return QdrantClient(url=url, check_compatibility=False)


def _ensure_collection(client: QdrantClient, name: str, vector_size: int) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=vector_size, distance=models.Distance.COSINE
            ),
        )
        print(f"  Created collection: {name}")
    else:
        print(f"  Collection already exists: {name}")


def _upsert_batch(
    client: QdrantClient,
    encoder: SentenceTransformer,
    collection: str,
    docs: List[Dict[str, Any]],
) -> None:
    for start in tqdm(range(0, len(docs), BATCH_SIZE), desc=f"  upserting {collection}"):
        batch = docs[start : start + BATCH_SIZE]
        texts = [d["text"] for d in batch]
        vectors = encoder.encode(texts, normalize_embeddings=True).tolist()
        points = [
            models.PointStruct(id=d["id"], vector=vec, payload=d["payload"])
            for d, vec in zip(batch, vectors)
        ]
        client.upsert(collection_name=collection, points=points)


def build_additive_docs(path: Path) -> List[Dict[str, Any]]:
    data: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    docs = []
    for code, entry in data.items():
        name = entry.get("name", "")
        functions = entry.get("functions", [])
        aliases = entry.get("aliases", [])
        chunk_text = (
            f"E-Number: {code}\n"
            f"Name: {name}\n"
            f"Functions: {', '.join(functions) if functions else 'Unknown'}\n"
            f"Aliases: {', '.join(aliases) if aliases else 'N/A'}\n"
            "Jurisdiction: EU"
        )
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"additive::{code}"))
        docs.append({
            "id": point_id,
            "text": chunk_text,
            "payload": {
                "e_number": code,
                "name": name,
                "functions": functions,
                "aliases": aliases,
                "jurisdiction": "EU",
                "doc_type": "additive_entity",
                "text": chunk_text,
            },
        })
    return docs


def _extract_product_name(raw: Any) -> str:
    """product_name can be a string or a dict like {"lang": "main", "text": "..."}."""
    if isinstance(raw, dict):
        return str(raw.get("text") or raw.get("lang") or "Unknown")
    if isinstance(raw, str):
        return raw
    return "Unknown"


def _extract_ingredients_text(raw: Any) -> str:
    """ingredients_text can be a plain string or a stringified list of dicts."""
    if isinstance(raw, str):
        # Try to strip the stringified-list wrapper that some records have
        match = re.search(r"'text':\s*'([^']+)'", raw)
        if match:
            return match.group(1)
        return raw
    return str(raw)


def build_cot_docs(path: Path) -> List[Dict[str, Any]]:
    docs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            trace = rec.get("trace") or {}
            nova = rec.get("ground_truth_nova_group")
            ingredients_raw = rec.get("ingredients_text", "")
            product_name = _extract_product_name(rec.get("product_name", ""))
            ingredients_text = _extract_ingredients_text(ingredients_raw)
            reasoning = trace.get("reasoning_summary", "")

            # Build a chunk that is useful as a few-shot example:
            # ingredient list + final reasoning + NOVA answer
            steps = trace.get("ingredient_steps", [])
            step_lines = []
            for step in steps if isinstance(steps, list) else []:
                ing = step.get("ingredient", "")
                analysis = step.get("analysis", "")
                e_num = step.get("e_number") or ""
                fn = step.get("cited_function") or ""
                line_parts = [f"- {ing}: {analysis}"]
                if e_num:
                    line_parts.append(f"({e_num}, {fn})" if fn else f"({e_num})")
                step_lines.append(" ".join(line_parts))

            chunk_text = (
                f"Product: {product_name}\n"
                f"Ingredients: {ingredients_text[:400]}\n"
                f"Analysis:\n" + "\n".join(step_lines[:10]) + "\n"
                f"Conclusion: {reasoning}\n"
                f"NOVA group: {nova}"
            )

            source_key = rec.get("source_key") or rec.get("id") or str(uuid.uuid4())
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"cot::{source_key}"))
            docs.append({
                "id": point_id,
                "text": chunk_text,
                "payload": {
                    "source_key": source_key,
                    "product_name": product_name,
                    "nova_group": nova,
                    "doc_type": "cot_trace",
                    "text": chunk_text,
                },
            })
    return docs


def run(qdrant_url: Optional[str], in_memory: bool) -> QdrantClient:
    client = _make_client(qdrant_url, in_memory)

    print("\nLoading embedding model...")
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    vector_size = int(encoder.get_embedding_dimension())

    # Additives
    print(f"\nLoading EU additives from {EU_ADDITIVES_PATH} ...")
    additive_docs = build_additive_docs(EU_ADDITIVES_PATH)
    print(f"  {len(additive_docs)} additive entries")
    _ensure_collection(client, ADDITIVES_COLLECTION, vector_size)
    _upsert_batch(client, encoder, ADDITIVES_COLLECTION, additive_docs)
    info = client.get_collection(ADDITIVES_COLLECTION)
    print(f"  additives_corpus points: {info.points_count}")

    # CoT traces
    print(f"\nLoading CoT traces from {COT_DATASET_PATH} ...")
    cot_docs = build_cot_docs(COT_DATASET_PATH)
    print(f"  {len(cot_docs)} CoT traces")
    _ensure_collection(client, COT_COLLECTION, vector_size)
    _upsert_batch(client, encoder, COT_COLLECTION, cot_docs)
    info = client.get_collection(COT_COLLECTION)
    print(f"  cot_corpus points: {info.points_count}")

    print("\nIngestion complete.")
    return client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TruthBite Qdrant ingestion (additives + CoT).")
    parser.add_argument("--qdrant-url", default=None, help="Qdrant URL (default: http://localhost:6333)")
    parser.add_argument("--in-memory", action="store_true", help="Use in-memory Qdrant (no Docker needed)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(qdrant_url=args.qdrant_url, in_memory=args.in_memory)
