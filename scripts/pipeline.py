from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchValue
from sentence_transformers import CrossEncoder, SentenceTransformer

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
ADDITIVES_COLLECTION = "additives_corpus"
COT_COLLECTION = "cot_corpus"

ENUM_RE = re.compile(r"\bE\d{3,4}[a-zA-Z]?\b", re.IGNORECASE)

@dataclass
class ContextResult:
    source_id: str
    score: float
    doc_type: str          # "additive_entity" | "cot_trace"
    text: str
    payload: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.payload is None:
            self.payload = {}


def _extract_enumbers(ingredients_text: str) -> List[str]:
    """Return unique E-numbers found in the ingredient list, upper-cased."""
    return list({m.upper() for m in ENUM_RE.findall(ingredients_text or "")})


def _dense_search(
    client: QdrantClient,
    encoder: SentenceTransformer,
    collection: str,
    query: str,
    top_k: int,
    metadata_filter: Optional[Dict[str, str]] = None,
) -> List[ContextResult]:
    vec = encoder.encode(query, normalize_embeddings=True).tolist()
    q_filter = None
    if metadata_filter:
        q_filter = Filter(
            must=[
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in metadata_filter.items()
            ]
        )
    results = client.query_points(
        collection_name=collection,
        query=vec,
        query_filter=q_filter,
        limit=top_k,
        with_payload=True,
    ).points
    return [
        ContextResult(
            source_id=str(r.id),
            score=float(r.score),
            doc_type=(r.payload or {}).get("doc_type", ""),
            text=(r.payload or {}).get("text", ""),
            payload=r.payload or {},
        )
        for r in results
    ]


def _bm25_score(query_tokens: List[str], doc_text: str, avg_len: float, n_docs: int) -> float:
    """Simple in-process BM25 score for a single document."""
    import math
    from collections import Counter
    k1, b = 1.5, 0.75
    tokens = [t.lower() for t in re.findall(r"[a-zA-Z0-9]+", doc_text)]
    counts = Counter(tokens)
    doc_len = max(len(tokens), 1)
    score = 0.0
    for term in query_tokens:
        tf = counts.get(term, 0)
        if tf == 0:
            continue
        df = 1  # simplified: treat every term as appearing in 1 doc
        idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
        num = tf * (k1 + 1.0)
        den = tf + k1 * (1.0 - b + b * (doc_len / (avg_len + 1e-9)))
        score += idf * (num / (den + 1e-9))
    return score


def _hybrid_search(
    client: QdrantClient,
    encoder: SentenceTransformer,
    collection: str,
    query: str,
    top_k: int,
    metadata_filter: Optional[Dict[str, str]] = None,
    alpha: float = 0.7,
) -> List[ContextResult]:
    """Dense + BM25 hybrid, re-scored and merged."""
    pool_size = min(top_k * 6, 60)
    candidates = _dense_search(client, encoder, collection, query, pool_size, metadata_filter)
    if not candidates:
        return []

    import numpy as np
    query_tokens = [t.lower() for t in re.findall(r"[a-zA-Z0-9]+", query)]
    avg_len = sum(len(re.findall(r"[a-zA-Z0-9]+", c.text)) for c in candidates) / len(candidates)
    n_docs = len(candidates)

    dense_scores = np.array([c.score for c in candidates], dtype=np.float32)
    d_min, d_max = dense_scores.min(), dense_scores.max()
    dense_norm = (dense_scores - d_min) / (d_max - d_min + 1e-9)

    bm25_raw = np.array(
        [_bm25_score(query_tokens, c.text, avg_len, n_docs) for c in candidates],
        dtype=np.float32,
    )
    b_min, b_max = bm25_raw.min(), bm25_raw.max()
    bm25_norm = (bm25_raw - b_min) / (b_max - b_min + 1e-9)

    hybrid = alpha * dense_norm + (1.0 - alpha) * bm25_norm
    ranked = sorted(zip(candidates, hybrid.tolist()), key=lambda x: x[1], reverse=True)
    return [r for r, _ in ranked[:top_k]]


def _retrieve_additives(
    client: QdrantClient,
    encoder: SentenceTransformer,
    ingredients_text: str,
    enumbers: List[str],
    country: Optional[str],
    strategy: int,
    top_k: int,
) -> List[ContextResult]:
    """
    Search additives_corpus.
    Run one query per E-number found + one semantic query on the full text.
    Deduplicate by id, keep highest score.
    """
    metadata_filter = {"jurisdiction": "EU"} if country else None
    seen: Dict[str, ContextResult] = {}

    queries = list(enumbers) + [ingredients_text]
    for q in queries:
        if strategy == 2:
            results = _hybrid_search(client, encoder, ADDITIVES_COLLECTION, q, top_k, metadata_filter)
        else:
            results = _dense_search(client, encoder, ADDITIVES_COLLECTION, q, top_k, metadata_filter)
        for r in results:
            if r.source_id not in seen or r.score > seen[r.source_id].score:
                seen[r.source_id] = r

    combined = sorted(seen.values(), key=lambda x: x.score, reverse=True)
    return combined[:top_k * 2]


def _retrieve_cot(
    client: QdrantClient,
    encoder: SentenceTransformer,
    ingredients_text: str,
    top_k: int,
) -> List[ContextResult]:
    """Search cot_corpus for similar past reasoning examples."""
    return _dense_search(client, encoder, COT_COLLECTION, ingredients_text, top_k)


def _rerank(
    reranker: CrossEncoder,
    query: str,
    candidates: List[ContextResult],
    top_k: int,
) -> List[ContextResult]:
    """Re-score candidates with a cross-encoder and return top_k."""
    if not candidates:
        return []
    pairs = [(query, c.payload.get("text", "")) for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:top_k]]


def _format_additive_context(
    additive_results: List[ContextResult],
    cot_results: List[ContextResult],
) -> str:
    lines: List[str] = []

    if additive_results:
        lines.append("=== EU Additive Definitions ===")
        for r in additive_results:
            p = r.payload
            code = p.get("e_number", "")
            name = p.get("name", "")
            functions = p.get("functions", [])
            aliases = p.get("aliases", [])
            fn_str = ", ".join(functions) if functions else "unknown"
            alias_str = ", ".join(aliases[:3]) if aliases else ""
            line = f"{code}: name={name}, functions=[{fn_str}]"
            if alias_str:
                line += f", aliases=[{alias_str}]"
            lines.append(line)

    if cot_results:
        lines.append("\n=== Similar Product Reasoning Examples ===")
        for r in cot_results:
            p = r.payload
            product = p.get("product_name", "Unknown")
            nova = p.get("nova_group", "?")
            lines.append(f"[NOVA {nova}] {product}")
            # Include the first 3 lines of the reasoning chunk as context
            chunk_lines = p.get("text", "").splitlines()
            for cl in chunk_lines[2:6]:  # skip "Product:" and "Ingredients:" header
                if cl.strip():
                    lines.append(f"  {cl.strip()}")

    return "\n".join(lines) if lines else "No retrieved additive context available."


def validate_citations(
    ingredient_steps: List[Dict[str, Any]],
    additives_client: QdrantClient,
) -> Tuple[bool, List[str]]:
    """
    Check each cited E-number in the model's ingredient_steps against
    what is actually stored in additives_corpus.

    Returns (all_valid, list_of_error_strings).
    """
    errors: List[str] = []

    for idx, step in enumerate(ingredient_steps or []):
        if not isinstance(step, dict):
            continue
        e_number = str(step.get("e_number") or "").strip().upper()
        cited_fn = str(step.get("cited_function") or "").strip().lower()
        if not e_number:
            continue

        # Search for this exact E-number in additives_corpus
        results = additives_client.scroll(
            collection_name=ADDITIVES_COLLECTION,
            scroll_filter={
                "must": [{"key": "e_number", "match": {"value": e_number}}]
            },
            limit=1,
            with_payload=True,
        )
        records = results[0]
        if not records:
            errors.append(f"Step {idx}: {e_number} not found in additives_corpus")
            continue

        known_fns = [
            f.lower().replace("en:", "")
            for f in (records[0].payload or {}).get("functions", [])
        ]
        if cited_fn and known_fns:
            if not any(cited_fn in kf or kf in cited_fn for kf in known_fns):
                errors.append(
                    f"Step {idx}: {e_number} cited as '{cited_fn}' "
                    f"but known functions are {known_fns}"
                )

    return len(errors) == 0, errors


def retrieve_context(
    ingredients_text: str,
    country: Optional[str] = None,
    strategy: int = 2,
    top_k: int = 5,
    qdrant_url: Optional[str] = None,
    qdrant_client: Optional[QdrantClient] = None,
) -> Tuple[str, List[str]]:
    """
    Given an ingredient list, retrieve relevant additive definitions and
    CoT reasoning examples from Qdrant, then return:
      - additive_context : formatted string ready to inject into the prompt
      - sources          : list of source identifiers for the response

    Parameters
    ----------
    ingredients_text : raw ingredient text from the product
    country          : product country (used for jurisdiction filter)
    strategy         : 1 = dense-only, 2 = hybrid dense+BM25  (default 2)
    top_k            : final number of results per collection
    qdrant_url       : override Qdrant URL (default: QDRANT_URL env or localhost:6333)
    qdrant_client    : pass a pre-built client (useful for in-memory testing)
    """
    if qdrant_client is None:
        url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_client = QdrantClient(url=url, check_compatibility=False)

    encoder = SentenceTransformer(EMBEDDING_MODEL)
    reranker = CrossEncoder(RERANKER_MODEL)

    enumbers = _extract_enumbers(ingredients_text)

    # Retrieve candidates via hybrid/dense search
    additive_candidates = _retrieve_additives(
        qdrant_client, encoder, ingredients_text, enumbers, country, strategy, top_k=top_k * 4
    )
    cot_candidates = _retrieve_cot(qdrant_client, encoder, ingredients_text, top_k=top_k * 2)

    # Rerank additives (query = ingredient text)
    additive_reranked = _rerank(reranker, ingredients_text, additive_candidates, top_k=top_k)

    # Guaranteed pass-through: directly look up every E-number that appears
    # literally in the ingredient text. Vector search can miss short codes like
    # "E338" because their embeddings score low; a payload filter lookup is exact.
    guaranteed_ids = {r.source_id for r in additive_reranked}
    for enumber in enumbers:
        if any(r.payload.get("e_number", "").upper() == enumber for r in additive_reranked):
            continue  # already in final list
        # Direct payload lookup — bypasses vector scoring entirely
        records, _ = qdrant_client.scroll(
            collection_name=ADDITIVES_COLLECTION,
            scroll_filter=Filter(
                must=[FieldCondition(key="e_number", match=MatchValue(value=enumber))]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        for rec in records:
            rid = str(rec.id)
            if rid not in guaranteed_ids:
                additive_reranked.append(ContextResult(
                    source_id=rid,
                    score=1.0,  # exact match, highest priority
                    doc_type="additive_entity",
                    text=(rec.payload or {}).get("text", ""),
                    payload=rec.payload or {},
                ))
                guaranteed_ids.add(rid)
    additive_final = additive_reranked

    # Rerank CoT examples — keep 3 (richer few-shot context than 2)
    cot_final = _rerank(reranker, ingredients_text, cot_candidates, top_k=min(3, top_k))

    # Format context string
    context = _format_additive_context(additive_final, cot_final)

    # Build sources list
    sources: List[str] = []
    for r in additive_final:
        e = r.payload.get("e_number")
        if e:
            sources.append(f"additives_corpus:{e}")
    for r in cot_final:
        sk = r.payload.get("source_key") or r.payload.get("product_name", "")
        if sk:
            sources.append(f"cot_corpus:{sk}")

    return context, sources

# Standalone test
if __name__ == "__main__":
    ingredients = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Carbonated Water, Sugar, Colour (E150d), Phosphoric Acid (E338), Natural Flavourings including Caffeine"
    )

    from ingest import run as run_ingest
    print("Running in-memory ingestion for standalone test...")
    client = run_ingest(qdrant_url=None, in_memory=True)

    print(f"\nQuery: {ingredients}\n")
    context, sources = retrieve_context(
        ingredients_text=ingredients,
        country="United Kingdom",
        strategy=2,
        top_k=5,
        qdrant_client=client,
    )
    print("=== Retrieved Context ===")
    print(context.encode("ascii", errors="replace").decode("ascii"))
    print("\n=== Sources ===")
    for s in sources:
        print(" -", s)
