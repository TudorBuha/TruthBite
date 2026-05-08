from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchValue
from sentence_transformers import SentenceTransformer


TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


@dataclass
class RetrievedItem:
    id: str
    score: float
    payload: Dict[str, Any]


class Retriever:
    """
    Strategy 1:
      - Dense-only semantic search with all-MiniLM-L6-v2

    Strategy 2:
      - Hybrid score = dense score + bm25 score
      - Optional metadata filtering (e.g. by Country)
    """

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "ingredients_corpus",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.collection_name = collection_name
        self.client = QdrantClient(url=qdrant_url)
        self.encoder = SentenceTransformer(embedding_model)
        self._bm25_index_ready = False
        self._doc_tokens: Dict[str, Counter] = {}
        self._doc_len: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}
        self._avg_doc_len = 0.0
        self._doc_payload: Dict[str, Dict[str, Any]] = {}

    def _embed(self, text: str) -> List[float]:
        vec = self.encoder.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def _build_bm25_index(self, metadata_filter: Optional[Dict[str, Any]] = None) -> None:
        points: List[Tuple[str, Dict[str, Any], str]] = []
        next_offset = None
        while True:
            record_batch, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=next_offset,
                with_payload=True,
            )
            if not record_batch:
                break
            for rec in record_batch:
                payload = rec.payload or {}
                if metadata_filter and not self._payload_matches(payload, metadata_filter):
                    continue
                text = payload.get("text", "")
                points.append((str(rec.id), payload, text))
            if next_offset is None:
                break

        self._doc_tokens = {}
        self._doc_len = {}
        self._doc_payload = {}
        term_doc_freq = defaultdict(int)

        for pid, payload, text in points:
            tokens = _tokenize(text)
            counts = Counter(tokens)
            self._doc_tokens[pid] = counts
            self._doc_len[pid] = max(len(tokens), 1)
            self._doc_payload[pid] = payload
            for tok in counts:
                term_doc_freq[tok] += 1

        n_docs = max(len(points), 1)
        self._avg_doc_len = sum(self._doc_len.values()) / n_docs if points else 1.0
        self._idf = {
            tok: math.log(1 + (n_docs - df + 0.5) / (df + 0.5)) for tok, df in term_doc_freq.items()
        }
        self._bm25_index_ready = True

    @staticmethod
    def _payload_matches(payload: Dict[str, Any], metadata_filter: Dict[str, Any]) -> bool:
        for key, expected_value in metadata_filter.items():
            if payload.get(key) != expected_value:
                return False
        return True

    def strategy1_dense_search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedItem]:
        qvec = self._embed(query)
        q_filter = None
        if metadata_filter:
            q_filter = Filter(
                must=[FieldCondition(key=k, match=MatchValue(value=v)) for k, v in metadata_filter.items()]
            )
        result = self.client.search(
            collection_name=self.collection_name,
            query_vector=qvec,
            query_filter=q_filter,
            limit=top_k,
            with_payload=True,
        )
        return [RetrievedItem(id=str(r.id), score=float(r.score), payload=r.payload or {}) for r in result]

    def strategy2_hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
        alpha: float = 0.7,
        dense_pool_size: int = 30,
    ) -> List[RetrievedItem]:
        dense_candidates = self.strategy1_dense_search(
            query=query,
            top_k=dense_pool_size,
            metadata_filter=metadata_filter,
        )

        if not self._bm25_index_ready:
            self._build_bm25_index(metadata_filter=metadata_filter)

        query_tokens = _tokenize(query)
        k1, b = 1.5, 0.75

        dense_scores = np.array([item.score for item in dense_candidates], dtype=np.float32)
        if dense_scores.size:
            d_min, d_max = float(dense_scores.min()), float(dense_scores.max())
            dense_norm = (dense_scores - d_min) / (d_max - d_min + 1e-9)
        else:
            dense_norm = np.array([], dtype=np.float32)

        bm25_scores = []
        for item in dense_candidates:
            counts = self._doc_tokens.get(item.id, Counter(_tokenize(item.payload.get("text", ""))))
            doc_len = self._doc_len.get(item.id, max(sum(counts.values()), 1))
            score = 0.0
            for term in query_tokens:
                tf = counts.get(term, 0)
                if tf == 0:
                    continue
                idf = self._idf.get(term, 0.0)
                numerator = tf * (k1 + 1.0)
                denominator = tf + k1 * (1.0 - b + b * (doc_len / (self._avg_doc_len + 1e-9)))
                score += idf * (numerator / (denominator + 1e-9))
            bm25_scores.append(score)
        bm25_arr = np.array(bm25_scores, dtype=np.float32)
        if bm25_arr.size:
            b_min, b_max = float(bm25_arr.min()), float(bm25_arr.max())
            bm25_norm = (bm25_arr - b_min) / (b_max - b_min + 1e-9)
        else:
            bm25_norm = np.array([], dtype=np.float32)

        hybrid_scores = alpha * dense_norm + (1.0 - alpha) * bm25_norm
        ranked = sorted(
            zip(dense_candidates, hybrid_scores.tolist()),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return [
            RetrievedItem(id=item.id, score=float(score), payload=item.payload)
            for item, score in ranked[:top_k]
        ]


def pretty_print_results(results: Iterable[RetrievedItem]) -> None:
    for rank, item in enumerate(results, start=1):
        title = item.payload.get("product_name") or item.payload.get("label") or "untitled"
        print(f"{rank}. id={item.id} score={item.score:.4f} title={title}")
