"""Hybrid FAISS dense + BM25 lexical store using Weighted Reciprocal Rank Fusion."""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

BoostFn = Callable[[dict[str, Any]], float]
ExcludeFn = Callable[[dict[str, Any]], bool]


def tokenize_lexical(text: str) -> list[str]:
    """Extract lowercase alphanumeric tokens for BM25 indexing and querying."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Normalize 2D float32 embedding rows to unit L2 length."""
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return np.ascontiguousarray(arr / np.maximum(norms, 1e-12), dtype=np.float32)


@dataclass(frozen=True)
class HybridSearchHit:
    """Single ranked hit returned by HybridVectorStore."""

    index: int
    score: float
    metadata: dict[str, Any]


class HybridVectorStore:
    """In-memory and disk-persisted FAISS + BM25 hybrid retrieval store."""

    def __init__(
        self,
        faiss_index: faiss.IndexFlatIP,
        bm25: BM25Okapi,
        texts: list[str],
        metadata: list[dict[str, Any]],
    ) -> None:
        self.faiss_index = faiss_index
        self.bm25 = bm25
        self.texts = texts
        self.metadata = metadata

    @classmethod
    def build(
        cls,
        texts: list[str],
        embeddings: np.ndarray,
        metadata: list[dict[str, Any]],
    ) -> "HybridVectorStore":
        """Build FAISS IndexFlatIP and BM25Okapi indices from parallel texts and embeddings."""
        if not texts:
            raise ValueError("Cannot build HybridVectorStore from empty texts.")
        norm_embs = _l2_normalize(embeddings)
        if norm_embs.shape[0] != len(texts) or len(metadata) != len(texts):
            raise ValueError("Texts, embeddings, and metadata lengths must match.")

        index = faiss.IndexFlatIP(norm_embs.shape[1])
        index.add(norm_embs)
        tokenized = [tokenize_lexical(text) or [""] for text in texts]
        return cls(faiss_index=index, bm25=BM25Okapi(tokenized), texts=texts, metadata=metadata)

    def _dense_ranks(
        self,
        query_embedding: np.ndarray,
        pool: int,
        exclude_fn: ExcludeFn | None,
    ) -> dict[int, int]:
        """Collect 1-based dense FAISS ranks up to pool size while filtering excluded items."""
        fetch = len(self.texts) if exclude_fn is not None else min(len(self.texts), pool)
        if exclude_fn is not None:
            fetch = min(len(self.texts), max(pool * 2, pool + 16))
        q_vec = _l2_normalize(query_embedding)
        _, indices = self.faiss_index.search(q_vec, fetch)

        ranks: dict[int, int] = {}
        rank = 1
        for raw_idx in indices[0]:
            idx = int(raw_idx)
            if idx < 0:
                continue
            if exclude_fn is not None and exclude_fn(self.metadata[idx]):
                continue
            ranks[idx] = rank
            rank += 1
            if len(ranks) >= pool:
                break
        return ranks

    def _bm25_ranks(
        self,
        query_text: str,
        pool: int,
        exclude_fn: ExcludeFn | None,
    ) -> dict[int, int]:
        """Collect 1-based BM25 lexical ranks for documents with positive term overlap."""
        tokens = tokenize_lexical(query_text)
        if not tokens:
            return {}
        scores = self.bm25.get_scores(tokens)
        ordered = np.argsort(-scores)

        ranks: dict[int, int] = {}
        rank = 1
        for raw_idx in ordered:
            idx = int(raw_idx)
            if scores[idx] <= 0.0:
                break
            if exclude_fn is not None and exclude_fn(self.metadata[idx]):
                continue
            ranks[idx] = rank
            rank += 1
            if len(ranks) >= pool:
                break
        return ranks

    def search(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        top_k: int,
        dense_weight: float = 0.65,
        bm25_weight: float = 0.35,
        rrf_k: int = 60,
        boost_fn: BoostFn | None = None,
        exclude_fn: ExcludeFn | None = None,
        candidate_pool: int | None = None,
    ) -> list[HybridSearchHit]:
        """Search with weighted RRF fusion over dense + BM25 ranks and multiplicative soft boost."""
        pool = candidate_pool or min(len(self.texts), max(top_k * 8, 50))
        dense_ranks = (
            self._dense_ranks(query_embedding, pool, exclude_fn) if dense_weight > 0.0 else {}
        )
        bm25_ranks = self._bm25_ranks(query_text, pool, exclude_fn) if bm25_weight > 0.0 else {}

        candidates = set(dense_ranks.keys()) | set(bm25_ranks.keys())
        scored: list[tuple[float, int]] = []
        for idx in candidates:
            rrf_score = 0.0
            if idx in dense_ranks:
                rrf_score += dense_weight / (rrf_k + dense_ranks[idx])
            if idx in bm25_ranks:
                rrf_score += bm25_weight / (rrf_k + bm25_ranks[idx])
            # Multiplicative boosting preserves rank proportionality; rank-1 RRF is ~1/(rrf_k+1)
            # (~0.016), so an additive +0.15 boost would overpower RRF and act like a hard filter.
            multiplier = boost_fn(self.metadata[idx]) if boost_fn is not None else 1.0
            scored.append((rrf_score * multiplier, idx))

        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [
            HybridSearchHit(index=idx, score=float(score), metadata=self.metadata[idx])
            for score, idx in scored[:top_k]
        ]

    def save(self, output_dir: Path) -> Path:
        """Persist FAISS index and UTF-8 JSON records to disk without pickle."""
        output_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.faiss_index, str(output_dir / "dense.faiss"))
        payload = {"texts": self.texts, "metadata": self.metadata}
        (output_dir / "records.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return output_dir

    @classmethod
    def load(cls, index_dir: Path) -> "HybridVectorStore":
        """Load persisted FAISS index and rebuild BM25Okapi from JSON records."""
        faiss_index = faiss.read_index(str(index_dir / "dense.faiss"))
        payload = json.loads((index_dir / "records.json").read_text(encoding="utf-8"))
        texts = payload["texts"]
        metadata = payload["metadata"]
        tokenized = [tokenize_lexical(text) or [""] for text in texts]
        return cls(
            faiss_index=faiss_index,
            bm25=BM25Okapi(tokenized),
            texts=texts,
            metadata=metadata,
        )
