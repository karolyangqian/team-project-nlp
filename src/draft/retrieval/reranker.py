"""Cross-encoder reranking interface and SentenceTransformer implementation."""

from typing import Protocol

import numpy as np


class BaseReranker(Protocol):
    """Protocol for query-passage relevance rerankers."""

    def score_pairs(self, query: str, passages: list[str]) -> list[float]:
        """Return relevance scores in (0, 1) aligned with passages."""
        ...


class CrossEncoderReranker:
    """CrossEncoder reranker normalizing raw logits with a numerically stable sigmoid."""

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        max_length: int = 512,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(model_name, device=device, max_length=max_length)

    def score_pairs(self, query: str, passages: list[str]) -> list[float]:
        """Score (query, passage) pairs and return sigmoid probabilities."""
        if not passages:
            return []
        pairs = [(query, passage) for passage in passages]
        logits = np.asarray(
            self.model.predict(pairs, show_progress_bar=False),
            dtype=np.float64,
        ).reshape(-1)
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
        return [float(score) for score in probs.tolist()]
