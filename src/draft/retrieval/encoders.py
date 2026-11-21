"""Text embedding protocols and SentenceTransformer encoder wrapper."""

from pathlib import Path
from typing import Protocol

import numpy as np


class TextEncoder(Protocol):
    """Protocol for dense text encoders returning L2-normalized float32 matrices."""

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """Encode texts into a contiguous L2-normalized float32 array of shape (N, D)."""
        ...


class SentenceTransformerEncoder:
    """SentenceTransformer bi-encoder wrapper with automatic BGE query prefixing."""

    _BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(
        self,
        model_name_or_path: str | Path,
        device: str | None = None,
        batch_size: int = 64,
        show_progress_bar: bool = False,
        query_prefix: str | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        model_ref = str(model_name_or_path)
        self.model = SentenceTransformer(model_ref, device=device)
        self.model_name_or_path = model_ref
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        if query_prefix is not None:
            self.query_prefix = query_prefix
        elif "bge" in model_ref.lower():
            self.query_prefix = self._BGE_QUERY_PREFIX
        else:
            self.query_prefix = ""

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """Encode input strings into contiguous L2-normalized float32 vectors."""
        if not texts:
            dim = int(self.model.get_sentence_embedding_dimension() or 0)
            return np.zeros((0, dim), dtype=np.float32)

        inputs = (
            [f"{self.query_prefix}{text}" for text in texts]
            if is_query and self.query_prefix
            else texts
        )
        embeddings = self.model.encode(
            inputs,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=self.show_progress_bar,
        )
        return np.ascontiguousarray(embeddings, dtype=np.float32)
