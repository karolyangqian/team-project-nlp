"""Dual-source retriever combining organization policy docs and reference replies."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from draft.config import AppConfig
from draft.retrieval.encoders import SentenceTransformerEncoder, TextEncoder
from draft.retrieval.hybrid_store import BoostFn, HybridSearchHit, HybridVectorStore
from draft.retrieval.reranker import BaseReranker, CrossEncoderReranker
from draft.schemas import (
    DocCategory,
    IncomingEmail,
    IntentLabel,
    OrgDocument,
    RetrievedContext,
    RetrievedDoc,
    RetrievedResponse,
    UrgencyLevel,
)


class DualSourceRetriever:
    """Retrieves Top-K policy docs and Top-K reference responses with soft intent boosting."""

    def __init__(
        self,
        config: AppConfig,
        encoder: TextEncoder,
        org_docs_store: HybridVectorStore,
        ref_emails_store: HybridVectorStore,
        reranker: BaseReranker | None = None,
    ) -> None:
        self.config = config
        self.encoder = encoder
        self.org_docs_store = org_docs_store
        self.ref_emails_store = ref_emails_store
        self.reranker = reranker

    def _doc_boost_fn(
        self,
        intent: IntentLabel | None,
        intent_boost: float | None = None,
    ) -> BoostFn | None:
        """Return multiplicative category boost function for a predicted intent."""
        boost = self.config.retrieval.intent_boost if intent_boost is None else intent_boost
        if intent is None or boost <= 0.0:
            return None
        preferred = {
            str(cat) for cat in self.config.taxonomy.intent_to_doc_categories[intent]
        }
        return lambda meta: (1.0 + boost) if meta.get("category") in preferred else 1.0

    def _email_boost_fn(
        self,
        intent: IntentLabel | None,
        intent_boost: float | None = None,
    ) -> BoostFn | None:
        """Return multiplicative intent boost function for historical email replies."""
        boost = self.config.retrieval.intent_boost if intent_boost is None else intent_boost
        if intent is None or boost <= 0.0:
            return None
        target = str(intent)
        return lambda meta: (1.0 + boost) if meta.get("intent") == target else 1.0

    def _rerank_doc_hits(
        self,
        query_text: str,
        hits: list[HybridSearchHit],
        boost_fn: BoostFn | None,
        top_k: int,
    ) -> list[HybridSearchHit]:
        """Rerank candidate policy docs using CrossEncoder while keeping soft intent preference."""
        if not hits or self.reranker is None:
            return hits[:top_k]
        passages = [f"{hit.metadata['title']}\n\n{hit.metadata['content']}" for hit in hits]
        ce_scores = self.reranker.score_pairs(query_text, passages)
        rescored = [
            HybridSearchHit(
                index=hit.index,
                score=ce_score * (boost_fn(hit.metadata) if boost_fn is not None else 1.0),
                metadata=hit.metadata,
            )
            for hit, ce_score in zip(hits, ce_scores, strict=True)
        ]
        rescored.sort(key=lambda item: (-item.score, item.index))
        return rescored[:top_k]

    def _retrieve_org_docs_with_embedding(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        intent: IntentLabel | None = None,
        top_k: int | None = None,
        use_reranker: bool = True,
        dense_weight: float | None = None,
        bm25_weight: float | None = None,
        intent_boost: float | None = None,
    ) -> list[RetrievedDoc]:
        """Retrieve Top-K organization policy documents using a precomputed query embedding."""
        cfg = self.config.retrieval
        k = top_k or cfg.top_k_docs
        dw = cfg.dense_weight if dense_weight is None else dense_weight
        bw = cfg.bm25_weight if bm25_weight is None else bm25_weight
        boost_fn = self._doc_boost_fn(intent, intent_boost=intent_boost)
        rerank_active = use_reranker and self.reranker is not None
        fetch_k = min(len(self.org_docs_store.texts), max(k * 4, 12)) if rerank_active else k

        hits = self.org_docs_store.search(
            query_text=query_text,
            query_embedding=query_embedding,
            top_k=fetch_k,
            dense_weight=dw,
            bm25_weight=bw,
            rrf_k=cfg.rrf_k,
            boost_fn=boost_fn,
        )
        if rerank_active:
            hits = self._rerank_doc_hits(query_text, hits, boost_fn, k)
        else:
            hits = hits[:k]

        return [
            RetrievedDoc(
                doc_id=str(hit.metadata["doc_id"]),
                category=DocCategory(hit.metadata["category"]),
                title=str(hit.metadata["title"]),
                content=str(hit.metadata["content"]),
                score=float(hit.score),
            )
            for hit in hits
        ]

    def _retrieve_ref_responses_with_embedding(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        intent: IntentLabel | None = None,
        top_k: int | None = None,
        exclude_email_ids: set[str] | None = None,
        dense_weight: float | None = None,
        bm25_weight: float | None = None,
        intent_boost: float | None = None,
    ) -> list[RetrievedResponse]:
        """Retrieve Top-K reference email responses using a precomputed query embedding."""
        cfg = self.config.retrieval
        k = top_k or cfg.top_k_responses
        dw = cfg.dense_weight if dense_weight is None else dense_weight
        bw = cfg.bm25_weight if bm25_weight is None else bm25_weight
        boost_fn = self._email_boost_fn(intent, intent_boost=intent_boost)
        excluded = set(exclude_email_ids) if exclude_email_ids else None
        exclude_fn = (lambda meta: meta["email_id"] in excluded) if excluded else None

        hits = self.ref_emails_store.search(
            query_text=query_text,
            query_embedding=query_embedding,
            top_k=k,
            dense_weight=dw,
            bm25_weight=bw,
            rrf_k=cfg.rrf_k,
            boost_fn=boost_fn,
            exclude_fn=exclude_fn,
        )
        return [
            RetrievedResponse(
                email_id=str(hit.metadata["email_id"]),
                email=str(hit.metadata["email"]),
                answer=str(hit.metadata["answer"]),
                intent=IntentLabel(hit.metadata["intent"]),
                urgency=UrgencyLevel(hit.metadata["urgency"]),
                score=float(hit.score),
            )
            for hit in hits
        ]

    def retrieve_org_docs(
        self,
        query_text: str,
        intent: IntentLabel | None = None,
        top_k: int | None = None,
        use_reranker: bool = True,
        dense_weight: float | None = None,
        bm25_weight: float | None = None,
        intent_boost: float | None = None,
    ) -> list[RetrievedDoc]:
        """Retrieve Top-K relevant organization policy documents."""
        query_emb = self.encoder.encode([query_text], is_query=True)[0]
        return self._retrieve_org_docs_with_embedding(
            query_text=query_text,
            query_embedding=query_emb,
            intent=intent,
            top_k=top_k,
            use_reranker=use_reranker,
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
            intent_boost=intent_boost,
        )

    def retrieve_reference_responses(
        self,
        query_text: str,
        intent: IntentLabel | None = None,
        top_k: int | None = None,
        exclude_email_ids: set[str] | None = None,
        dense_weight: float | None = None,
        bm25_weight: float | None = None,
        intent_boost: float | None = None,
    ) -> list[RetrievedResponse]:
        """Retrieve Top-K historical reference email-response pairs."""
        query_emb = self.encoder.encode([query_text], is_query=True)[0]
        return self._retrieve_ref_responses_with_embedding(
            query_text=query_text,
            query_embedding=query_emb,
            intent=intent,
            top_k=top_k,
            exclude_email_ids=exclude_email_ids,
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
            intent_boost=intent_boost,
        )

    def retrieve(
        self,
        email: IncomingEmail,
        intent: IntentLabel | None = None,
        top_k_docs: int | None = None,
        top_k_responses: int | None = None,
        exclude_email_ids: set[str] | None = None,
        use_reranker: bool = True,
    ) -> RetrievedContext:
        """Encode an incoming email once and retrieve both policy docs and reference replies."""
        query_text = email.formatted_text
        query_emb = self.encoder.encode([query_text], is_query=True)[0]
        docs = self._retrieve_org_docs_with_embedding(
            query_text=query_text,
            query_embedding=query_emb,
            intent=intent,
            top_k=top_k_docs,
            use_reranker=use_reranker,
        )
        responses = self._retrieve_ref_responses_with_embedding(
            query_text=query_text,
            query_embedding=query_emb,
            intent=intent,
            top_k=top_k_responses,
            exclude_email_ids=exclude_email_ids,
        )
        return RetrievedContext(top_k_docs=docs, top_k_responses=responses)

    @classmethod
    def build_and_save_indices(
        cls,
        config: AppConfig,
        encoder: TextEncoder,
        train_df: pd.DataFrame,
        org_docs: list[OrgDocument],
        output_dir: Path | None = None,
    ) -> Path:
        """Encode organization docs and train reference emails and save dual indices to disk."""
        target_dir = output_dir or (config.paths.artifacts_dir / "retriever" / "index")
        target_dir.mkdir(parents=True, exist_ok=True)

        doc_texts = [f"{doc.title}\n\n{doc.content}" for doc in org_docs]
        doc_meta = [doc.model_dump(mode="json") for doc in org_docs]
        doc_embs = encoder.encode(doc_texts, is_query=False)
        org_store = HybridVectorStore.build(doc_texts, doc_embs, doc_meta)
        org_store.save(target_dir / "org_docs")

        ref_texts = train_df["email"].astype(str).tolist()
        ref_meta: list[dict[str, Any]] = []
        for row in train_df.to_dict(orient="records"):
            raw_tags = row.get("tags")
            tags = [str(tag) for tag in raw_tags] if raw_tags is not None else []
            ref_meta.append(
                {
                    "email_id": str(row["email_id"]),
                    "email": str(row["email"]),
                    "answer": str(row["answer"]),
                    "intent": str(row["intent"]),
                    "urgency": str(row["urgency"]),
                    "tags": tags,
                }
            )
        ref_embs = encoder.encode(ref_texts, is_query=False)
        ref_store = HybridVectorStore.build(ref_texts, ref_embs, ref_meta)
        ref_store.save(target_dir / "ref_emails")

        meta = {
            "embedding_model": getattr(
                encoder, "model_name_or_path", config.retrieval.embedding_model
            ),
            "org_docs_count": len(org_docs),
            "ref_emails_count": len(train_df),
            "embedding_dim": int(doc_embs.shape[1]),
        }
        (target_dir / "index_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return target_dir

    @classmethod
    def load(
        cls,
        config: AppConfig,
        encoder: TextEncoder | None = None,
        reranker: BaseReranker | None = None,
        index_dir: Path | None = None,
        encoder_path: Path | None = None,
        enable_reranker: bool = False,
    ) -> "DualSourceRetriever":
        """Load persisted dual-source indices and resolve encoder and optional reranker."""
        resolved_index = index_dir or (config.paths.artifacts_dir / "retriever" / "index")
        missing_org = not (resolved_index / "org_docs").exists()
        missing_ref = not (resolved_index / "ref_emails").exists()
        if missing_org or missing_ref:
            raise FileNotFoundError(
                f"Dual-source retrieval index not found in {resolved_index}. "
                "Download pre-built artifacts with "
                "`uv run --with gdown python scripts/download_artifacts.py --profile core` "
                "or build locally with `uv run draft prepare-data` and `uv run draft build-index`."
            )
        org_store = HybridVectorStore.load(resolved_index / "org_docs")
        ref_store = HybridVectorStore.load(resolved_index / "ref_emails")

        resolved_encoder = encoder
        if resolved_encoder is None:
            default_ft_dir = config.paths.artifacts_dir / "retriever" / "encoder"
            fallback_ref = (
                default_ft_dir if default_ft_dir.exists() else config.retrieval.embedding_model
            )
            model_ref: str | Path = encoder_path if encoder_path is not None else fallback_ref
            resolved_encoder = SentenceTransformerEncoder(model_ref)

        resolved_reranker = reranker
        if resolved_reranker is None and enable_reranker:
            resolved_reranker = CrossEncoderReranker(config.retrieval.reranker_model)

        return cls(
            config=config,
            encoder=resolved_encoder,
            org_docs_store=org_store,
            ref_emails_store=ref_store,
            reranker=resolved_reranker,
        )
