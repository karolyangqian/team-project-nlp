"""Information retrieval evaluation metrics and ablation runners."""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from draft.retrieval.dual_retriever import DualSourceRetriever
from draft.schemas import IntentLabel


def hit_rate_at_k(flags: Sequence[bool], k: int) -> float:
    """Return 1.0 if any relevant item appears within the top k results, else 0.0."""
    return 1.0 if any(flags[:k]) else 0.0


def precision_at_k(flags: Sequence[bool], k: int) -> float:
    """Return fraction of top k slots occupied by relevant items."""
    if k <= 0:
        return 0.0
    return sum(1 for flag in flags[:k] if flag) / float(k)


def mrr_at_k(flags: Sequence[bool], k: int) -> float:
    """Return reciprocal rank of the first relevant item within top k, else 0.0."""
    for idx, flag in enumerate(flags[:k], start=1):
        if flag:
            return 1.0 / float(idx)
    return 0.0


def ndcg_at_k(relevances: Sequence[float], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain at rank k."""
    if k <= 0 or not relevances:
        return 0.0
    dcg = sum(
        float(rel) / math.log2(rank + 1)
        for rank, rel in enumerate(relevances[:k], start=1)
        if rel > 0.0
    )
    ideal = sorted(relevances, reverse=True)[:k]
    idcg = sum(
        float(rel) / math.log2(rank + 1)
        for rank, rel in enumerate(ideal, start=1)
        if rel > 0.0
    )
    return dcg / idcg if idcg > 0.0 else 0.0


def _sample_eval_df(eval_df: pd.DataFrame, max_queries: int | None, seed: int) -> pd.DataFrame:
    """Deterministically subsample evaluation dataframe when max_queries is smaller."""
    if max_queries is not None and max_queries < len(eval_df):
        return eval_df.sample(n=max_queries, random_state=seed).reset_index(drop=True)
    return eval_df.reset_index(drop=True)


def _mean_metrics_over_k(
    rows_flags: list[list[bool]],
    k_values: Sequence[int],
    prefix: str,
) -> dict[str, float]:
    """Aggregate Hit@K, Precision@K, MRR@K, and binary NDCG@K across queries."""
    if not rows_flags:
        return {}
    metrics: dict[str, float] = {}
    n = float(len(rows_flags))
    for k in k_values:
        metrics[f"{prefix}hit@{k}"] = sum(hit_rate_at_k(f, k) for f in rows_flags) / n
        metrics[f"{prefix}precision@{k}"] = sum(precision_at_k(f, k) for f in rows_flags) / n
        metrics[f"{prefix}mrr@{k}"] = sum(mrr_at_k(f, k) for f in rows_flags) / n
        metrics[f"{prefix}ndcg@{k}"] = (
            sum(ndcg_at_k([1.0 if x else 0.0 for x in f], k) for f in rows_flags) / n
        )
    return metrics


def evaluate_reference_retrieval(
    retriever: DualSourceRetriever,
    eval_df: pd.DataFrame,
    k_values: Sequence[int] = (1, 3, 5),
    dense_weight: float | None = None,
    bm25_weight: float | None = None,
    intent_boost: float | None = None,
    max_queries: int | None = None,
    seed: int = 42,
    query_embeddings: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate historical reference retrieval on intent match and tag overlap."""
    df = _sample_eval_df(eval_df, max_queries=max_queries, seed=seed)
    queries = df["email"].astype(str).tolist()
    embs = (
        query_embeddings
        if query_embeddings is not None
        else retriever.encoder.encode(queries, is_query=True)
    )
    max_k = max(k_values)
    tags_by_id = {
        str(meta["email_id"]): {str(t) for t in meta.get("tags", [])}
        for meta in retriever.ref_emails_store.metadata
    }

    intent_flags_all: list[list[bool]] = []
    tag_flags_all: list[list[bool]] = []
    for row_idx, row in enumerate(df.to_dict(orient="records")):
        true_intent = IntentLabel(row["intent"])
        query_id = str(row["email_id"])
        raw_tags = row.get("tags")
        query_tags = {str(t) for t in raw_tags} if raw_tags is not None else set()

        results = retriever._retrieve_ref_responses_with_embedding(
            query_text=queries[row_idx],
            query_embedding=embs[row_idx],
            intent=true_intent,
            top_k=max_k,
            exclude_email_ids={query_id},
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
            intent_boost=intent_boost,
        )
        intent_flags_all.append([resp.intent == true_intent for resp in results])
        if query_tags:
            tag_flags_all.append(
                [bool(query_tags & tags_by_id.get(resp.email_id, set())) for resp in results]
            )

    summary: dict[str, Any] = {
        "num_queries": len(df),
        "num_tag_queries": len(tag_flags_all),
        **_mean_metrics_over_k(intent_flags_all, k_values, prefix="intent_"),
        **_mean_metrics_over_k(tag_flags_all, k_values, prefix="tag_"),
    }
    return summary


def evaluate_org_doc_retrieval(
    retriever: DualSourceRetriever,
    eval_df: pd.DataFrame,
    k_values: Sequence[int] = (1, 3, 5),
    use_reranker: bool = False,
    dense_weight: float | None = None,
    bm25_weight: float | None = None,
    intent_boost: float | None = None,
    max_queries: int | None = None,
    seed: int = 42,
    query_embeddings: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate organization policy retrieval against mapped intent doc categories."""
    df = _sample_eval_df(eval_df, max_queries=max_queries, seed=seed)
    queries = df["email"].astype(str).tolist()
    embs = (
        query_embeddings
        if query_embeddings is not None
        else retriever.encoder.encode(queries, is_query=True)
    )
    max_k = max(k_values)

    cat_flags_all: list[list[bool]] = []
    unique_doc_ids: set[str] = set()
    for row_idx, row in enumerate(df.to_dict(orient="records")):
        true_intent = IntentLabel(row["intent"])
        expected_cats = set(retriever.config.taxonomy.intent_to_doc_categories[true_intent])

        docs = retriever._retrieve_org_docs_with_embedding(
            query_text=queries[row_idx],
            query_embedding=embs[row_idx],
            intent=true_intent,
            top_k=max_k,
            use_reranker=use_reranker,
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
            intent_boost=intent_boost,
        )
        cat_flags_all.append([doc.category in expected_cats for doc in docs])
        unique_doc_ids.update(doc.doc_id for doc in docs)

    return {
        "num_queries": len(df),
        "unique_docs_retrieved": len(unique_doc_ids),
        **_mean_metrics_over_k(cat_flags_all, k_values, prefix="category_"),
    }


def run_retrieval_ablations(
    retriever: DualSourceRetriever,
    eval_df: pd.DataFrame,
    k_values: Sequence[int] = (1, 3, 5),
    max_queries: int | None = 300,
    include_reranker: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare BM25-only, dense-only, hybrid, soft-boosted, and optional reranked modes."""
    sampled = _sample_eval_df(eval_df, max_queries=max_queries, seed=seed)
    query_embs = retriever.encoder.encode(sampled["email"].astype(str).tolist(), is_query=True)
    cfg = retriever.config.retrieval

    modes = [
        ("bm25_only", 0.0, 1.0, 0.0, False),
        ("dense_only", 1.0, 0.0, 0.0, False),
        ("hybrid", cfg.dense_weight, cfg.bm25_weight, 0.0, False),
        ("hybrid_boosted", cfg.dense_weight, cfg.bm25_weight, cfg.intent_boost, False),
    ]

    org_docs_results: dict[str, Any] = {}
    ref_results: dict[str, Any] = {}
    for name, dw, bw, boost, rerank in modes:
        org_docs_results[name] = evaluate_org_doc_retrieval(
            retriever,
            sampled,
            k_values=k_values,
            use_reranker=rerank,
            dense_weight=dw,
            bm25_weight=bw,
            intent_boost=boost,
            query_embeddings=query_embs,
        )
        ref_results[name] = evaluate_reference_retrieval(
            retriever,
            sampled,
            k_values=k_values,
            dense_weight=dw,
            bm25_weight=bw,
            intent_boost=boost,
            query_embeddings=query_embs,
        )

    if include_reranker and retriever.reranker is not None:
        org_docs_results["hybrid_boosted_reranked"] = evaluate_org_doc_retrieval(
            retriever,
            sampled,
            k_values=k_values,
            use_reranker=True,
            dense_weight=cfg.dense_weight,
            bm25_weight=cfg.bm25_weight,
            intent_boost=cfg.intent_boost,
            query_embeddings=query_embs,
        )

    return {
        "num_queries": len(sampled),
        "org_docs": org_docs_results,
        "ref_emails": ref_results,
    }
