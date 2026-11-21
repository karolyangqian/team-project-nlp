"""Unit tests for Stage 2 hybrid store, dual-source retriever, metrics, and trainer."""

import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from draft.config import load_config
from draft.retrieval.dual_retriever import DualSourceRetriever
from draft.retrieval.hybrid_store import HybridVectorStore, tokenize_lexical
from draft.retrieval.metrics import (
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    run_retrieval_ablations,
)
from draft.retrieval.trainer import build_contrastive_pairs
from draft.schemas import DocCategory, IncomingEmail, IntentLabel, OrgDocument, UrgencyLevel


class FakeHashEncoder:
    """Deterministic bag-of-tokens hash encoder for fast offline unit testing."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.model_name_or_path = "fake-hash-encoder"

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """Hash lowercase tokens into unit-normalized float32 vectors."""
        del is_query
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row_idx, text in enumerate(texts):
            for tok in tokenize_lexical(text):
                bucket = zlib.crc32(tok.encode("utf-8")) % self.dim
                matrix[row_idx, bucket] += 1.0
            norm = float(np.linalg.norm(matrix[row_idx]))
            if norm > 0.0:
                matrix[row_idx] /= norm
        return matrix


class FakeKeywordReranker:
    """Deterministic keyword-overlap reranker returning scores in (0, 1)."""

    def score_pairs(self, query: str, passages: list[str]) -> list[float]:
        """Score each passage by fraction of query tokens present."""
        q_tokens = set(tokenize_lexical(query))
        if not q_tokens:
            return [0.5 for _ in passages]
        scores: list[float] = []
        for passage in passages:
            p_tokens = set(tokenize_lexical(passage))
            overlap = len(q_tokens & p_tokens) / float(len(q_tokens))
            scores.append(0.1 + 0.8 * overlap)
        return scores


def test_hybrid_vector_store_search_soft_boost_exclude_and_persistence(tmp_path: Path) -> None:
    """Verify RRF fusion, multiplicative soft boost, exclusion, and disk round-trip."""
    texts = [
        "Monthly invoice billing cycle and payment reminders",
        "Monthly invoice billing tax certificate upload rules",
        "Server outage status page and incident recovery SLA",
        "Hardware RMA return shipping label instructions",
        "Data encryption AES256 and GDPR deletion compliance",
        "Employee onboarding laptop provisioning HR checklist",
    ]
    metadata = [
        {"doc_id": "DOC001", "category": "Billing and Payments"},
        {"doc_id": "DOC002", "category": "General Inquiry"},
        {"doc_id": "DOC003", "category": "Outages and Maintenance"},
        {"doc_id": "DOC004", "category": "Returns and Exchanges"},
        {"doc_id": "DOC005", "category": "Security and Compliance"},
        {"doc_id": "DOC006", "category": "Human Resources"},
    ]
    encoder = FakeHashEncoder(dim=256)
    store = HybridVectorStore.build(texts, encoder.encode(texts), metadata)

    query = "monthly invoice billing"
    q_emb = encoder.encode([query], is_query=True)[0]

    # Soft boost promotes Billing over an equally relevant General Inquiry doc.
    boosted_hits = store.search(
        query_text=query,
        query_embedding=q_emb,
        top_k=3,
        boost_fn=lambda m: 1.15 if m["category"] == "Billing and Payments" else 1.0,
    )
    assert boosted_hits[0].metadata["doc_id"] == "DOC001"

    # Multiplicative soft boost must not force an irrelevant boosted doc over strong matches.
    non_hard_filter_hits = store.search(
        query_text=query,
        query_embedding=q_emb,
        top_k=2,
        boost_fn=lambda m: 1.15 if m["category"] == "Returns and Exchanges" else 1.0,
    )
    top_ids = {hit.metadata["doc_id"] for hit in non_hard_filter_hits}
    assert "DOC004" not in top_ids

    # Exclude filter removes DOC001 cleanly.
    filtered = store.search(
        query_text=query,
        query_embedding=q_emb,
        top_k=2,
        exclude_fn=lambda m: m["doc_id"] == "DOC001",
    )
    assert all(hit.metadata["doc_id"] != "DOC001" for hit in filtered)

    saved_dir = store.save(tmp_path / "hybrid_store")
    reloaded = HybridVectorStore.load(saved_dir)
    reloaded_hits = reloaded.search(query_text=query, query_embedding=q_emb, top_k=3)
    assert [h.metadata["doc_id"] for h in reloaded_hits] == [
        h.metadata["doc_id"]
        for h in store.search(query_text=query, query_embedding=q_emb, top_k=3)
    ]


def test_dual_source_retriever_build_load_exclude_and_rerank(tmp_path: Path) -> None:
    """Verify DualSourceRetriever end-to-end indexing, loading, self-exclusion, and reranking."""
    cfg = load_config("configs/default.yaml")
    encoder = FakeHashEncoder(dim=32)
    reranker = FakeKeywordReranker()

    org_docs = [
        OrgDocument(
            doc_id="DOC001",
            category=DocCategory.BILLING_AND_PAYMENTS,
            title="Billing Cycle and Invoices",
            content="Invoices are issued monthly and cover billing portal downloads.",
        ),
        OrgDocument(
            doc_id="DOC002",
            category=DocCategory.OUTAGES_AND_MAINTENANCE,
            title="Outage Status Updates",
            content="Service outage incidents publish updates every 30 minutes.",
        ),
        OrgDocument(
            doc_id="DOC003",
            category=DocCategory.RETURNS_AND_EXCHANGES,
            title="Hardware Return RMA",
            content="Defective hardware returns require an approved RMA shipping label.",
        ),
    ]
    train_df = pd.DataFrame(
        [
            {
                "email_id": "EML00001",
                "email": "Subject: Invoice question\n\nNeed help downloading monthly invoice PDF.",
                "answer": "You can download your monthly PDF invoice from the Billing Portal.",
                "intent": str(IntentLabel.BILLING_AND_PAYMENTS),
                "urgency": str(UrgencyLevel.MEDIUM),
                "tags": ["Billing", "Invoice"],
            },
            {
                "email_id": "EML00002",
                "email": "Subject: Portal outage\n\nCentralized account portal appears offline.",
                "answer": "Our reliability team is investigating the portal outage.",
                "intent": str(IntentLabel.SERVICE_OUTAGES_AND_MAINTENANCE),
                "urgency": str(UrgencyLevel.HIGH),
                "tags": ["Outage", "Portal"],
            },
            {
                "email_id": "EML00003",
                "email": "Subject: RMA return\n\nNeed an RMA label for hardware device exchange.",
                "answer": "Please share your order number so we can issue an RMA label.",
                "intent": str(IntentLabel.RETURNS_AND_EXCHANGES),
                "urgency": str(UrgencyLevel.LOW),
                "tags": ["RMA", "Hardware"],
            },
        ]
    )

    index_dir = DualSourceRetriever.build_and_save_indices(
        config=cfg,
        encoder=encoder,
        train_df=train_df,
        org_docs=org_docs,
        output_dir=tmp_path / "index",
    )
    assert (index_dir / "index_meta.json").exists()

    retriever = DualSourceRetriever.load(
        config=cfg,
        encoder=encoder,
        reranker=reranker,
        index_dir=index_dir,
    )
    incoming = IncomingEmail(
        subject="Invoice question",
        body="Need help downloading monthly invoice PDF from billing portal.",
    )
    ctx = retriever.retrieve(
        incoming,
        intent=IntentLabel.BILLING_AND_PAYMENTS,
        top_k_docs=2,
        top_k_responses=2,
        exclude_email_ids={"EML00001"},
        use_reranker=True,
    )

    assert len(ctx.top_k_docs) == 2
    assert ctx.top_k_docs[0].doc_id == "DOC001"
    assert len(ctx.top_k_responses) == 2
    assert all(resp.email_id != "EML00001" for resp in ctx.top_k_responses)


def test_retrieval_metrics_and_ablations(tmp_path: Path) -> None:
    """Verify exact IR metrics formulas and ablation evaluation report structure."""
    flags = [False, True, False, True]
    assert hit_rate_at_k(flags, 1) == 0.0
    assert hit_rate_at_k(flags, 3) == 1.0
    assert precision_at_k(flags, 2) == pytest.approx(0.5)
    assert precision_at_k(flags, 4) == pytest.approx(0.5)
    assert mrr_at_k(flags, 3) == pytest.approx(0.5)
    assert ndcg_at_k([0.0, 1.0, 0.0], 3) == pytest.approx(1.0 / np.log2(3.0))

    cfg = load_config("configs/default.yaml")
    encoder = FakeHashEncoder(dim=32)
    reranker = FakeKeywordReranker()

    org_docs = [
        OrgDocument(
            doc_id="DOC001",
            category=DocCategory.BILLING_AND_PAYMENTS,
            title="Billing Invoices",
            content="Monthly invoice billing cycle and payment portal.",
        ),
        OrgDocument(
            doc_id="DOC002",
            category=DocCategory.OUTAGES_AND_MAINTENANCE,
            title="Outage Status",
            content="Service outage status updates and SLA credits.",
        ),
    ]
    train_df = pd.DataFrame(
        [
            {
                "email_id": "EML00001",
                "email": "Monthly invoice billing duplicate charge refund",
                "answer": "Billing team is checking your invoice.",
                "intent": str(IntentLabel.BILLING_AND_PAYMENTS),
                "urgency": str(UrgencyLevel.HIGH),
                "tags": ["Billing", "Refund"],
            },
            {
                "email_id": "EML00002",
                "email": "Service outage portal offline incident update",
                "answer": "Engineers are working on the outage.",
                "intent": str(IntentLabel.SERVICE_OUTAGES_AND_MAINTENANCE),
                "urgency": str(UrgencyLevel.HIGH),
                "tags": ["Outage"],
            },
        ]
    )
    index_dir = DualSourceRetriever.build_and_save_indices(
        cfg, encoder, train_df, org_docs, output_dir=tmp_path / "idx"
    )
    retriever = DualSourceRetriever.load(
        cfg, encoder=encoder, reranker=reranker, index_dir=index_dir
    )

    eval_df = pd.DataFrame(
        [
            {
                "email_id": "EML90001",
                "email": "Duplicate monthly invoice billing charge",
                "intent": str(IntentLabel.BILLING_AND_PAYMENTS),
                "urgency": str(UrgencyLevel.HIGH),
                "tags": ["Billing"],
            },
            {
                "email_id": "EML90002",
                "email": "Portal outage offline status update",
                "intent": str(IntentLabel.SERVICE_OUTAGES_AND_MAINTENANCE),
                "urgency": str(UrgencyLevel.HIGH),
                "tags": ["Outage"],
            },
        ]
    )
    ablations = run_retrieval_ablations(
        retriever,
        eval_df,
        k_values=(1, 2),
        max_queries=10,
        include_reranker=True,
    )
    assert ablations["num_queries"] == 2
    assert "hybrid_boosted_reranked" in ablations["org_docs"]
    assert ablations["org_docs"]["hybrid_boosted"]["category_hit@1"] == pytest.approx(1.0)
    assert ablations["ref_emails"]["hybrid_boosted"]["intent_hit@1"] == pytest.approx(1.0)
    assert ablations["ref_emails"]["hybrid_boosted"]["tag_hit@1"] == pytest.approx(1.0)

    anchors, positives = build_contrastive_pairs(train_df, max_samples=1, seed=42)
    assert len(anchors) == 1
    assert len(positives) == 1
