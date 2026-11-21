"""End-to-end tests for DraftPipeline, evaluation metrics, policy suite, and ablations."""

import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from draft.classifier.base import LabelEncoderPair
from draft.classifier.baseline import TfidfBaselineClassifier
from draft.config import load_config
from draft.data.org_docs import build_default_org_docs
from draft.evaluation.evaluator import PipelineEvaluator
from draft.evaluation.metrics import (
    build_policy_eval_suite,
    compute_placeholder_leakage_rate,
    compute_policy_adherence,
    compute_rouge_l,
)
from draft.generation.backends import MockGenerator
from draft.generation.prompts import PromptBuilder
from draft.pipeline import DraftPipeline
from draft.retrieval.dual_retriever import DualSourceRetriever
from draft.retrieval.hybrid_store import tokenize_lexical
from draft.schemas import (
    ClassificationResult,
    DocCategory,
    DraftResponse,
    IncomingEmail,
    IntentLabel,
    RetrievedContext,
    RetrievedDoc,
    UrgencyLevel,
)


class _OfflineEncoder:
    """Deterministic hash encoder for offline pipeline and evaluator tests."""

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim
        self.model_name_or_path = "offline-eval-encoder"

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """Hash lowercase tokens into unit-normalized float32 embeddings."""
        del is_query
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row_idx, text in enumerate(texts):
            for tok in tokenize_lexical(text):
                bucket = zlib.crc32(tok.encode("utf-8")) % self.dim
                matrix[row_idx, bucket] += 1.0
            norm = float(np.linalg.norm(matrix[row_idx]))
            if norm > 0.0:
                matrix[row_idx] /= norm
        return matrix


def _build_offline_pipeline(tmp_path: Path) -> tuple[DraftPipeline, pd.DataFrame]:
    """Assemble an offline DraftPipeline backed by baseline classifier and MockGenerator."""
    cfg = load_config("configs/default.yaml")
    labels = LabelEncoderPair.from_taxonomy(cfg.taxonomy)
    org_docs = build_default_org_docs()

    tickets = pd.DataFrame(
        [
            {
                "email_id": "EML00001",
                "subject": "Monthly invoice cycle",
                "body": "When are monthly billing invoices sent and what are payment terms?",
                "email": (
                    "Subject: Monthly invoice cycle\n\nWhen are monthly billing invoices sent?"
                ),
                "intent": str(IntentLabel.BILLING_AND_PAYMENTS),
                "urgency": str(UrgencyLevel.HIGH),
                "answer": "Invoices are issued on the first day of each month with Net 15 terms.",
                "tags": ["Billing", "Invoice"],
            },
            {
                "email_id": "EML00002",
                "subject": "Billing receipt request",
                "body": "Where can I download my monthly billing invoice PDF?",
                "email": (
                    "Subject: Billing receipt request\n\nDownload monthly billing invoice PDF."
                ),
                "intent": str(IntentLabel.BILLING_AND_PAYMENTS),
                "urgency": str(UrgencyLevel.MEDIUM),
                "answer": "Past PDF invoices are downloadable from the Billing Portal.",
                "tags": ["Billing"],
            },
            {
                "email_id": "EML00003",
                "subject": "Portal outage alert",
                "body": "The account management portal outage is blocking our whole team.",
                "email": "Subject: Portal outage alert\n\nAccount management portal outage.",
                "intent": str(IntentLabel.SERVICE_OUTAGES_AND_MAINTENANCE),
                "urgency": str(UrgencyLevel.HIGH),
                "answer": "Our incident team updates the Status Page every 30 minutes.",
                "tags": ["Outage"],
            },
            {
                "email_id": "EML00004",
                "subject": "Maintenance notice timing",
                "body": "How far ahead do you warn customers before scheduled maintenance?",
                "email": "Subject: Maintenance notice timing\n\nAdvance notice before maintenance?",
                "intent": str(IntentLabel.SERVICE_OUTAGES_AND_MAINTENANCE),
                "urgency": str(UrgencyLevel.LOW),
                "answer": "Routine maintenance happens on Sundays with 72 hours advance notice.",
                "tags": ["Maintenance"],
            },
        ]
    )

    clf = TfidfBaselineClassifier(labels=labels, min_df=1, random_state=42).fit(tickets)
    encoder = _OfflineEncoder()
    index_dir = DualSourceRetriever.build_and_save_indices(
        cfg, encoder, tickets, org_docs, output_dir=tmp_path / "idx"
    )
    retriever = DualSourceRetriever.load(cfg, encoder=encoder, index_dir=index_dir)
    prompt_builder = PromptBuilder(cfg.paths.prompts_dir)
    pipeline = DraftPipeline(cfg, clf, retriever, prompt_builder, MockGenerator())
    return pipeline, tickets


def test_draft_pipeline_run_with_timing_rag_and_no_rag(tmp_path: Path) -> None:
    """Verify DraftPipeline executes all three stages and toggles RAG cleanly."""
    pipeline, _ = _build_offline_pipeline(tmp_path)
    email = IncomingEmail(
        subject="Duplicate billing invoice",
        body="Urgent: we noticed a duplicate monthly billing invoice charge.",
    )

    rag_draft, timings = pipeline.run_with_timing(email, use_rag=True, use_reranker=False)
    assert rag_draft.subject == "Re: Duplicate billing invoice"
    assert len(rag_draft.retrieved_context.top_k_docs) == 3
    assert len(rag_draft.retrieved_context.top_k_responses) == 3
    assert len(rag_draft.cited_doc_ids) >= 1
    assert set(timings.keys()) == {"stage1_ms", "stage2_ms", "stage3_ms", "total_ms"}
    assert timings["total_ms"] >= 0.0

    no_rag_draft = pipeline.run(email, use_rag=False)
    assert no_rag_draft.retrieved_context.top_k_docs == []
    assert no_rag_draft.retrieved_context.top_k_responses == []
    assert no_rag_draft.cited_doc_ids == []


def test_evaluation_metrics_and_policy_suite() -> None:
    """Verify ROUGE scores, citation precision, urgency alignment, and 12-category policy suite."""
    rouge = compute_rouge_l(
        ["Invoices are issued on the first day of each month."],
        ["Invoices are issued on the first day of each month."],
    )
    assert rouge["rougeL_f1"] == pytest.approx(1.0)
    assert rouge["rouge1_f1"] == pytest.approx(1.0)

    docs = build_default_org_docs()
    suite = build_policy_eval_suite(docs)
    assert len(suite) == 12
    assert {item["expected_category"] for item in suite} == {str(c) for c in DocCategory}

    draft_valid = DraftResponse(
        subject="Re: Invoice",
        body="We prioritized this urgent issue per [DOC001] Net 15 billing schedule.",
        classification=ClassificationResult(
            intent=IntentLabel.BILLING_AND_PAYMENTS,
            urgency=UrgencyLevel.HIGH,
            intent_confidence=0.9,
            urgency_confidence=0.9,
        ),
        retrieved_context=RetrievedContext(
            top_k_docs=[
                RetrievedDoc(
                    doc_id="DOC001",
                    category=DocCategory.BILLING_AND_PAYMENTS,
                    title="Billing Cycle",
                    content="Invoices are issued monthly with Net 15 terms.",
                    score=0.9,
                )
            ]
        ),
        cited_doc_ids=["DOC001"],
        model_name="mock",
    )
    draft_hallucinated = DraftResponse(
        subject="Re: Other",
        body="Please see [DOC001] and [DOC999] for guidance.",
        classification=ClassificationResult(
            intent=IntentLabel.GENERAL_INQUIRY,
            urgency=UrgencyLevel.LOW,
            intent_confidence=0.8,
            urgency_confidence=0.8,
        ),
        retrieved_context=RetrievedContext(top_k_docs=draft_valid.retrieved_context.top_k_docs),
        cited_doc_ids=["DOC001"],
        model_name="mock",
    )

    adherence = compute_policy_adherence([draft_valid, draft_hallucinated])
    assert adherence["citation_rate"] == pytest.approx(1.0)
    assert adherence["valid_citation_precision"] == pytest.approx(0.75)
    assert adherence["urgency_alignment_rate"] == pytest.approx(1.0)
    assert adherence["policy_lexical_overlap"] > 0.0
    assert adherence["placeholder_leakage_rate"] == pytest.approx(0.0)
    assert compute_placeholder_leakage_rate(
        ["Hello <name>, check <acc_num>.", "Hello Jordan, invoices are Net 15."]
    ) == pytest.approx(0.5)


def test_pipeline_evaluator_tickets_policy_and_ablation(tmp_path: Path) -> None:
    """Verify PipelineEvaluator runs ticket, policy, and ablation benchmarks offline."""
    pipeline, tickets_df = _build_offline_pipeline(tmp_path)
    org_docs = build_default_org_docs()
    evaluator = PipelineEvaluator(pipeline)

    ablation = evaluator.run_ablation(
        eval_df=tickets_df,
        org_docs=org_docs,
        max_samples=4,
        use_reranker=False,
        include_bertscore=False,
    )
    assert set(ablation.keys()) == {"no_rag", "with_rag"}

    no_rag_t = ablation["no_rag"]["tickets"]
    with_rag_t = ablation["with_rag"]["tickets"]
    assert no_rag_t["num_samples"] == 4
    assert with_rag_t["policy_adherence"]["citation_rate"] > no_rag_t["policy_adherence"][
        "citation_rate"
    ]
    assert ablation["with_rag"]["policy_suite"]["num_scenarios"] == 12
    assert ablation["with_rag"]["policy_suite"]["doc_retrieval_hit_rate"] > 0.0
    assert ablation["no_rag"]["policy_suite"]["doc_retrieval_hit_rate"] == 0.0
