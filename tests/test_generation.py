"""Unit tests for Stage 3 prompt templates, backends, and SFT dataset builder."""

import json
import zlib
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest

from draft.config import load_config
from draft.generation.backends import (
    MockGenerator,
    OllamaGenerator,
    OpenAICompatibleGenerator,
)
from draft.generation.prompts import (
    PromptBuilder,
    PromptMessages,
    build_reply_subject,
    extract_cited_doc_ids,
    normalize_target_reply,
)
from draft.generation.sft_dataset import SftDatasetBuilder, build_policy_grounded_examples
from draft.retrieval.dual_retriever import DualSourceRetriever
from draft.retrieval.hybrid_store import tokenize_lexical
from draft.schemas import (
    ClassificationResult,
    DocCategory,
    IncomingEmail,
    IntentLabel,
    OrgDocument,
    RetrievedContext,
    RetrievedDoc,
    RetrievedResponse,
    UrgencyLevel,
)


class _TestEncoder:
    """Minimal offline hash encoder for SFT builder tests."""

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim
        self.model_name_or_path = "offline-test-encoder"

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """Hash tokens into L2-normalized float32 embeddings."""
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


def test_reply_normalization_citations_and_subject() -> None:
    """Verify HTML break cleanup, DOCxxx citation extraction, and subject formatting."""
    raw = "Hello <name>,<br/>  Please verify account <acc_num>.<BR>Best,<br />Support  "
    assert normalize_target_reply(raw) == (
        "Hello <name>,\nPlease verify account <acc_num>.\nBest,\nSupport"
    )

    body = "Per [DOC002] and DOC001, plus [DOC002] and [DOC999]."
    assert extract_cited_doc_ids(body) == ["DOC002", "DOC001", "DOC999"]
    assert extract_cited_doc_ids(body, valid_doc_ids={"DOC001", "DOC002"}) == [
        "DOC002",
        "DOC001",
    ]

    assert build_reply_subject("Monthly invoice") == "Re: Monthly invoice"
    assert build_reply_subject("Re: Monthly invoice") == "Re: Monthly invoice"
    assert build_reply_subject("re:   Outage alert") == "Re: Outage alert"
    assert build_reply_subject("   ") == "Re: Your Support Inquiry"


def test_prompt_builder_renders_all_stage3_sections() -> None:
    """PromptBuilder renders classification, policy docs, sanitized replies, and email."""
    cfg = load_config("configs/default.yaml")
    builder = PromptBuilder(cfg.paths.prompts_dir)

    email = IncomingEmail(
        subject="Duplicate billing charge",
        body="We were billed twice this month and need a refund.",
    )
    clf = ClassificationResult(
        intent=IntentLabel.BILLING_AND_PAYMENTS,
        urgency=UrgencyLevel.HIGH,
        intent_confidence=0.92,
        urgency_confidence=0.88,
    )
    context = RetrievedContext(
        top_k_docs=[
            RetrievedDoc(
                doc_id="DOC002",
                category=DocCategory.BILLING_AND_PAYMENTS,
                title="Disputed Charges",
                content="Customers may dispute charges within 30 calendar days.",
                score=0.95,
            )
        ],
        top_k_responses=[
            RetrievedResponse(
                email_id="EML00001",
                email="Subject: Overcharge\n\nCharged twice.",
                answer="Hello <name>,<br/>We are reviewing your invoice.",
                intent=IntentLabel.BILLING_AND_PAYMENTS,
                urgency=UrgencyLevel.HIGH,
                score=0.85,
            )
        ],
    )

    messages = builder.render(email, clf, context)
    assert "[DOCxxx]" in messages.system
    assert "### Ticket Classification" in messages.user
    assert "Intent: Billing and Payments" in messages.user
    assert "Urgency: high" in messages.user
    assert "[DOC002] Disputed Charges (Billing and Payments)" in messages.user
    assert "Hello <name>,\nWe are reviewing your invoice." in messages.user
    assert "<br/>" not in messages.user
    assert "Subject: Duplicate billing charge" in messages.user

    chat_with_shots = messages.to_chat_messages()
    assert len(chat_with_shots) == 6
    assert [m["role"] for m in chat_with_shots] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert "[DOC901]" in chat_with_shots[1]["content"]
    assert "[DOC902]" in chat_with_shots[4]["content"]

    no_shot_messages = builder.render(email, clf, context, use_few_shot=False)
    assert len(no_shot_messages.to_chat_messages()) == 2
    no_rag_messages = builder.render(email, clf, RetrievedContext(), use_few_shot=True)
    assert len(no_rag_messages.to_chat_messages()) == 2


def test_mock_ollama_and_openai_generators() -> None:
    """Verify MockGenerator, OllamaGenerator, and OpenAICompatibleGenerator with mocks."""
    prompt = PromptMessages(
        system="System rule.",
        user="Intent: Billing and Payments\nUrgency: high\n[DOC002] Disputed Charges",
    )

    mock_out = MockGenerator().generate(prompt)
    assert "prioritized it" in mock_out
    assert "[DOC002]" in mock_out

    captured_ollama: dict = {}

    def ollama_handler(request: httpx.Request) -> httpx.Response:
        captured_ollama["url"] = str(request.url)
        captured_ollama["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json={"message": {"content": "  Mocked Ollama reply.  "}})

    ollama_gen = OllamaGenerator(
        model_name="draft-qwen25-3b",
        base_url="http://localhost:11434",
        transport=httpx.MockTransport(ollama_handler),
    )
    assert ollama_gen.generate(prompt) == "Mocked Ollama reply."
    assert captured_ollama["url"] == "http://localhost:11434/api/chat"
    assert captured_ollama["body"]["model"] == "draft-qwen25-3b"

    def connect_fail_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    failing_ollama = OllamaGenerator(
        model_name="draft-qwen25-3b",
        transport=httpx.MockTransport(connect_fail_handler),
    )
    with pytest.raises(RuntimeError, match="Could not connect to Ollama"):
        failing_ollama.generate(prompt)

    captured_oai: dict = {}

    def openai_handler(request: httpx.Request) -> httpx.Response:
        captured_oai["auth"] = request.headers.get("authorization")
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "OpenAI reply."}}]},
        )

    oai_gen = OpenAICompatibleGenerator(
        model_name="qwen2.5-3b",
        base_url="https://api.example.com/v1",
        api_key="secret-token",
        transport=httpx.MockTransport(openai_handler),
    )
    assert oai_gen.generate(prompt) == "OpenAI reply."
    assert captured_oai["auth"] == "Bearer secret-token"


def test_sft_dataset_builder_policy_grounding_and_zero_leakage(tmp_path: Path) -> None:
    """Verify RAG SFT JSONL generation, self-exclusion, and policy-grounded citations."""
    cfg = load_config("configs/default.yaml")
    encoder = _TestEncoder()
    org_docs = [
        OrgDocument(
            doc_id="DOC001",
            category=DocCategory.BILLING_AND_PAYMENTS,
            title="Billing Cycle",
            content="Invoices are issued on the first day of each month.",
        ),
        OrgDocument(
            doc_id="DOC016",
            category=DocCategory.OUTAGES_AND_MAINTENANCE,
            title="Outage Updates",
            content="Status updates are published every 30 minutes.",
        ),
    ]
    train_df = pd.DataFrame(
        [
            {
                "email_id": "EML00001",
                "subject": "Invoice question",
                "body": "When are monthly billing invoices sent to customers?",
                "email": "Subject: Invoice question\n\nWhen are monthly billing invoices sent?",
                "intent": str(IntentLabel.BILLING_AND_PAYMENTS),
                "urgency": str(UrgencyLevel.MEDIUM),
                "answer": "Invoices are emailed monthly.<br/>Please check the portal.",
                "tags": ["Billing"],
            },
            {
                "email_id": "EML00002",
                "subject": "Billing receipt copy",
                "body": "Need a PDF copy of last month invoice receipt.",
                "email": "Subject: Billing receipt copy\n\nNeed a PDF copy of invoice receipt.",
                "intent": str(IntentLabel.BILLING_AND_PAYMENTS),
                "urgency": str(UrgencyLevel.LOW),
                "answer": "You can download PDF receipts inside Billing Settings.",
                "tags": ["Invoice"],
            },
            {
                "email_id": "EML00003",
                "subject": "Portal outage",
                "body": "The account management portal outage is blocking logins.",
                "email": "Subject: Portal outage\n\nAccount portal outage blocking logins.",
                "intent": str(IntentLabel.SERVICE_OUTAGES_AND_MAINTENANCE),
                "urgency": str(UrgencyLevel.HIGH),
                "answer": "Our engineers are resolving the portal outage urgently.",
                "tags": ["Outage"],
            },
            {
                "email_id": "EML00004",
                "subject": "Service downtime update",
                "body": "How often do you post outage maintenance status updates?",
                "email": "Subject: Service downtime update\n\nOutage status update frequency?",
                "intent": str(IntentLabel.SERVICE_OUTAGES_AND_MAINTENANCE),
                "urgency": str(UrgencyLevel.MEDIUM),
                "answer": "Updates are posted every 30 minutes on the status page.",
                "tags": ["Status"],
            },
        ]
    )
    index_dir = DualSourceRetriever.build_and_save_indices(
        cfg, encoder, train_df, org_docs, output_dir=tmp_path / "idx"
    )
    retriever = DualSourceRetriever.load(cfg, encoder=encoder, index_dir=index_dir)
    prompt_builder = PromptBuilder(cfg.paths.prompts_dir)

    policy_rows = build_policy_grounded_examples(org_docs, retriever, prompt_builder)
    assert len(policy_rows) == 4
    assert all(row["cited_doc_ids"] for row in policy_rows)

    sft_dir = tmp_path / "sft"
    summary = SftDatasetBuilder(cfg, retriever, prompt_builder).build(
        train_df=train_df,
        org_docs=org_docs,
        output_dir=sft_dir,
        num_ticket_samples=4,
        include_policy_grounded=True,
        policy_upsample=2,
        eval_fraction=0.25,
    )
    assert summary["ticket_examples"] == 4
    assert summary["policy_grounded_examples"] == 8
    assert summary["total_examples"] == 12
    assert summary["zero_self_leakage"] is True
    assert (sft_dir / "sft_train.jsonl").exists()
    assert (sft_dir / "sft_eval.jsonl").exists()
    assert (sft_dir / "sft_summary.json").exists()

    all_rows = [
        json.loads(line)
        for p in (sft_dir / "sft_train.jsonl", sft_dir / "sft_eval.jsonl")
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(all_rows) == 12
    for row in all_rows:
        assert len(row["messages"]) == 3
        assert [m["role"] for m in row["messages"]] == ["system", "user", "assistant"]
        assert "<br/>" not in row["messages"][2]["content"]
