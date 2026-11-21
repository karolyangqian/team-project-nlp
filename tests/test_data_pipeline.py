"""Unit tests for configuration, cleaning, splitting, and organization docs."""

from pathlib import Path

import pandas as pd

from draft.config import DataConfig, load_config
from draft.data.cleaning import clean_email_dataframe, clean_text, normalize_key
from draft.data.org_docs import build_default_org_docs, load_org_docs, save_org_docs
from draft.data.splitter import create_canonical_splits
from draft.schemas import DocCategory, IncomingEmail, IntentLabel, UrgencyLevel, format_email_text


def test_clean_text_handles_nulls_escapes_and_whitespace() -> None:
    """clean_text should unescape literal newlines, trim spaces, and guard nulls."""
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""
    assert clean_text(pd.NA) == ""

    raw = "Hello   team,\\n\\n\\n  Line  two\\r\\n\\tIndented   end  "
    assert clean_text(raw) == "Hello team,\n\nLine two\nIndented end"


def test_format_email_text_and_incoming_email_parity() -> None:
    """format_email_text and IncomingEmail.formatted_text must produce identical output."""
    with_subj = format_email_text("  Billing Issue ", "Please inspect invoice #42. ")
    assert with_subj == "Subject: Billing Issue\n\nPlease inspect invoice #42."

    incoming = IncomingEmail(subject="  Billing Issue ", body="Please inspect invoice #42. ")
    assert incoming.formatted_text == with_subj

    without_subj = format_email_text("   ", "Body only message")
    assert without_subj == "Body only message"
    assert IncomingEmail(body="Body only message").formatted_text == without_subj


def test_clean_email_dataframe_filters_and_deduplicates() -> None:
    """clean_email_dataframe filters language, invalid labels, short text, and duplicates."""
    raw = pd.DataFrame(
        [
            {
                "language": "de",
                "queue": "Technical Support",
                "priority": "high",
                "type": "Incident",
                "subject": "German ticket",
                "body": "Dies ist ein langer deutscher Text fuer den Test.",
                "answer": "Vielen Dank fuer Ihre Nachricht an den Support.",
                "tag_1": "DE",
            },
            {
                "language": "en",
                "queue": "Unknown Queue",
                "priority": "high",
                "type": "Incident",
                "subject": "Invalid queue",
                "body": "Valid length body for testing invalid queue drop.",
                "answer": "Valid length answer for testing invalid queue drop.",
                "tag_1": None,
            },
            {
                "language": "en",
                "queue": "Technical Support",
                "priority": "medium",
                "type": "Request",
                "subject": "Short",
                "body": "Too short",
                "answer": "Too short",
                "tag_1": None,
            },
            {
                "language": "en",
                "queue": "Billing and Payments",
                "priority": "HIGH",
                "type": "Request",
                "subject": "Invoice Question",
                "body": "Hello team,\\nPlease send my latest monthly invoice.",
                "answer": "Thank you for reaching out. Here is your invoice info.",
                "tag_1": "Billing",
                "tag_2": "Invoice",
            },
            {
                "language": "en",
                "queue": "Billing and Payments",
                "priority": "low",
                "type": "Request",
                "subject": "invoice question",
                "body": "Hello team,   Please send my latest monthly invoice.",
                "answer": "Duplicate normalized email should be removed cleanly.",
                "tag_1": "Dup",
            },
        ]
    )
    config = DataConfig(
        hf_dataset_id="dummy/ds",
        language="en",
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_seed=42,
        min_email_chars=25,
        min_answer_chars=25,
    )
    cleaned, stats = clean_email_dataframe(
        raw,
        config=config,
        valid_intents={str(i) for i in IntentLabel},
        valid_urgencies={str(u) for u in UrgencyLevel},
    )

    assert stats["raw_rows"] == 5
    assert stats["after_language_filter"] == 4
    assert stats["after_label_filter"] == 3
    assert stats["after_text_length_filter"] == 2
    assert stats["after_deduplication"] == 1
    assert len(cleaned) == 1
    row = cleaned.iloc[0]
    assert row["email_id"] == "EML00001"
    assert row["urgency"] == "high"
    assert row["tags"] == ["Billing", "Invoice"]
    assert "\n\nHello team,\nPlease send my latest monthly invoice." in row["email"]


def test_create_canonical_splits_ratios_stratification_and_zero_leakage() -> None:
    """create_canonical_splits preserves approximate ratios and zero overlap across splits."""
    rows = []
    idx = 0
    for intent in ["Technical Support", "Billing and Payments"]:
        for urgency in ["low", "high"]:
            for _ in range(25):
                idx += 1
                rows.append(
                    {
                        "email_id": f"EML{idx:05d}",
                        "subject": f"Subject {idx}",
                        "body": f"Unique customer email body number {idx} for {intent}.",
                        "email": f"Subject: Subject {idx}\n\nUnique customer email body {idx}.",
                        "intent": intent,
                        "urgency": urgency,
                        "answer": f"Support answer number {idx} addressing the request.",
                        "ticket_type": "Request",
                        "tags": [],
                    }
                )
    df = pd.DataFrame(rows)
    config = DataConfig(
        hf_dataset_id="dummy/ds",
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_seed=42,
    )
    splits = create_canonical_splits(df, config)

    assert len(splits["train"]) == 80
    assert len(splits["val"]) == 10
    assert len(splits["test"]) == 10

    for name, split_df in splits.items():
        assert split_df["intent"].value_counts().to_dict() == {
            "Technical Support": len(split_df) // 2,
            "Billing and Payments": len(split_df) // 2,
        }, f"Unexpected stratification in {name}"

    train_keys = set(splits["train"]["email"].apply(normalize_key))
    val_keys = set(splits["val"]["email"].apply(normalize_key))
    test_keys = set(splits["test"]["email"].apply(normalize_key))
    assert train_keys.isdisjoint(val_keys)
    assert train_keys.isdisjoint(test_keys)
    assert val_keys.isdisjoint(test_keys)


def test_org_docs_synthesis_and_csv_roundtrip(tmp_path: Path) -> None:
    """Default org docs must contain 42 docs across all 12 categories and match DOC001."""
    docs = build_default_org_docs()
    assert len(docs) == 42

    doc_ids = [d.doc_id for d in docs]
    assert doc_ids == [f"DOC{i:03d}" for i in range(1, 43)]
    assert {d.category for d in docs} == set(DocCategory)

    first = docs[0]
    assert first.doc_id == "DOC001"
    assert first.category == DocCategory.BILLING_AND_PAYMENTS
    assert first.title == "Billing Cycle and Invoices"
    assert first.content.startswith(
        "Invoices are issued on the first day of each month for the previous month's usage "
        "and are emailed to the billing contact."
    )

    csv_path = tmp_path / "organization_docs.csv"
    save_org_docs(docs, csv_path)
    loaded = load_org_docs(csv_path)
    assert loaded == docs


def test_load_default_config_validates_taxonomy_and_mappings() -> None:
    """Default YAML config must load cleanly and map every intent to valid doc categories."""
    cfg = load_config("configs/default.yaml")
    assert set(cfg.taxonomy.intents) == set(IntentLabel)
    assert set(cfg.taxonomy.urgencies) == set(UrgencyLevel)
    assert set(cfg.taxonomy.doc_categories) == set(DocCategory)
    assert set(cfg.taxonomy.intent_to_doc_categories.keys()) == set(IntentLabel)
