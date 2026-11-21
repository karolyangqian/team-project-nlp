"""Unit tests for Stage 1 classifiers, encoders, weighting, and metrics."""

from pathlib import Path

import pandas as pd
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import BertConfig, PreTrainedTokenizerFast

from draft.classifier.base import (
    LabelEncoderPair,
    compute_classification_metrics,
    compute_inverse_class_weights,
)
from draft.classifier.baseline import TfidfBaselineClassifier
from draft.classifier.multitask import (
    MultiTaskEmailClassifier,
    MultiTaskTrainer,
    TransformerEmailClassifier,
)
from draft.config import load_config
from draft.schemas import IncomingEmail, IntentLabel, UrgencyLevel


def _build_synthetic_dataframe(repeats: int = 3) -> pd.DataFrame:
    """Create a small deterministic email dataframe spanning intents and urgencies."""
    templates = [
        (
            "Invoice charge dispute",
            "Our monthly invoice billing charge was duplicated on our credit card.",
            IntentLabel.BILLING_AND_PAYMENTS,
            UrgencyLevel.HIGH,
        ),
        (
            "Portal login outage",
            "The centralized account portal outage is blocking team login access.",
            IntentLabel.SERVICE_OUTAGES_AND_MAINTENANCE,
            UrgencyLevel.HIGH,
        ),
        (
            "Webhook API error",
            "Our technical webhook integration endpoint returns a server error.",
            IntentLabel.TECHNICAL_SUPPORT,
            UrgencyLevel.MEDIUM,
        ),
        (
            "Hardware return RMA",
            "We need to return a defective hardware device for an exchange.",
            IntentLabel.RETURNS_AND_EXCHANGES,
            UrgencyLevel.LOW,
        ),
        (
            "Enterprise pricing quote",
            "Could sales send a formal pricing quote for fifty annual seats?",
            IntentLabel.SALES_AND_PRE_SALES,
            UrgencyLevel.LOW,
        ),
        (
            "New hire onboarding",
            "Please prepare employee onboarding laptop access for next week.",
            IntentLabel.HUMAN_RESOURCES,
            UrgencyLevel.MEDIUM,
        ),
    ]
    rows = []
    idx = 0
    for _ in range(repeats):
        for subject, body, intent, urgency in templates:
            idx += 1
            rows.append(
                {
                    "email_id": f"EML{idx:05d}",
                    "subject": subject,
                    "body": body,
                    "email": f"Subject: {subject}\n\n{body}",
                    "intent": str(intent),
                    "urgency": str(urgency),
                    "answer": "Thank you for contacting support.",
                }
            )
    return pd.DataFrame(rows)


def _build_tiny_tokenizer() -> PreTrainedTokenizerFast:
    """Build a minimal offline WordLevel fast tokenizer for tiny BERT unit tests."""
    tokens = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "subject",
        "invoice",
        "billing",
        "charge",
        "portal",
        "outage",
        "login",
        "webhook",
        "api",
        "error",
        "return",
        "hardware",
        "pricing",
        "quote",
        "onboarding",
        "access",
        "urgent",
        "support",
    ]
    vocab = {tok: idx for idx, tok in enumerate(tokens)}
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )


def test_label_encoder_weights_and_metrics() -> None:
    """Verify LabelEncoderPair serialization, class weights, and metric structure."""
    cfg = load_config("configs/default.yaml")
    labels = LabelEncoderPair.from_taxonomy(cfg.taxonomy)
    reloaded = LabelEncoderPair.from_dict(labels.to_dict())

    assert reloaded.intent_labels == labels.intent_labels
    assert reloaded.urgency_labels == labels.urgency_labels
    assert reloaded.decode_intent(reloaded.encode_intent("IT Support")) == IntentLabel.IT_SUPPORT
    assert reloaded.decode_urgency(reloaded.encode_urgency("high")) == UrgencyLevel.HIGH

    weights = compute_inverse_class_weights([0, 0, 0, 1], num_classes=3)
    assert weights[0] == pytest.approx(4 / (3 * 3))
    assert weights[1] == pytest.approx(4 / (3 * 1))
    assert weights[2] == pytest.approx(4 / (3 * 1))

    metrics = compute_classification_metrics(
        y_true_intent=[0, 1, 2],
        y_pred_intent=[0, 1, 0],
        y_true_urgency=[0, 1, 2],
        y_pred_urgency=[0, 1, 2],
        labels=labels,
    )
    assert metrics["urgency"]["accuracy"] == pytest.approx(1.0)
    assert metrics["intent"]["accuracy"] == pytest.approx(2 / 3)
    assert len(metrics["intent"]["confusion_matrix"]) == 10
    assert len(metrics["urgency"]["confusion_matrix"]) == 3
    assert metrics["combined_macro_f1"] == pytest.approx(
        (metrics["intent"]["macro_f1"] + metrics["urgency"]["macro_f1"]) / 2.0
    )


def test_tfidf_baseline_fit_predict_evaluate_and_persistence(tmp_path: Path) -> None:
    """Verify TF-IDF baseline training, batch prediction, evaluation, and reload parity."""
    cfg = load_config("configs/default.yaml")
    labels = LabelEncoderPair.from_taxonomy(cfg.taxonomy)
    train_df = _build_synthetic_dataframe(repeats=3)
    val_df = _build_synthetic_dataframe(repeats=1)

    baseline = TfidfBaselineClassifier(labels=labels, random_state=42)
    baseline.fit(train_df)

    sample = IncomingEmail(
        subject="Invoice billing error",
        body="We observed a duplicated invoice billing charge on our credit card.",
    )
    pred = baseline.predict(sample)
    assert pred.intent == IntentLabel.BILLING_AND_PAYMENTS
    assert sum(pred.intent_scores.values()) == pytest.approx(1.0, abs=1e-4)
    assert sum(pred.urgency_scores.values()) == pytest.approx(1.0, abs=1e-4)

    batch_preds = baseline.predict_batch(
        [sample, IncomingEmail(body="Portal outage login blocked")]
    )
    assert len(batch_preds) == 2

    metrics = baseline.evaluate(val_df)
    assert metrics["intent"]["accuracy"] >= 0.8

    saved_dir = baseline.save(tmp_path / "baseline")
    reloaded = TfidfBaselineClassifier.load(saved_dir)
    reloaded_pred = reloaded.predict(sample)
    assert reloaded_pred.intent == pred.intent
    assert reloaded_pred.urgency == pred.urgency


def test_multitask_transformer_offline_train_save_load_and_predict(tmp_path: Path) -> None:
    """Train, save, load, and evaluate MultiTaskEmailClassifier completely offline on CPU."""
    cfg = load_config("configs/default.yaml")
    cfg.classifier.epochs = 2
    cfg.classifier.batch_size = 4
    cfg.classifier.max_length = 32
    labels = LabelEncoderPair.from_taxonomy(cfg.taxonomy)

    tiny_config = BertConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    tiny_model = MultiTaskEmailClassifier(
        backbone_config=tiny_config,
        num_intents=len(labels.intent_labels),
        num_urgencies=len(labels.urgency_labels),
    )
    tiny_tokenizer = _build_tiny_tokenizer()

    train_df = _build_synthetic_dataframe(repeats=2)
    val_df = _build_synthetic_dataframe(repeats=1)
    test_df = _build_synthetic_dataframe(repeats=1)

    trainer = MultiTaskTrainer(
        config=cfg,
        labels=labels,
        model=tiny_model,
        tokenizer=tiny_tokenizer,
        device="cpu",
        use_amp=False,
    )
    checkpoint_dir = tmp_path / "transformer_ckpt"
    summary = trainer.fit(train_df, val_df, test_df, output_dir=checkpoint_dir)

    assert (checkpoint_dir / "model.safetensors").exists()
    assert (checkpoint_dir / "model_meta.json").exists()
    assert (checkpoint_dir / "metrics.json").exists()
    assert summary["best_epoch"] in {1, 2}
    assert len(summary["history"]) == 2

    loaded = TransformerEmailClassifier.load(checkpoint_dir, device="cpu")
    email = IncomingEmail(subject="Portal outage", body="portal outage login urgent")
    pred = loaded.predict(email)
    assert pred.intent in set(IntentLabel)
    assert pred.urgency in set(UrgencyLevel)
    assert sum(pred.intent_scores.values()) == pytest.approx(1.0, abs=1e-4)
    assert sum(pred.urgency_scores.values()) == pytest.approx(1.0, abs=1e-4)

    batch = loaded.predict_batch([email, IncomingEmail(body="invoice billing charge")])
    assert len(batch) == 2

    eval_metrics = loaded.evaluate(test_df, batch_size=4)
    assert eval_metrics["combined_macro_f1"] == pytest.approx(
        summary["test"]["combined_macro_f1"], abs=1e-5
    )
