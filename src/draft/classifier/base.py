"""Shared classifier protocol, label encodings, class weights, and evaluation metrics."""

from collections import Counter
from collections.abc import Sequence
from typing import Any, Protocol

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from draft.config import TaxonomyConfig
from draft.schemas import ClassificationResult, IncomingEmail, IntentLabel, UrgencyLevel


class EmailClassifier(Protocol):
    """Protocol implemented by all Stage 1 intent and urgency classifiers."""

    def predict(self, email: IncomingEmail) -> ClassificationResult:
        """Classify a single incoming customer email."""
        ...

    def predict_batch(self, emails: list[IncomingEmail]) -> list[ClassificationResult]:
        """Classify a batch of incoming customer emails."""
        ...


class LabelEncoderPair:
    """Bidirectional integer mappings for intent and urgency labels."""

    def __init__(
        self,
        intent_labels: Sequence[IntentLabel | str],
        urgency_labels: Sequence[UrgencyLevel | str],
    ) -> None:
        self.intent_labels = [IntentLabel(label) for label in intent_labels]
        self.urgency_labels = [UrgencyLevel(label) for label in urgency_labels]
        self.intent_to_id = {label: idx for idx, label in enumerate(self.intent_labels)}
        self.urgency_to_id = {label: idx for idx, label in enumerate(self.urgency_labels)}

    @classmethod
    def from_taxonomy(cls, taxonomy: TaxonomyConfig) -> "LabelEncoderPair":
        """Build label encoders from validated taxonomy configuration."""
        return cls(taxonomy.intents, taxonomy.urgencies)

    def encode_intent(self, label: IntentLabel | str) -> int:
        """Convert an intent label string or enum into its integer ID."""
        return self.intent_to_id[IntentLabel(label)]

    def decode_intent(self, idx: int) -> IntentLabel:
        """Convert an integer ID back into an IntentLabel."""
        return self.intent_labels[idx]

    def encode_urgency(self, label: UrgencyLevel | str) -> int:
        """Convert an urgency level string or enum into its integer ID."""
        return self.urgency_to_id[UrgencyLevel(label)]

    def decode_urgency(self, idx: int) -> UrgencyLevel:
        """Convert an integer ID back into an UrgencyLevel."""
        return self.urgency_labels[idx]

    def to_dict(self) -> dict[str, list[str]]:
        """Serialize ordered label lists to JSON-compatible dictionaries."""
        return {
            "intent_labels": [str(label) for label in self.intent_labels],
            "urgency_labels": [str(label) for label in self.urgency_labels],
        }

    @classmethod
    def from_dict(cls, data: dict[str, list[str]]) -> "LabelEncoderPair":
        """Reconstruct LabelEncoderPair from a serialized dictionary."""
        return cls(data["intent_labels"], data["urgency_labels"])


def compute_inverse_class_weights(label_ids: Sequence[int], num_classes: int) -> list[float]:
    """Compute balanced inverse-frequency weights total / (num_classes * count)."""
    total = len(label_ids)
    if total == 0:
        return [1.0] * num_classes
    counts = Counter(label_ids)
    return [total / (num_classes * max(counts.get(idx, 0), 1)) for idx in range(num_classes)]


def _argmax(probs: Sequence[float]) -> int:
    """Return index of maximum value in a sequence."""
    return max(range(len(probs)), key=lambda idx: probs[idx])


def _clamp_prob(value: float) -> float:
    """Clamp probability floats to [0.0, 1.0] to guard minor numerical rounding."""
    return min(1.0, max(0.0, float(value)))


def build_classification_result(
    intent_probs: Sequence[float],
    urgency_probs: Sequence[float],
    labels: LabelEncoderPair,
) -> ClassificationResult:
    """Build a ClassificationResult from aligned intent and urgency probability vectors."""
    intent_idx = _argmax(intent_probs)
    urgency_idx = _argmax(urgency_probs)
    return ClassificationResult(
        intent=labels.decode_intent(intent_idx),
        urgency=labels.decode_urgency(urgency_idx),
        intent_confidence=_clamp_prob(intent_probs[intent_idx]),
        urgency_confidence=_clamp_prob(urgency_probs[urgency_idx]),
        intent_scores={
            str(label): _clamp_prob(prob)
            for label, prob in zip(labels.intent_labels, intent_probs, strict=True)
        },
        urgency_scores={
            str(label): _clamp_prob(prob)
            for label, prob in zip(labels.urgency_labels, urgency_probs, strict=True)
        },
    )


def _compute_head_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    label_names: list[str],
) -> dict[str, Any]:
    """Compute accuracy, macro/weighted F1, per-class metrics, and confusion matrix."""
    label_ids = list(range(len(label_names)))
    report = classification_report(
        y_true,
        y_pred,
        labels=label_ids,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    per_class = {
        name: {
            "precision": float(report[name]["precision"]),
            "recall": float(report[name]["recall"]),
            "f1": float(report[name]["f1-score"]),
            "support": int(report[name]["support"]),
        }
        for name in label_names
    }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=label_ids, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=label_ids, average="weighted", zero_division=0)
        ),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=label_ids).tolist(),
    }


def compute_classification_metrics(
    y_true_intent: Sequence[int],
    y_pred_intent: Sequence[int],
    y_true_urgency: Sequence[int],
    y_pred_urgency: Sequence[int],
    labels: LabelEncoderPair,
) -> dict[str, Any]:
    """Compute evaluation metrics for both intent and urgency heads."""
    intent_metrics = _compute_head_metrics(
        y_true_intent,
        y_pred_intent,
        [str(label) for label in labels.intent_labels],
    )
    urgency_metrics = _compute_head_metrics(
        y_true_urgency,
        y_pred_urgency,
        [str(label) for label in labels.urgency_labels],
    )
    return {
        "intent": intent_metrics,
        "urgency": urgency_metrics,
        "combined_macro_f1": (intent_metrics["macro_f1"] + urgency_metrics["macro_f1"]) / 2.0,
    }
