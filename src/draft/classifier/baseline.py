"""Classical TF-IDF + LogisticRegression baseline for Stage 1 classification."""

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from draft.classifier.base import (
    LabelEncoderPair,
    build_classification_result,
    compute_classification_metrics,
)
from draft.schemas import ClassificationResult, IncomingEmail


def _align_class_probabilities(
    probs: np.ndarray,
    present_classes: np.ndarray,
    total_classes: int,
) -> np.ndarray:
    """Align classifier probability columns to the full fixed label space."""
    is_complete = probs.shape[1] == total_classes
    if is_complete and np.array_equal(present_classes, np.arange(total_classes)):
        return probs
    aligned = np.zeros((probs.shape[0], total_classes), dtype=np.float64)
    for col_idx, class_id in enumerate(present_classes):
        aligned[:, int(class_id)] = probs[:, col_idx]
    return aligned


class TfidfBaselineClassifier:
    """Shared TF-IDF vectorizer with balanced LogisticRegression heads for intent and urgency."""

    def __init__(
        self,
        labels: LabelEncoderPair,
        random_state: int = 42,
        max_features: int = 30000,
        min_df: int = 2,
    ) -> None:
        self.labels = labels
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=max_features,
            sublinear_tf=True,
            min_df=min_df,
            strip_accents="unicode",
        )
        self.intent_clf = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=2.0,
            random_state=random_state,
        )
        self.urgency_clf = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=2.0,
            random_state=random_state,
        )

    def fit(self, train_df: pd.DataFrame) -> "TfidfBaselineClassifier":
        """Fit shared TF-IDF features and both linear heads on training data."""
        features = self.vectorizer.fit_transform(train_df["email"].astype(str).tolist())
        y_intent = [self.labels.encode_intent(val) for val in train_df["intent"]]
        y_urgency = [self.labels.encode_urgency(val) for val in train_df["urgency"]]
        self.intent_clf.fit(features, y_intent)
        self.urgency_clf.fit(features, y_urgency)
        return self

    def predict_texts(self, texts: list[str]) -> list[ClassificationResult]:
        """Classify raw email strings into intent and urgency predictions."""
        if not texts:
            return []
        features = self.vectorizer.transform(texts)
        intent_probs = _align_class_probabilities(
            self.intent_clf.predict_proba(features),
            self.intent_clf.classes_,
            len(self.labels.intent_labels),
        )
        urgency_probs = _align_class_probabilities(
            self.urgency_clf.predict_proba(features),
            self.urgency_clf.classes_,
            len(self.labels.urgency_labels),
        )
        return [
            build_classification_result(i_probs.tolist(), u_probs.tolist(), self.labels)
            for i_probs, u_probs in zip(intent_probs, urgency_probs, strict=True)
        ]

    def predict(self, email: IncomingEmail) -> ClassificationResult:
        """Classify a single incoming customer email."""
        return self.predict_texts([email.formatted_text])[0]

    def predict_batch(self, emails: list[IncomingEmail]) -> list[ClassificationResult]:
        """Classify a batch of incoming customer emails."""
        return self.predict_texts([email.formatted_text for email in emails])

    def evaluate(self, df: pd.DataFrame) -> dict[str, Any]:
        """Evaluate baseline accuracy, F1 scores, and confusion matrices on a dataframe."""
        preds = self.predict_texts(df["email"].astype(str).tolist())
        y_true_intent = [self.labels.encode_intent(val) for val in df["intent"]]
        y_pred_intent = [self.labels.encode_intent(p.intent) for p in preds]
        y_true_urgency = [self.labels.encode_urgency(val) for val in df["urgency"]]
        y_pred_urgency = [self.labels.encode_urgency(p.urgency) for p in preds]
        return compute_classification_metrics(
            y_true_intent,
            y_pred_intent,
            y_true_urgency,
            y_pred_urgency,
            self.labels,
        )

    def save(self, output_dir: Path) -> Path:
        """Save vectorizer, classifiers, and label mappings to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, output_dir / "vectorizer.joblib")
        joblib.dump(self.intent_clf, output_dir / "intent_clf.joblib")
        joblib.dump(self.urgency_clf, output_dir / "urgency_clf.joblib")
        (output_dir / "labels.json").write_text(
            json.dumps(self.labels.to_dict(), indent=2),
            encoding="utf-8",
        )
        return output_dir

    @classmethod
    def load(cls, model_dir: Path) -> "TfidfBaselineClassifier":
        """Load a previously saved TF-IDF baseline classifier from disk."""
        labels_raw = json.loads((model_dir / "labels.json").read_text(encoding="utf-8"))
        instance = cls(labels=LabelEncoderPair.from_dict(labels_raw))
        instance.vectorizer = joblib.load(model_dir / "vectorizer.joblib")
        instance.intent_clf = joblib.load(model_dir / "intent_clf.joblib")
        instance.urgency_clf = joblib.load(model_dir / "urgency_clf.joblib")
        return instance
