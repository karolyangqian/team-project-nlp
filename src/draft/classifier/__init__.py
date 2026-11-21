"""Stage 1 email intent and urgency classification models and metrics."""

from draft.classifier.base import (
    EmailClassifier,
    LabelEncoderPair,
    build_classification_result,
    compute_classification_metrics,
    compute_inverse_class_weights,
)
from draft.classifier.baseline import TfidfBaselineClassifier

__all__ = [
    "EmailClassifier",
    "LabelEncoderPair",
    "TfidfBaselineClassifier",
    "build_classification_result",
    "compute_classification_metrics",
    "compute_inverse_class_weights",
]
