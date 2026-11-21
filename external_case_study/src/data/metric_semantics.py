"""Explicit ER-class semantics for descriptive classification metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


ER_NEGATIVE_LABEL = 0
ER_POSITIVE_LABEL = 1
ER_LABEL_TO_NAME = {ER_NEGATIVE_LABEL: "ER_negative", ER_POSITIVE_LABEL: "ER_positive"}


def er_negative_score(er_positive_score: np.ndarray, score_type: str) -> np.ndarray:
    """Orient a persisted ER+ score so larger values support ER−."""

    score = np.asarray(er_positive_score, dtype=float)
    if score_type == "positive_class_probability":
        return 1.0 - score
    if score_type == "decision_function":
        return -score
    raise ValueError(f"Unsupported Phase 1C score_type: {score_type}")


def explicit_er_metrics(
    y_true: np.ndarray,
    er_positive_score: np.ndarray,
    predicted_class: np.ndarray,
    score_type: str,
) -> dict[str, Any]:
    """Compute class-explicit metrics with ER+=1 and ER−=0.

    ``ER_negative_specificity_recall`` is the conventional specificity when
    ER+ is the positive class; it is also recall for the ER− class.
    """

    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(er_positive_score, dtype=float)
    predicted = np.asarray(predicted_class, dtype=int)
    if set(np.unique(y_true)) != {ER_NEGATIVE_LABEL, ER_POSITIVE_LABEL}:
        raise ValueError("Both ER_negative=0 and ER_positive=1 must be present")
    if not set(np.unique(predicted)).issubset({ER_NEGATIVE_LABEL, ER_POSITIVE_LABEL}):
        raise ValueError("Predictions must use ER_negative=0 and ER_positive=1")

    tn, fp, fn, tp = confusion_matrix(
        y_true, predicted, labels=[ER_NEGATIVE_LABEL, ER_POSITIVE_LABEL]
    ).ravel()
    er_positive_recall = float(tp / (tp + fn))
    er_negative_recall = float(tn / (tn + fp))
    negative_label = 1 - y_true
    negative_score = er_negative_score(score, score_type)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        predicted,
        labels=[ER_NEGATIVE_LABEL, ER_POSITIVE_LABEL],
        zero_division=0,
    )
    return {
        "class_mapping": {"ER_positive": 1, "ER_negative": 0},
        "n_patients": int(len(y_true)),
        "er_positive_count": int((y_true == ER_POSITIVE_LABEL).sum()),
        "er_negative_count": int((y_true == ER_NEGATIVE_LABEL).sum()),
        "er_positive_prevalence_ap_baseline": float(y_true.mean()),
        "er_negative_prevalence_ap_baseline": float(negative_label.mean()),
        "er_positive_pr_auc_average_precision": float(average_precision_score(y_true, score)),
        "er_negative_pr_auc_average_precision": float(
            average_precision_score(negative_label, negative_score)
        ),
        "er_positive_sensitivity_recall": er_positive_recall,
        "er_negative_specificity_recall": er_negative_recall,
        "er_positive_one_vs_rest_specificity": er_negative_recall,
        "er_negative_one_vs_rest_specificity": er_positive_recall,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "er_negative_precision": float(precision[0]),
        "er_positive_precision": float(precision[1]),
        "er_negative_f1": float(f1[0]),
        "er_positive_f1": float(f1[1]),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "class_support": {"ER_negative": int(support[0]), "ER_positive": int(support[1])},
        "confusion_matrix_explicit": {
            "true_ER_negative_predicted_ER_negative_TN": int(tn),
            "true_ER_negative_predicted_ER_positive_FP": int(fp),
            "true_ER_positive_predicted_ER_negative_FN": int(fn),
            "true_ER_positive_predicted_ER_positive_TP": int(tp),
        },
    }
