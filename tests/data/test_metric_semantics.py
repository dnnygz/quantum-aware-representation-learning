import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.metrics import average_precision_score

from src.data.clinical import parse_er_status
from src.data.metric_semantics import (
    ER_NEGATIVE_LABEL,
    ER_POSITIVE_LABEL,
    er_negative_score,
    explicit_er_metrics,
)
from src.data.phase1c_semantic_audit import build_semantic_audit


def test_er_status_mapping_is_er_positive_one_er_negative_zero():
    assert ER_POSITIVE_LABEL == 1
    assert ER_NEGATIVE_LABEL == 0
    assert parse_er_status("Positive") == 1
    assert parse_er_status("Negative") == 0


def logistic_l1_fixture():
    y = np.array([0] * 47 + [1] * 162)
    predicted = np.array([0] * 46 + [1] + [0] * 25 + [1] * 137)
    score = np.linspace(0.0, 1.0, len(y))
    return y, predicted, score


def test_confusion_matrix_order_and_named_cells_are_explicit():
    y, predicted, score = logistic_l1_fixture()
    metrics = explicit_er_metrics(y, score, predicted, "positive_class_probability")
    assert metrics["confusion_matrix_explicit"] == {
        "true_ER_negative_predicted_ER_negative_TN": 46,
        "true_ER_negative_predicted_ER_positive_FP": 1,
        "true_ER_positive_predicted_ER_negative_FN": 25,
        "true_ER_positive_predicted_ER_positive_TP": 137,
    }


def test_sensitivity_and_specificity_match_manual_cell_formulas():
    y, predicted, score = logistic_l1_fixture()
    metrics = explicit_er_metrics(y, score, predicted, "positive_class_probability")
    assert metrics["er_positive_sensitivity_recall"] == 137 / (137 + 25)
    assert metrics["er_negative_specificity_recall"] == 46 / (46 + 1)
    assert metrics["balanced_accuracy"] == ((137 / 162) + (46 / 47)) / 2


def test_er_negative_average_precision_inverts_label_and_probability_score():
    y = np.array([0, 1, 0, 1])
    er_positive_probability = np.array([0.1, 0.9, 0.4, 0.7])
    predicted = (er_positive_probability >= 0.5).astype(int)
    metrics = explicit_er_metrics(
        y, er_positive_probability, predicted, "positive_class_probability"
    )
    expected = average_precision_score(1 - y, 1 - er_positive_probability)
    assert metrics["er_negative_pr_auc_average_precision"] == expected
    assert np.array_equal(
        er_negative_score(er_positive_probability, "positive_class_probability"),
        1 - er_positive_probability,
    )


def test_er_negative_decision_score_reverses_direction():
    score = np.array([-2.0, 0.5, 1.2])
    assert np.array_equal(er_negative_score(score, "decision_function"), -score)


def test_semantic_audit_does_not_mutate_predictions_or_frozen_results():
    labels = pd.DataFrame(
        {
            "case_id": ["a", "b", "c", "d"],
            "er_status": ["ER_negative", "ER_negative", "ER_positive", "ER_positive"],
            "er_label": [0, 0, 1, 1],
        }
    )
    predictions = pd.DataFrame(
        {
            "patient_id": ["a", "b", "c", "d"],
            "y_true": [0, 0, 1, 1],
            "score": [0.1, 0.8, 0.7, 0.9],
            "predicted_class": [0, 1, 1, 1],
            "model": ["model"] * 4,
            "score_type": ["positive_class_probability"] * 4,
        }
    )
    metrics = explicit_er_metrics(
        predictions.y_true, predictions.score, predictions.predicted_class,
        "positive_class_probability",
    )
    frozen = {
        "models": {
            "model": {
                "metrics": {
                    "pr_auc": metrics["er_positive_pr_auc_average_precision"],
                    "sensitivity": metrics["er_positive_sensitivity_recall"],
                    "specificity": metrics["er_negative_specificity_recall"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "confusion_matrix": {"tn": 1, "fp": 1, "fn": 0, "tp": 2},
                }
            }
        }
    }
    labels_before = labels.copy(deep=True)
    predictions_before = predictions.copy(deep=True)
    frozen_before = repr(frozen)
    report = build_semantic_audit(labels, predictions, frozen)
    assert report["status"] == "pass"
    assert report["discrepancies"] == []
    assert_frame_equal(labels, labels_before)
    assert_frame_equal(predictions, predictions_before)
    assert repr(frozen) == frozen_before
