import numpy as np
import pandas as pd
import pytest

from src.data.classical_baseline import (
    bootstrap_metrics,
    build_estimator,
    classification_metrics,
    model_score_and_prediction,
    select_hyperparameters,
    stratified_bootstrap_indices,
)


def test_classification_metrics_use_er_positive_and_expected_confusion_order():
    y = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.8, 0.7, 0.9])
    predicted = np.array([0, 1, 1, 1])
    result = classification_metrics(y, score, predicted)
    assert result["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 0, "tp": 2}
    assert result["sensitivity"] == 1.0
    assert result["specificity"] == 0.5
    assert result["balanced_accuracy"] == 0.75


def cv_fixture() -> pd.DataFrame:
    rows = []
    for model, values in {
        "logistic_l1": {0.1: [0.8, 0.8], 1.0: [0.8, 0.8]},
        "linear_svm": {0.1: [0.79, 0.81], 1.0: [0.82, 0.82]},
    }.items():
        for c_value, aucs in values.items():
            for fold, auc in enumerate(aucs):
                rows.append(
                    {
                        "model": model,
                        "C": c_value,
                        "fold": fold,
                        "roc_auc": auc,
                        "pr_auc": auc,
                        "balanced_accuracy": 0.7,
                        "sensitivity": 0.8,
                        "specificity": 0.6,
                        "f1": 0.75,
                        "accuracy": 0.72,
                        "n_features": 100,
                        "preprocessing_runtime_seconds": 1.0,
                        "training_runtime_seconds": 0.2,
                        "inference_runtime_seconds": 0.1,
                    }
                )
    return pd.DataFrame(rows)


def test_selection_uses_mean_roc_auc_smallest_c_tie_and_cv_family_choice():
    _, selected = select_hyperparameters(cv_fixture())
    assert selected["models"]["logistic_l1"]["C"] == 0.1
    assert selected["models"]["linear_svm"]["C"] == 1.0
    assert selected["classical_reference_model"] == "linear_svm"


def test_stratified_bootstrap_is_reproducible_and_preserves_class_counts():
    y = np.array([0, 0, 0, 1, 1])
    first = stratified_bootstrap_indices(y, n_resamples=25, seed=2718)
    second = stratified_bootstrap_indices(y, n_resamples=25, seed=2718)
    assert np.array_equal(first, second)
    for sample in first:
        assert int((y[sample] == 0).sum()) == 3
        assert int((y[sample] == 1).sum()) == 2


def test_bootstrap_metrics_returns_requested_percentile_intervals():
    predictions = pd.DataFrame(
        {
            "y_true": [0, 0, 0, 1, 1, 1],
            "score": [0.1, 0.2, 0.4, 0.6, 0.8, 0.9],
            "predicted_class": [0, 0, 0, 1, 1, 1],
        }
    )
    indices = stratified_bootstrap_indices(predictions["y_true"], 20, 7)
    rows, intervals = bootstrap_metrics(predictions, indices, 0.95)
    assert len(rows) == 20
    assert set(intervals) == {
        "roc_auc",
        "pr_auc",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1",
        "accuracy",
    }
    assert intervals["roc_auc"] == {"lower": 1.0, "upper": 1.0}


@pytest.mark.parametrize(
    ("name", "fixed", "expected_type"),
    [
        (
            "logistic_l2",
            {"penalty": "l2", "solver": "liblinear", "dual": True},
            "LogisticRegression",
        ),
        (
            "logistic_l1",
            {"penalty": "l1", "solver": "liblinear", "dual": False},
            "LogisticRegression",
        ),
        (
            "linear_svm",
            {"penalty": "l2", "loss": "squared_hinge", "dual": "auto"},
            "LinearSVC",
        ),
        (
            "rbf_svm",
            {"kernel": "rbf", "gamma": "scale", "probability": False},
            "SVC",
        ),
    ],
)
def test_model_factory_preserves_frozen_configuration(name, fixed, expected_type):
    estimator = build_estimator(name, {"fixed_parameters": fixed}, 0.1)
    assert type(estimator).__name__ == expected_type
    assert estimator.C == 0.1
    for key, value in fixed.items():
        assert getattr(estimator, key) == value


class DummyLogistic:
    def predict_proba(self, X):
        return np.array([[0.5, 0.5], [0.5001, 0.4999]])


class DummySVM:
    def decision_function(self, X):
        return np.array([0.0, -0.001])


def test_fixed_thresholds_are_not_data_optimized():
    X = pd.DataFrame({"x": [1.0, 2.0]})
    _, logistic_pred, _ = model_score_and_prediction("logistic_l2", DummyLogistic(), X)
    _, svm_pred, _ = model_score_and_prediction("linear_svm", DummySVM(), X)
    assert logistic_pred.tolist() == [1, 0]
    assert svm_pred.tolist() == [1, 0]
