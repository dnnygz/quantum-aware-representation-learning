import numpy as np
import pandas as pd
import pytest

from src.data.xgboost_baseline import (
    _load_counts,
    _paired_comparisons,
    build_model,
    scalar_metrics,
)


def minimal_protocol():
    return {
        "fixed_model_parameters": {
            "objective": "binary:logistic",
            "booster": "gbtree",
            "tree_method": "hist",
            "device": "cpu",
            "grow_policy": "depthwise",
            "max_bin": 256,
            "eval_metric": "auc",
            "random_state": 1729,
            "n_jobs": 4,
            "scale_pos_weight": 1.0,
        }
    }


def test_model_factory_preserves_frozen_parameters_and_no_scaler():
    candidate = {
        "id": "c",
        "n_estimators": 10,
        "max_depth": 2,
        "learning_rate": 0.03,
        "min_child_weight": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.1,
        "reg_alpha": 1.0,
        "reg_lambda": 10.0,
    }
    params = build_model(minimal_protocol(), candidate).get_params()
    assert params["scale_pos_weight"] == 1.0
    assert params["random_state"] == 1729
    assert params["n_estimators"] == 10
    assert "id" not in params


def test_explicit_metric_semantics_er_positive_is_one():
    y = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.8, 0.4, 0.9])
    predicted = np.array([0, 1, 0, 1])
    metrics = scalar_metrics(y, score, predicted)
    confusion = metrics["confusion_matrix_explicit"]
    assert confusion == {
        "true_ER_negative_predicted_ER_negative_TN": 1,
        "true_ER_negative_predicted_ER_positive_FP": 1,
        "true_ER_positive_predicted_ER_negative_FN": 1,
        "true_ER_positive_predicted_ER_positive_TP": 1,
    }
    assert metrics["er_positive_sensitivity_recall"] == 0.5
    assert metrics["er_negative_specificity_recall"] == 0.5


def test_load_counts_reads_only_requested_patients_and_preserves_order(tmp_path):
    frame = pd.DataFrame(
        {"g1": [1, 2, 3], "g2": [4, 5, 6]},
        index=pd.Index(["dev_a", "test", "dev_b"], name="case_id"),
    )
    path = tmp_path / "expression.parquet"
    frame.to_parquet(path)
    observed = _load_counts(path, ["dev_b", "dev_a"])
    assert observed.index.tolist() == ["dev_b", "dev_a"]
    assert "test" not in observed.index


def test_load_counts_rejects_missing_patient(tmp_path):
    pd.DataFrame(
        {"g1": [1]}, index=pd.Index(["dev"], name="case_id")
    ).to_parquet(tmp_path / "expression.parquet")
    with pytest.raises(ValueError, match="does not match"):
        _load_counts(tmp_path / "expression.parquet", ["dev", "missing"])


def test_paired_bootstrap_uses_same_indices_and_probability_orientation():
    patient_order = ["a", "b", "c", "d"]
    original = pd.DataFrame(
        {
            "patient_id": patient_order,
            "y_true": [0, 0, 1, 1],
            "score": [0.1, 0.2, 0.8, 0.9],
            "predicted_class": [0, 0, 1, 1],
            "model": ["logistic_l1"] * 4,
            "score_type": ["positive_class_probability"] * 4,
        }
    )
    indices = np.array([[0, 1, 2, 3]], dtype=np.int32)
    xgb_bootstrap = pd.DataFrame(
        {
            "bootstrap_id": [0],
            "roc_auc": [1.0],
            "er_positive_pr_auc_average_precision": [1.0],
            "er_negative_pr_auc_average_precision": [1.0],
            "balanced_accuracy": [1.0],
            "er_positive_sensitivity_recall": [1.0],
            "er_negative_specificity_recall": [1.0],
            "macro_f1": [1.0],
            "accuracy": [1.0],
        }
    )
    rows, summary = _paired_comparisons(xgb_bootstrap, original, patient_order, indices)
    assert np.allclose(rows["xgboost_minus_model"], 0.0)
    assert len(summary) == 8
