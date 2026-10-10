from copy import deepcopy

import pytest

from src.data.xgboost_protocol import (
    balanced_sample_weights,
    complexity_key,
    select_candidate,
    validate_protocol,
)


def candidates():
    base = {
        "learning_rate": 0.03,
        "min_child_weight": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.1,
        "reg_alpha": 1.0,
        "reg_lambda": 10.0,
    }
    return [
        {"id": "simple", "n_estimators": 250, "max_depth": 2, **base},
        {"id": "complex", "n_estimators": 300, "max_depth": 3, **base},
    ]


def test_balanced_sample_weights_equalize_total_weight_per_class():
    y = [0, 0, 1, 1, 1, 1]
    weights = balanced_sample_weights(y)
    assert sum(weight for weight, label in zip(weights, y) if label == 0) == 3.0
    assert sum(weight for weight, label in zip(weights, y) if label == 1) == 3.0
    assert weights[:2] == [1.5, 1.5]
    assert weights[2:] == [0.75] * 4


def test_balanced_weights_require_both_classes():
    with pytest.raises(ValueError, match="both ER classes"):
        balanced_sample_weights([1, 1, 1])


def test_tie_band_prefers_lower_complexity():
    result = select_candidate(
        {"simple": 0.95 - 5e-13, "complex": 0.9500}, candidates(), tolerance=1e-12
    )
    assert result == "simple"
    assert complexity_key(candidates()[0]) < complexity_key(candidates()[1])


def test_outside_tie_band_selects_highest_mean_auc():
    result = select_candidate(
        {"simple": 0.95 - 2e-12, "complex": 0.9500}, candidates(), tolerance=1e-12
    )
    assert result == "complex"


def test_selection_requires_all_and_only_candidate_scores():
    with pytest.raises(ValueError, match="every candidate"):
        select_candidate({"simple": 0.9}, candidates(), tolerance=0.001)


def full_protocol_fixture():
    six = []
    for index in range(6):
        six.append(
            {
                "id": f"c{index}",
                "n_estimators": 100 + index,
                "max_depth": 2,
                "learning_rate": 0.03,
                "min_child_weight": 5,
                "subsample": 0.8,
                "colsample_bytree": 0.1,
                "reg_alpha": 1.0,
                "reg_lambda": 10.0,
            }
        )
    return {
        "role": "post-hoc complementary classical baseline",
        "must_not_replace_primary_reference": True,
        "primary_reference": {"model": "logistic_l1", "displayed_value": 0.9660},
        "target": {"ER_positive": 1, "ER_negative": 0, "positive_class": "ER_positive"},
        "preprocessing": {"standard_scaler": False, "supervised_feature_selection": False},
        "imbalance": {
            "scale_pos_weight": 1.0,
            "strategy": "exact_fold_local_sklearn_balanced_sample_weight",
            "prohibited_alternative_selection": "do_not_tune_weighting_strategy",
        },
        "candidates": six,
        "expected_execution": {"inner_fold_count": 5, "cv_model_fits": 30, "total_model_fits": 31},
        "selection": {
            "locked_test_in_selection": False,
            "numerical_tie_tolerance_auc": 1e-12,
            "rule": "strict_argmax_mean_inner_validation_ROC_AUC_then_numerical_tie_break",
        },
        "evaluation": {"bootstrap": {"mode": "reuse_frozen_indices_exactly", "regenerate": False}},
        "feature_importance": {"descriptive_only": True, "prohibited_as_phase2_or_vqc_selector": True},
    }


def test_protocol_invariants_pass_and_detect_scaling_or_retroactive_change():
    protocol = full_protocol_fixture()
    assert all(item["passed"] for item in validate_protocol(protocol))
    tampered = deepcopy(protocol)
    tampered["preprocessing"]["standard_scaler"] = True
    tampered["must_not_replace_primary_reference"] = False
    failed = {item["check"] for item in validate_protocol(tampered) if not item["passed"]}
    assert "no_standard_scaler_or_supervised_selection" in failed
    assert "primary_reference_immutable" in failed
