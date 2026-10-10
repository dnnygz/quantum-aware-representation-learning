"""Validate the post-hoc XGBoost protocol gate without importing or fitting XGBoost."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .utils import load_config, resolve_path, sha256_file, utc_now, write_json_atomic


REQUIRED_CANDIDATE_PARAMETERS = {
    "n_estimators",
    "max_depth",
    "learning_rate",
    "min_child_weight",
    "subsample",
    "colsample_bytree",
    "reg_alpha",
    "reg_lambda",
}

PROTECTED_PHASE1C_PATHS = (
    "data/processed/expression.parquet",
    "data/processed/labels.parquet",
    "data/processed/splits.json",
    "data/processed/splits/phase1c_inner_cv.json",
    "data/processed/phase1c_classical_baseline_v1/bootstrap_indices.npz",
    "data/processed/phase1c_classical_baseline_v1/bootstrap_patient_order.parquet",
    "data/processed/phase1c_classical_baseline_v1/oof_predictions.parquet",
    "data/processed/phase1c_classical_baseline_v1/locked_test_predictions.parquet",
    "reports/data/phase1c_selected_hyperparameters.json",
    "reports/data/phase1c_locked_test_performance.json",
)


def balanced_sample_weights(y: Sequence[int]) -> list[float]:
    """Return exact sklearn-balanced binary weights from training labels only."""

    labels = [int(value) for value in y]
    counts = {label: labels.count(label) for label in (0, 1)}
    if not counts[0] or not counts[1]:
        raise ValueError("Balanced weights require both ER classes")
    n_samples = len(labels)
    return [n_samples / (2.0 * counts[label]) for label in labels]


def complexity_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    """Frozen lower-complexity/higher-regularization ordering."""

    depth = int(candidate["max_depth"])
    return (
        depth,
        int(candidate["n_estimators"]) * (2**depth),
        float(candidate["colsample_bytree"]),
        -float(candidate["min_child_weight"]),
        -float(candidate["reg_lambda"]),
        -float(candidate["reg_alpha"]),
        float(candidate["learning_rate"]),
        str(candidate["id"]),
    )


def select_candidate(
    mean_validation_roc_auc: Mapping[str, float],
    candidates: Sequence[Mapping[str, Any]],
    tolerance: float,
) -> str:
    """Select within the frozen AUC tie band, then apply complexity ordering."""

    candidate_by_id = {str(candidate["id"]): candidate for candidate in candidates}
    if set(mean_validation_roc_auc) != set(candidate_by_id):
        raise ValueError("Scores must cover every candidate exactly once")
    best = max(float(value) for value in mean_validation_roc_auc.values())
    tied = [
        candidate_by_id[candidate_id]
        for candidate_id, value in mean_validation_roc_auc.items()
        if best - float(value) <= float(tolerance)
    ]
    return str(min(tied, key=complexity_key)["id"])


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_protocol(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate scientific invariants that do not require XGBoost."""

    candidates = list(protocol["candidates"])
    checks = [
        {
            "check": "posthoc_role_explicit",
            "passed": protocol["role"] == "post-hoc complementary classical baseline",
            "evidence": protocol["role"],
        },
        {
            "check": "primary_reference_immutable",
            "passed": protocol["must_not_replace_primary_reference"] is True
            and protocol["primary_reference"]["model"] == "logistic_l1"
            and float(protocol["primary_reference"]["displayed_value"]) == 0.9660,
            "evidence": protocol["primary_reference"],
        },
        {
            "check": "target_semantics",
            "passed": protocol["target"] == {
                "ER_positive": 1,
                "ER_negative": 0,
                "positive_class": "ER_positive",
            },
            "evidence": protocol["target"],
        },
        {
            "check": "no_standard_scaler_or_supervised_selection",
            "passed": protocol["preprocessing"]["standard_scaler"] is False
            and protocol["preprocessing"]["supervised_feature_selection"] is False,
            "evidence": protocol["preprocessing"],
        },
        {
            "check": "balanced_weights_prespecified_not_tuned",
            "passed": protocol["imbalance"]["scale_pos_weight"] == 1.0
            and protocol["imbalance"]["strategy"]
            == "exact_fold_local_sklearn_balanced_sample_weight"
            and protocol["imbalance"]["prohibited_alternative_selection"]
            == "do_not_tune_weighting_strategy",
            "evidence": protocol["imbalance"],
        },
        {
            "check": "candidate_set_complete_and_unique",
            "passed": len(candidates) == 6
            and len({item["id"] for item in candidates}) == 6
            and all(REQUIRED_CANDIDATE_PARAMETERS.issubset(item) for item in candidates),
            "evidence": [item["id"] for item in candidates],
        },
        {
            "check": "fit_count",
            "passed": protocol["expected_execution"]["cv_model_fits"]
            == len(candidates) * protocol["expected_execution"]["inner_fold_count"]
            and protocol["expected_execution"]["total_model_fits"]
            == protocol["expected_execution"]["cv_model_fits"] + 1,
            "evidence": protocol["expected_execution"],
        },
        {
            "check": "locked_test_not_used_for_selection",
            "passed": protocol["selection"]["locked_test_in_selection"] is False,
            "evidence": protocol["selection"],
        },
        {
            "check": "strict_argmax_with_computational_tie_only",
            "passed": float(protocol["selection"]["numerical_tie_tolerance_auc"])
            == 1e-12
            and protocol["selection"]["rule"]
            == "strict_argmax_mean_inner_validation_ROC_AUC_then_numerical_tie_break",
            "evidence": protocol["selection"],
        },
        {
            "check": "bootstrap_reuse_not_regeneration",
            "passed": protocol["evaluation"]["bootstrap"]["mode"]
            == "reuse_frozen_indices_exactly"
            and protocol["evaluation"]["bootstrap"]["regenerate"] is False,
            "evidence": protocol["evaluation"]["bootstrap"],
        },
        {
            "check": "importance_descriptive_only",
            "passed": protocol["feature_importance"]["descriptive_only"] is True
            and protocol["feature_importance"]["prohibited_as_phase2_or_vqc_selector"] is True,
            "evidence": protocol["feature_importance"],
        },
    ]
    return checks


def run(
    protocol_path: str | Path = "configs/xgboost_posthoc.yaml",
    data_config_path: str | Path = "configs/data.yaml",
) -> dict[str, Any]:
    protocol_path = Path(protocol_path).resolve()
    repo_root = protocol_path.parent.parent
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    data_config, data_repo_root = load_config(data_config_path)
    if data_repo_root != repo_root:
        raise ValueError("Protocol and data configuration belong to different repositories")

    protected_before = {
        path: sha256_file(repo_root / path) for path in PROTECTED_PHASE1C_PATHS
    }
    inner_path = resolve_path(repo_root, data_config["phase1c"]["inner_cv"]["artifact"])
    inner = json.loads(inner_path.read_text(encoding="utf-8"))
    bootstrap_path = repo_root / protocol["inputs"]["bootstrap_indices"]
    receipt_path = repo_root / "data/interim/phase1c_classical_baseline_v1/locked_test_access_receipt.json"
    phase1c_performance = json.loads(
        (repo_root / "reports/data/phase1c_locked_test_performance.json").read_text(encoding="utf-8")
    )
    checks = validate_protocol(protocol)
    checks.extend(
        [
            {
                "check": "same_frozen_inner_folds",
                "passed": len(inner["folds"]) == 5
                and len(inner["development_case_ids"]) == 835
                and len(inner["locked_test_case_ids"]) == 209,
                "evidence": {
                    "inner_cv_sha256": sha256_file(inner_path),
                    "development": len(inner["development_case_ids"]),
                    "locked_test": len(inner["locked_test_case_ids"]),
                },
            },
            {
                "check": "same_frozen_bootstrap_indices",
                "passed": sha256_file(bootstrap_path)
                == "f68d06b2d62321bc935bca91810ac4a7d3b40efad2914ba0b2dce5f60f7386b6",
                "evidence": sha256_file(bootstrap_path),
            },
            {
                "check": "original_locked_test_receipt_preserved",
                "passed": receipt_path.is_file(),
                "evidence": sha256_file(receipt_path),
            },
            {
                "check": "primary_reference_matches_frozen_report",
                "passed": abs(
                    float(phase1c_performance["P_classical_full"]["point"])
                    - float(protocol["primary_reference"]["value"])
                )
                <= 1e-15,
                "evidence": phase1c_performance["P_classical_full"],
            },
        ]
    )
    failures = [item["check"] for item in checks if not item["passed"]]
    protected_after = {
        path: sha256_file(repo_root / path) for path in PROTECTED_PHASE1C_PATHS
    }
    if protected_before != protected_after:
        failures.append("protected_phase1c_artifacts_changed_during_gate")
    report = {
        "generated_at": utc_now(),
        "status": "pass" if not failures else "fail",
        "training_performed": False,
        "xgboost_imported": False,
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": sha256_file(protocol_path),
        "protocol_semantic_sha256": _canonical_json_sha256(protocol),
        "checks": checks,
        "failures": failures,
        "protected_phase1c_sha256_before": protected_before,
        "protected_phase1c_sha256_after": protected_after,
    }
    output = repo_root / protocol["outputs"]["gate_report"]
    write_json_atomic(output, report)
    if failures:
        raise RuntimeError(f"XGBoost protocol gate failed: {failures}")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="configs/xgboost_posthoc.yaml")
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args(argv)
    report = run(args.protocol, args.config)
    print(json.dumps({"status": report["status"], "training_performed": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
