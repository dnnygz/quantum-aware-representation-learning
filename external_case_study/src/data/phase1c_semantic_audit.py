"""Audit Phase 1C label and metric semantics from persisted predictions only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import yaml

from .metric_semantics import explicit_er_metrics
from .utils import load_config, resolve_path, sha256_file, utc_now, write_json_atomic


PROTECTED_RELATIVE_PATHS = (
    "data/processed/phase1c_classical_baseline_v1/locked_test_predictions.parquet",
    "data/processed/phase1c_classical_baseline_v1/oof_predictions.parquet",
    "reports/data/phase1c_selected_hyperparameters.json",
    "reports/data/phase1c_locked_test_performance.json",
    "reports/data/phase1c_locked_test_performance.csv",
    "reports/data/phase1c_confusion_matrices.csv",
)


def _protected_hashes(repo_root: Path) -> dict[str, str]:
    return {path: sha256_file(repo_root / path) for path in PROTECTED_RELATIVE_PATHS}


def build_semantic_audit(
    labels: pd.DataFrame,
    predictions: pd.DataFrame,
    frozen_performance: dict[str, Any],
) -> dict[str, Any]:
    """Independently reconcile explicit metrics with frozen Phase 1C values."""

    required_labels = {"case_id", "er_status", "er_label"}
    required_predictions = {
        "patient_id", "y_true", "score", "predicted_class", "model", "score_type"
    }
    if missing := required_labels - set(labels):
        raise ValueError(f"Missing label columns: {sorted(missing)}")
    if missing := required_predictions - set(predictions):
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    if labels["case_id"].duplicated().any():
        raise ValueError("Canonical labels are not one row per patient")
    mapping_errors = labels.loc[
        ~(
            (labels["er_status"].eq("ER_positive") & labels["er_label"].eq(1))
            | (labels["er_status"].eq("ER_negative") & labels["er_label"].eq(0))
        )
    ]
    if not mapping_errors.empty:
        raise ValueError("Canonical er_status and er_label mapping is inconsistent")

    rows: list[dict[str, Any]] = []
    reconciliation: list[dict[str, Any]] = []
    for model_name, model_predictions in predictions.groupby("model", sort=True):
        if model_predictions["patient_id"].duplicated().any():
            raise ValueError(f"Duplicate locked-test patients for {model_name}")
        score_types = model_predictions["score_type"].unique().tolist()
        if len(score_types) != 1:
            raise ValueError(f"Mixed score types for {model_name}: {score_types}")
        metrics = explicit_er_metrics(
            model_predictions["y_true"].to_numpy(),
            model_predictions["score"].to_numpy(),
            model_predictions["predicted_class"].to_numpy(),
            score_types[0],
        )
        row = {"model": model_name, "score_type": score_types[0], **metrics}
        rows.append(row)

        frozen = frozen_performance["models"][model_name]["metrics"]
        cm = metrics["confusion_matrix_explicit"]
        comparisons = {
            "legacy_pr_auc_equals_ER_positive_average_precision": (
                frozen["pr_auc"], metrics["er_positive_pr_auc_average_precision"]
            ),
            "legacy_sensitivity_equals_ER_positive_recall": (
                frozen["sensitivity"], metrics["er_positive_sensitivity_recall"]
            ),
            "legacy_specificity_equals_ER_negative_recall": (
                frozen["specificity"], metrics["er_negative_specificity_recall"]
            ),
            "balanced_accuracy": (frozen["balanced_accuracy"], metrics["balanced_accuracy"]),
            "TN": (frozen["confusion_matrix"]["tn"], cm["true_ER_negative_predicted_ER_negative_TN"]),
            "FP": (frozen["confusion_matrix"]["fp"], cm["true_ER_negative_predicted_ER_positive_FP"]),
            "FN": (frozen["confusion_matrix"]["fn"], cm["true_ER_positive_predicted_ER_negative_FN"]),
            "TP": (frozen["confusion_matrix"]["tp"], cm["true_ER_positive_predicted_ER_positive_TP"]),
        }
        for check, (reported, recomputed) in comparisons.items():
            reconciliation.append(
                {
                    "model": model_name,
                    "check": check,
                    "reported": float(reported),
                    "recomputed": float(recomputed),
                    "absolute_difference": abs(float(reported) - float(recomputed)),
                    "passed": abs(float(reported) - float(recomputed)) <= 1e-12,
                }
            )

    cohort_count = len(labels)
    er_positive_count = int(labels["er_label"].eq(1).sum())
    er_negative_count = int(labels["er_label"].eq(0).sum())
    failures = [item for item in reconciliation if not item["passed"]]
    return {
        "generated_at": utc_now(),
        "status": "pass" if not failures else "fail",
        "audit_scope": "persisted locked-test predictions; no fit, model selection, or inference",
        "class_semantics": {
            "ER_positive": 1,
            "ER_negative": 0,
            "positive_class_for_legacy_PR_AUC": "ER_positive",
            "legacy_metric_aliases": {
                "pr_auc": "ER_positive PR-AUC / average precision",
                "sensitivity": "ER_positive sensitivity / recall",
                "specificity": "ER_negative specificity / recall",
            },
        },
        "canonical_cohort": {
            "patients": cohort_count,
            "ER_positive": er_positive_count,
            "ER_negative": er_negative_count,
            "ER_positive_prevalence": float(er_positive_count / cohort_count),
            "ER_negative_prevalence": float(er_negative_count / cohort_count),
        },
        "models": {row["model"]: row for row in rows},
        "reconciliation": reconciliation,
        "discrepancies": failures,
    }


def run(config_path: str | Path = "configs/data.yaml") -> dict[str, Any]:
    config, repo_root = load_config(config_path)
    semantics_path = repo_root / "configs/label_semantics.yaml"
    semantics = yaml.safe_load(semantics_path.read_text(encoding="utf-8"))
    if semantics["class_mapping"] != {"ER_positive": 1, "ER_negative": 0}:
        raise ValueError("Configured ER class mapping differs from ER+=1 / ER−=0")
    if semantics["positive_class"] != "ER_positive" or semantics["positive_label"] != 1:
        raise ValueError("Configured positive class must be ER_positive=1")
    protected_before = _protected_hashes(repo_root)
    processed = resolve_path(repo_root, config["paths"]["processed_dir"])
    phase_output = resolve_path(repo_root, config["phase1c"]["outputs"]["processed_dir"])
    performance_path = resolve_path(
        repo_root, config["phase1c"]["outputs"]["locked_test_performance"]
    )
    labels = pd.read_parquet(processed / "labels.parquet")
    predictions = pd.read_parquet(phase_output / "locked_test_predictions.parquet")
    frozen_performance = json.loads(performance_path.read_text(encoding="utf-8"))
    report = build_semantic_audit(labels, predictions, frozen_performance)
    protected_after = _protected_hashes(repo_root)
    if protected_before != protected_after:
        raise RuntimeError("A frozen Phase 1C artifact changed during semantic audit")
    report["frozen_artifact_sha256_before"] = protected_before
    report["frozen_artifact_sha256_after"] = protected_after
    report["frozen_artifacts_unchanged"] = True
    report["label_semantics_config"] = str(semantics_path.relative_to(repo_root))
    report["label_semantics_config_sha256"] = sha256_file(semantics_path)

    output_json = repo_root / "reports/data/phase1c_label_metric_semantic_audit.json"
    write_json_atomic(output_json, report)
    flat_rows = []
    for model_name, model in report["models"].items():
        flat_rows.append(
            {
                key: value
                for key, value in model.items()
                if not isinstance(value, dict)
            }
        )
    pd.DataFrame(flat_rows).to_csv(
        repo_root / "reports/data/phase1c_label_metric_semantic_audit.csv", index=False
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args(argv)
    report = run(args.config)
    print(json.dumps({"status": report["status"], "discrepancies": len(report["discrepancies"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
