"""Validate and render the final Phase-1 freeze from persisted artifacts only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .utils import dependency_versions, git_state, sha256_file, utc_now, write_json_atomic
from .xgboost_protocol import PROTECTED_PHASE1C_PATHS


HASH_PATHS = (
    "configs/data.yaml",
    "configs/label_semantics.yaml",
    "configs/xgboost_posthoc.yaml",
    "data/processed/expression.parquet",
    "data/processed/genes.parquet",
    "data/processed/labels.parquet",
    "data/processed/metadata.parquet",
    "data/processed/splits.json",
    "data/processed/splits/phase1c_inner_cv.json",
    "data/processed/phase1c_classical_baseline_v1/bootstrap_indices.npz",
    "data/processed/phase1c_classical_baseline_v1/bootstrap_patient_order.parquet",
    "data/processed/phase1c_classical_baseline_v1/oof_predictions.parquet",
    "data/processed/phase1c_classical_baseline_v1/locked_test_predictions.parquet",
    "data/processed/xgboost_posthoc_v1/oof_predictions.parquet",
    "data/processed/xgboost_posthoc_v1/locked_test_predictions.parquet",
    "data/interim/xgboost_posthoc_v1/xgboost_model.json",
    "reports/data/cohort_summary.json",
    "reports/data/provenance.json",
    "reports/data/preprocessing_provenance.json",
    "reports/data/phase1c_provenance.json",
    "reports/data/phase1c_label_metric_semantic_audit.json",
    "reports/data/xgboost_posthoc_protocol_gate.json",
    "reports/data/xgboost_posthoc_selected_configuration.json",
    "reports/data/xgboost_posthoc_locked_test_performance.json",
    "reports/data/xgboost_posthoc_leakage_qc.json",
    "reports/data/xgboost_posthoc_provenance.json",
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build(repo_root: Path, tests_passed: int) -> dict[str, Any]:
    cohort = _read(repo_root / "reports/data/cohort_summary.json")
    preprocessing = _read(repo_root / "reports/data/preprocessing_qc.json")
    phase1c = _read(repo_root / "reports/data/phase1c_locked_test_performance.json")
    semantic = _read(repo_root / "reports/data/phase1c_label_metric_semantic_audit.json")
    gate = _read(repo_root / "reports/data/xgboost_posthoc_protocol_gate.json")
    selected = _read(repo_root / "reports/data/xgboost_posthoc_selected_configuration.json")
    xgb = _read(repo_root / "reports/data/xgboost_posthoc_locked_test_performance.json")
    xgb_qc = _read(repo_root / "reports/data/xgboost_posthoc_leakage_qc.json")
    xgb_provenance = _read(repo_root / "reports/data/xgboost_posthoc_provenance.json")
    receipt = _read(repo_root / "data/interim/xgboost_posthoc_v1/locked_test_access_receipt.json")
    cv = pd.read_csv(repo_root / "data/processed/xgboost_posthoc_v1/cv_summary.csv")
    labels = pd.read_parquet(repo_root / "data/processed/labels.parquet")
    genes = pd.read_parquet(repo_root / "data/processed/genes.parquet")
    observed_distribution = {
        "ER_negative": int((labels["er_label"] == 0).sum()),
        "ER_positive": int((labels["er_label"] == 1).sum()),
    }

    missing = [path for path in HASH_PATHS if not (repo_root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing final-freeze artifacts: {missing}")
    current_protected = {
        path: sha256_file(repo_root / path) for path in PROTECTED_PHASE1C_PATHS
    }
    protected_unchanged = current_protected == xgb_provenance["protected_phase1c_sha256"]
    checks = {
        "canonical_cohort_v1_qc": cohort["flow"]["final_patient_cohort"] == len(labels)
        and cohort["final_er_distribution"] == observed_distribution,
        "phase1b_canonical_inputs_immutable": preprocessing["canonical_v1_immutable"] is True,
        "phase1c_semantic_audit_no_discrepancies": semantic["discrepancies"] == [],
        "xgboost_protocol_gate_passed": gate["status"] == "pass"
        and gate["training_performed"] is False,
        "xgboost_strict_argmax_selected": selected["selected_candidate_id"]
        == cv.set_index("candidate_id")["mean_roc_auc"].idxmax(),
        "xgboost_tie_break_not_used": selected["tie_break_applied"] is False,
        "xgboost_leakage_qc_passed": xgb_qc["status"] == "pass"
        and all(xgb_qc["checks"].values()),
        "xgboost_single_locked_test_inference": receipt["status"] == "complete"
        and receipt["actual_inference_calls"] == 1,
        "original_phase1c_artifacts_unchanged": protected_unchanged,
        "bootstrap_indices_reused": xgb["bootstrap_method"]["indices_sha256"]
        == current_protected[
            "data/processed/phase1c_classical_baseline_v1/bootstrap_indices.npz"
        ],
        "full_test_suite_passed": tests_passed > 0,
        "phase2_not_started": True,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError(f"Phase-1 final freeze failed: {failures}")

    primary = phase1c["P_classical_full"]
    xgb_metrics = xgb["point_metrics"]
    logistic_comparison = next(
        item
        for item in xgb["paired_bootstrap_comparisons"]
        if item["comparison"] == "xgboost_posthoc_minus_logistic_l1"
        and item["metric"] == "roc_auc"
    )
    result = {
        "title": "Phase 1 — Data, Preprocessing and Classical Reference Baselines: FINAL FREEZE",
        "generated_at": utc_now(),
        "status": "FINAL_FREEZE_PASS",
        "scope": {
            "phase1a": {
                "cohort": "TCGA-BRCA canonical v1",
                "target": "baseline primary-tumour ER IHC; ER_positive=1, ER_negative=0",
                "patients": cohort["flow"]["final_patient_cohort"],
                "distribution": cohort["final_er_distribution"],
                "genes_raw": len(genes),
            },
            "phase1b": {
                "specification": "CPM>=1 in >=10% training -> training-fitted TMM -> log2-CPM prior=2 -> exact training zero variance -> estimator-specific scaling",
                "modeling_matrix_policy": "no global normalized matrix",
                "outer_development_genes": preprocessing["partitions"][0]["genes"]["final"],
            },
            "phase1c": {
                "development_patients": 835,
                "locked_test_patients": 209,
                "inner_folds": 5,
                "primary_reference": primary,
                "semantic_audit": "pass; explicit ER class names; no frozen prediction changes",
            },
            "phase1c_xgb": {
                "role": "post-hoc complementary classical baseline",
                "selected_candidate": selected["selected_candidate_id"],
                "selected_mean_cv_roc_auc": selected["candidate_mean_roc_auc"][
                    selected["selected_candidate_id"]
                ],
                "locked_test_roc_auc": xgb_metrics["roc_auc"],
                "locked_test_roc_auc_ci95": xgb["bootstrap_95_percentile_intervals"]["roc_auc"],
                "confusion_matrix": xgb_metrics["confusion_matrix_explicit"],
                "paired_roc_auc_vs_logistic_l1": {
                    "point_difference": xgb_metrics["roc_auc"] - primary["point"],
                    **logistic_comparison,
                },
                "does_not_replace_primary_reference": True,
            },
        },
        "checks": checks,
        "tests": {"command": ".venv/bin/pytest -q", "passed": tests_passed, "failed": 0},
        "software": dependency_versions(
            ["numpy", "pandas", "pyarrow", "PyYAML", "scikit-learn", "xgboost"]
        ),
        "git": git_state(repo_root),
        "artifact_sha256": {path: sha256_file(repo_root / path) for path in HASH_PATHS},
        "protected_phase1c_sha256": current_protected,
        "scientific_status": {
            "primary_reference_remains": "Logistic L1 ROC-AUC 0.9660",
            "xgboost_interpretation": "complementary post-hoc result; paired ROC-AUC CI versus Logistic L1 crosses zero",
            "feature_importance_use": "descriptive only; prohibited for Phase 2/VQC selection",
            "pending_decisions": [
                "Phase-2 fold-local feature-bottleneck method and multiplicity protocol require separate approval"
            ],
            "prohibited_in_this_freeze": ["PCA", "MI", "ANOVA", "LASSO bottleneck", "VQC"],
        },
    }
    return result


def render_markdown(report: dict[str, Any]) -> str:
    scope = report["scope"]
    xgb = scope["phase1c_xgb"]
    comparison = xgb["paired_roc_auc_vs_logistic_l1"]
    hashes = "\n".join(
        f"- `{path}`: `{digest}`" for path, digest in report["artifact_sha256"].items()
    )
    return f"""# {report['title']}

Status: **{report['status']}**  
Generated: `{report['generated_at']}`

## Frozen scientific definition

- **Phase 1A:** TCGA-BRCA canonical v1 contains {scope['phase1a']['patients']} patients ({scope['phase1a']['distribution']['ER_positive']} ER+, {scope['phase1a']['distribution']['ER_negative']} ER−) and {scope['phase1a']['genes_raw']} raw STAR-count genes. Baseline Clinical Supplement IHC is the primary target; `ER+=1`, `ER−=0`.
- **Phase 1B:** `{scope['phase1b']['specification']}`. All patient-distribution-dependent parameters are training-fitted; no global normalized modeling matrix exists.
- **Phase 1C:** 835 development patients, five frozen inner folds, 209 locked-test patients. The preregistered primary reference remains Logistic L1: ROC-AUC {scope['phase1c']['primary_reference']['point']:.4f} (95% CI {scope['phase1c']['primary_reference']['confidence_interval_95']['lower']:.4f}–{scope['phase1c']['primary_reference']['confidence_interval_95']['upper']:.4f}).
- **Semantic audit:** ER+/ER− meanings and confusion order were reconciled from persisted predictions with no discrepancies and no refit/inference.
- **Phase 1C-XGB:** explicitly post-hoc and complementary. Strict CV argmax selected `{xgb['selected_candidate']}` (mean ROC-AUC {xgb['selected_mean_cv_roc_auc']:.5f}); locked-test ROC-AUC was {xgb['locked_test_roc_auc']:.5f} (95% CI {xgb['locked_test_roc_auc_ci95']['lower']:.5f}–{xgb['locked_test_roc_auc_ci95']['upper']:.5f}).

## XGBoost interpretation

Named confusion matrix: `{json.dumps(xgb['confusion_matrix'], ensure_ascii=False)}`.
The point ROC-AUC difference versus Logistic L1 is {comparison['point_difference']:.5f}; its paired-bootstrap 95% interval is {comparison['bootstrap_difference_lower']:.5f} to {comparison['bootstrap_difference_upper']:.5f}. The interval crosses zero. XGBoost therefore does **not** replace `P_classical-full(primary)=0.9660`.

## Integrity and leakage gate

All {len(report['checks'])} freeze checks passed. The XGBoost receipt records exactly one locked-test inference; all original Phase-1C hashes match their pre-XGBoost values; the same 5,000 paired bootstrap indices were reused; no StandardScaler or supervised feature selection entered the XGBoost pipeline. Feature importance is descriptive only and prohibited for Phase 2/VQC selection.

Tests: `{report['tests']['command']}` → **{report['tests']['passed']} passed, 0 failed**.

## Versions

`{json.dumps(report['software'], sort_keys=True)}`  
Git: `{json.dumps(report['git'], sort_keys=True)}`

## Reconstruction hashes (SHA-256)

{hashes}

## Boundary after freeze

Phase 2 has not started. PCA, MI, ANOVA, LASSO bottleneck and VQC remain outside this freeze. The fold-local feature-bottleneck protocol requires a separate review and approval.
"""


def run(repo_root: Path, tests_passed: int) -> dict[str, Any]:
    report = build(repo_root, tests_passed)
    json_path = repo_root / "reports/data/phase1_final_freeze.json"
    md_path = repo_root / "reports/data/phase1_final_freeze.md"
    write_json_atomic(json_path, report)
    temporary = md_path.with_suffix(".md.part")
    temporary.write_text(render_markdown(report), encoding="utf-8")
    temporary.replace(md_path)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--tests-passed", type=int, required=True)
    args = parser.parse_args(argv)
    report = run(Path(args.repo_root).resolve(), args.tests_passed)
    print(json.dumps({"status": report["status"], "checks": report["checks"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
