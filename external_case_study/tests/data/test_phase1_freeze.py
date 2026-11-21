from src.data.phase1_freeze import HASH_PATHS, render_markdown


def test_freeze_hash_manifest_contains_core_and_xgboost_artifacts():
    assert "data/processed/expression.parquet" in HASH_PATHS
    assert "data/processed/splits/phase1c_inner_cv.json" in HASH_PATHS
    assert "data/processed/xgboost_posthoc_v1/locked_test_predictions.parquet" in HASH_PATHS
    assert "reports/data/xgboost_posthoc_provenance.json" in HASH_PATHS


def test_render_freeze_keeps_primary_and_posthoc_roles_explicit():
    report = {
        "title": "FINAL FREEZE",
        "status": "FINAL_FREEZE_PASS",
        "generated_at": "now",
        "scope": {
            "phase1a": {"patients": 10, "distribution": {"ER_positive": 7, "ER_negative": 3}, "genes_raw": 20},
            "phase1b": {"specification": "fold local"},
            "phase1c": {"primary_reference": {"point": 0.9, "confidence_interval_95": {"lower": 0.8, "upper": 1.0}}},
            "phase1c_xgb": {
                "selected_candidate": "c1",
                "selected_mean_cv_roc_auc": 0.9,
                "locked_test_roc_auc": 0.91,
                "locked_test_roc_auc_ci95": {"lower": 0.8, "upper": 1.0},
                "confusion_matrix": {},
                "paired_roc_auc_vs_logistic_l1": {
                    "point_difference": 0.01,
                    "bootstrap_difference_lower": -0.01,
                    "bootstrap_difference_upper": 0.03,
                },
            },
        },
        "checks": {"ok": True},
        "tests": {"command": "pytest", "passed": 1},
        "software": {},
        "git": {},
        "artifact_sha256": {"a": "b"},
    }
    rendered = render_markdown(report)
    assert "primary reference remains Logistic L1" in rendered
    assert "post-hoc and complementary" in rendered
    assert "does **not** replace" in rendered
