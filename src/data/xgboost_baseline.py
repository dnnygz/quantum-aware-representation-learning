"""Execute the frozen post-hoc complementary XGBoost baseline.

The two-stage interface is intentional.  ``cv`` can read development patients
only.  ``final`` requires a frozen selection and records locked-test access
before the test counts are read, preventing an accidental second inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier, __version__ as xgboost_version, build_info

from .metric_semantics import explicit_er_metrics
from .preprocessing import _build_preprocessor, persist_fitted_preprocessor
from .utils import dependency_versions, git_state, sha256_file, utc_now, write_json_atomic
from .xgboost_protocol import (
    PROTECTED_PHASE1C_PATHS,
    balanced_sample_weights,
    select_candidate,
)


BOOTSTRAP_METRICS = (
    "roc_auc",
    "er_positive_pr_auc_average_precision",
    "er_negative_pr_auc_average_precision",
    "balanced_accuracy",
    "er_positive_sensitivity_recall",
    "er_negative_specificity_recall",
    "macro_f1",
    "accuracy",
)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_joblib(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    joblib.dump(value, temporary, compress=3)
    os.replace(temporary, path)


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _protected_hashes(repo_root: Path) -> dict[str, str]:
    return {path: sha256_file(repo_root / path) for path in PROTECTED_PHASE1C_PATHS}


def _assert_protected_unchanged(before: Mapping[str, str], repo_root: Path) -> None:
    after = _protected_hashes(repo_root)
    if dict(before) != after:
        changed = sorted(path for path in before if before[path] != after.get(path))
        raise RuntimeError(f"Frozen Phase 1C artifacts changed: {changed}")


def _load_counts(path: Path, patient_ids: Sequence[str]) -> pd.DataFrame:
    ids = list(map(str, patient_ids))
    frame = pd.read_parquet(path, filters=[("case_id", "in", ids)])
    frame.index = frame.index.astype(str)
    if set(frame.index) != set(ids) or len(frame) != len(ids):
        raise ValueError("Expression subset does not match requested patient IDs")
    return frame.loc[ids]


def _load_labels(path: Path) -> pd.Series:
    labels = pd.read_parquet(path)
    labels["case_id"] = labels["case_id"].astype(str)
    if labels["case_id"].duplicated().any():
        raise ValueError("Duplicate patient labels")
    result = labels.set_index("case_id")["er_label"].astype(int)
    if set(result.unique()) != {0, 1}:
        raise ValueError("Expected ER_negative=0 and ER_positive=1")
    return result


def build_model(protocol: Mapping[str, Any], candidate: Mapping[str, Any]) -> XGBClassifier:
    fixed = protocol["fixed_model_parameters"]
    params = {
        "objective": fixed["objective"],
        "booster": fixed["booster"],
        "tree_method": fixed["tree_method"],
        "device": fixed["device"],
        "grow_policy": fixed["grow_policy"],
        "max_bin": int(fixed["max_bin"]),
        "eval_metric": fixed["eval_metric"],
        "random_state": int(fixed["random_state"]),
        "n_jobs": int(fixed["n_jobs"]),
        "scale_pos_weight": float(fixed["scale_pos_weight"]),
        **{key: value for key, value in candidate.items() if key != "id"},
    }
    return XGBClassifier(**params)


def _preprocessor(data_config: Mapping[str, Any]):
    return _build_preprocessor(data_config["preprocessing"]).set_params(scale=False)


def scalar_metrics(y_true: np.ndarray, score: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    explicit = explicit_er_metrics(y_true, score, predicted, "positive_class_probability")
    return {
        "roc_auc": float(roc_auc_score(y_true, score)),
        **explicit,
        "accuracy": float(accuracy_score(y_true, predicted)),
    }


def _numeric_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    return {name: float(metrics[name]) for name in BOOTSTRAP_METRICS}


def _candidate_lookup(protocol: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["id"]): dict(item) for item in protocol["candidates"]}


def _context(protocol_path: Path, data_config_path: Path):
    protocol_path = protocol_path.resolve()
    repo_root = protocol_path.parent.parent
    protocol = _load_yaml(protocol_path)
    data_config = _load_yaml(data_config_path.resolve())
    if xgboost_version != protocol["software"]["version"]:
        raise RuntimeError(
            f"XGBoost version mismatch: {xgboost_version} != {protocol['software']['version']}"
        )
    inner_path = repo_root / protocol["inputs"]["inner_cv"]
    inner = json.loads(inner_path.read_text(encoding="utf-8"))
    return repo_root, protocol, data_config, inner


def run_cv(protocol_path: str | Path, data_config_path: str | Path) -> dict[str, Any]:
    repo_root, protocol, data_config, inner = _context(Path(protocol_path), Path(data_config_path))
    artifacts = repo_root / protocol["outputs"]["artifacts_dir"]
    processed = repo_root / protocol["outputs"]["processed_dir"]
    receipt = artifacts / "locked_test_access_receipt.json"
    if receipt.exists():
        raise FileExistsError("Locked-test access already recorded; CV rerun prohibited")
    gate = json.loads((repo_root / protocol["outputs"]["gate_report"]).read_text(encoding="utf-8"))
    if gate["status"] != "pass" or gate["protocol_sha256"] != sha256_file(Path(protocol_path)):
        raise RuntimeError("Current protocol does not match the passing protocol gate")
    artifacts.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    freeze_path = artifacts / "frozen_protocol.json"
    frozen = {
        "frozen_at": utc_now(),
        "protocol": protocol,
        "protocol_sha256": sha256_file(Path(protocol_path)),
        "gate_sha256": sha256_file(repo_root / protocol["outputs"]["gate_report"]),
    }
    if freeze_path.exists():
        existing = json.loads(freeze_path.read_text(encoding="utf-8"))
        if existing["protocol_sha256"] != frozen["protocol_sha256"]:
            raise RuntimeError("Existing frozen XGBoost protocol differs")
    else:
        write_json_atomic(freeze_path, frozen)

    protected_before = _protected_hashes(repo_root)
    locked = set(map(str, inner["locked_test_case_ids"]))
    development = list(map(str, inner["development_case_ids"]))
    if locked & set(development):
        raise AssertionError("Development and locked test overlap")
    counts = _load_counts(repo_root / protocol["inputs"]["raw_counts"], development)
    labels = _load_labels(repo_root / protocol["inputs"]["labels"])
    candidates = _candidate_lookup(protocol)
    results: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    started = time.perf_counter()

    for fold in inner["folds"]:
        fold_id = int(fold["fold"])
        train_ids = list(map(str, fold["train_case_ids"]))
        validation_ids = list(map(str, fold["validation_case_ids"]))
        if locked & (set(train_ids) | set(validation_ids)):
            raise AssertionError(f"Locked-test patient in fold {fold_id}")
        prep_started = time.perf_counter()
        prep = _preprocessor(data_config).fit(counts.loc[train_ids])
        X_train = prep.transform(counts.loc[train_ids]).astype(np.float32)
        X_validation = prep.transform(counts.loc[validation_ids]).astype(np.float32)
        prep_runtime = time.perf_counter() - prep_started
        y_train = labels.loc[train_ids].to_numpy(dtype=int)
        y_validation = labels.loc[validation_ids].to_numpy(dtype=int)
        filtered = prep.expression_filter_.transform(counts.loc[train_ids])
        persist_fitted_preprocessor(
            prep,
            artifacts / "cv_preprocessing" / f"fold_{fold_id}",
            prep.tmm_.diagnostics(filtered),
        )
        for candidate_id, candidate in candidates.items():
            model = build_model(protocol, candidate)
            train_started = time.perf_counter()
            model.fit(X_train, y_train, sample_weight=np.asarray(balanced_sample_weights(y_train)))
            train_runtime = time.perf_counter() - train_started
            inference_started = time.perf_counter()
            score = model.predict_proba(X_validation)[:, 1]
            inference_runtime = time.perf_counter() - inference_started
            predicted = (score >= float(protocol["evaluation"]["threshold"])).astype(int)
            metrics = scalar_metrics(y_validation, score, predicted)
            results.append(
                {
                    "candidate_id": candidate_id,
                    "fold": fold_id,
                    "roc_auc": metrics["roc_auc"],
                    "er_positive_ap": metrics["er_positive_pr_auc_average_precision"],
                    "er_negative_ap": metrics["er_negative_pr_auc_average_precision"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "er_positive_recall": metrics["er_positive_sensitivity_recall"],
                    "er_negative_recall_specificity": metrics["er_negative_specificity_recall"],
                    "macro_f1": metrics["macro_f1"],
                    "accuracy": metrics["accuracy"],
                    "n_features": int(X_train.shape[1]),
                    "preprocessing_runtime_seconds": float(prep_runtime),
                    "training_runtime_seconds": float(train_runtime),
                    "inference_runtime_seconds": float(inference_runtime),
                    "runtime_seconds": float(prep_runtime + train_runtime + inference_runtime),
                }
            )
            for patient_id, truth, value, classification in zip(
                validation_ids, y_validation, score, predicted
            ):
                predictions.append(
                    {
                        "patient_id": patient_id,
                        "y_true": int(truth),
                        "score": float(value),
                        "predicted_class": int(classification),
                        "model": "xgboost_posthoc",
                        "candidate_id": candidate_id,
                        "fold": fold_id,
                        "score_type": "positive_class_probability",
                    }
                )

    result_frame = pd.DataFrame(results)
    prediction_frame = pd.DataFrame(predictions)
    _atomic_csv(result_frame, repo_root / protocol["outputs"]["cv_results"])
    _atomic_parquet(result_frame, processed / "cv_results.parquet")
    _atomic_parquet(prediction_frame, processed / "cv_all_predictions.parquet")
    summary = (
        result_frame.groupby("candidate_id", sort=True)
        .agg(
            mean_roc_auc=("roc_auc", "mean"),
            sd_roc_auc=("roc_auc", "std"),
            mean_er_positive_ap=("er_positive_ap", "mean"),
            sd_er_positive_ap=("er_positive_ap", "std"),
            mean_er_negative_ap=("er_negative_ap", "mean"),
            sd_er_negative_ap=("er_negative_ap", "std"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            sd_balanced_accuracy=("balanced_accuracy", "std"),
            mean_macro_f1=("macro_f1", "mean"),
            sd_macro_f1=("macro_f1", "std"),
            total_training_seconds=("training_runtime_seconds", "sum"),
        )
        .reset_index()
    )
    _atomic_csv(summary, processed / "cv_summary.csv")
    means = dict(zip(summary["candidate_id"], summary["mean_roc_auc"]))
    selected_id = select_candidate(
        means, list(candidates.values()), float(protocol["selection"]["numerical_tie_tolerance_auc"])
    )
    selected = {
        "selected_at": utc_now(),
        "selected_candidate_id": selected_id,
        "selected_configuration": candidates[selected_id],
        "selection_source": "five_frozen_development_inner_folds_only",
        "selection_metric": "mean_inner_validation_ROC_AUC",
        "candidate_mean_roc_auc": {key: float(value) for key, value in means.items()},
        "tie_tolerance": float(protocol["selection"]["numerical_tie_tolerance_auc"]),
        "tie_break_applied": sum(abs(means[selected_id] - value) <= 1e-12 for value in means.values()) > 1,
        "protocol_sha256": sha256_file(Path(protocol_path)),
    }
    selected_path = repo_root / protocol["outputs"]["selected_configuration"]
    write_json_atomic(selected_path, selected)
    selected_oof = prediction_frame[prediction_frame["candidate_id"] == selected_id].copy()
    if len(selected_oof) != len(development) or selected_oof["patient_id"].nunique() != len(development):
        raise AssertionError("Selected OOF predictions do not cover development exactly once")
    _atomic_parquet(selected_oof, repo_root / protocol["outputs"]["oof_predictions"])
    _assert_protected_unchanged(protected_before, repo_root)
    complete = {
        "completed_at": utc_now(),
        "runtime_seconds": float(time.perf_counter() - started),
        "model_fits": len(results),
        "preprocessing_fits": len(inner["folds"]),
        "selected_configuration_sha256": sha256_file(selected_path),
        "selected_oof_sha256": sha256_file(repo_root / protocol["outputs"]["oof_predictions"]),
        "protected_phase1c_sha256": protected_before,
        "locked_test_expression_read": False,
        "locked_test_inference_count": 0,
    }
    write_json_atomic(artifacts / "cv_complete.json", complete)
    return {"selected": selected, "summary": summary.to_dict("records"), **complete}


def _bootstrap_values(predictions: pd.DataFrame, indices: np.ndarray) -> tuple[pd.DataFrame, dict[str, Any]]:
    y = predictions["y_true"].to_numpy(dtype=int)
    score = predictions["score"].to_numpy(dtype=float)
    predicted = predictions["predicted_class"].to_numpy(dtype=int)
    rows = []
    for bootstrap_id, positions in enumerate(indices):
        values = _numeric_metrics(scalar_metrics(y[positions], score[positions], predicted[positions]))
        rows.append({"bootstrap_id": bootstrap_id, **values})
    frame = pd.DataFrame(rows)
    intervals = {
        name: {
            "lower": float(frame[name].quantile(0.025)),
            "upper": float(frame[name].quantile(0.975)),
        }
        for name in BOOTSTRAP_METRICS
    }
    return frame, intervals


def _paired_comparisons(
    xgb_bootstrap: pd.DataFrame,
    original_predictions: pd.DataFrame,
    patient_order: Sequence[str],
    indices: np.ndarray,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for model_name, group in original_predictions.groupby("model", sort=True):
        aligned = group.set_index("patient_id").loc[list(patient_order)]
        model_rows = []
        y = aligned["y_true"].to_numpy(dtype=int)
        score = aligned["score"].to_numpy(dtype=float)
        predicted = aligned["predicted_class"].to_numpy(dtype=int)
        score_type = str(aligned["score_type"].iloc[0])
        for bootstrap_id, positions in enumerate(indices):
            values = scalar_metrics(y[positions], score[positions], predicted[positions])
            if score_type == "decision_function":
                values = {
                    "roc_auc": float(roc_auc_score(y[positions], score[positions])),
                    **explicit_er_metrics(
                        y[positions], score[positions], predicted[positions], score_type
                    ),
                    "accuracy": float(accuracy_score(y[positions], predicted[positions])),
                }
            numeric = _numeric_metrics(values)
            model_rows.append(numeric)
            for metric in BOOTSTRAP_METRICS:
                rows.append(
                    {
                        "model": model_name,
                        "bootstrap_id": bootstrap_id,
                        "metric": metric,
                        "xgboost_minus_model": float(
                            xgb_bootstrap.loc[bootstrap_id, metric] - numeric[metric]
                        ),
                    }
                )
        model_frame = pd.DataFrame(model_rows)
        for metric in BOOTSTRAP_METRICS:
            differences = xgb_bootstrap[metric].to_numpy() - model_frame[metric].to_numpy()
            summaries.append(
                {
                    "comparison": f"xgboost_posthoc_minus_{model_name}",
                    "metric": metric,
                    "bootstrap_difference_lower": float(np.quantile(differences, 0.025)),
                    "bootstrap_difference_upper": float(np.quantile(differences, 0.975)),
                    "proportion_difference_gt_zero_descriptive": float(np.mean(differences > 0)),
                }
            )
    return pd.DataFrame(rows), summaries


def _feature_importance(model: XGBClassifier, feature_names: Sequence[str]) -> pd.DataFrame:
    booster = model.get_booster()
    result = pd.DataFrame({"gene_id": list(feature_names), "feature_order": range(len(feature_names))})
    for importance_type in ("gain", "cover", "weight"):
        mapping = booster.get_score(importance_type=importance_type)
        result[importance_type] = result["gene_id"].map(mapping).fillna(0.0).astype(float)
    result["descriptive_only"] = True
    result["prohibited_for_phase2_vqc_selection"] = True
    return result


def run_final(protocol_path: str | Path, data_config_path: str | Path) -> dict[str, Any]:
    repo_root, protocol, data_config, inner = _context(Path(protocol_path), Path(data_config_path))
    artifacts = repo_root / protocol["outputs"]["artifacts_dir"]
    processed = repo_root / protocol["outputs"]["processed_dir"]
    receipt = artifacts / "locked_test_access_receipt.json"
    if receipt.exists():
        raise FileExistsError("XGBoost locked test was already accessed; second inference prohibited")
    cv_complete = json.loads((artifacts / "cv_complete.json").read_text(encoding="utf-8"))
    selected_path = repo_root / protocol["outputs"]["selected_configuration"]
    if sha256_file(selected_path) != cv_complete["selected_configuration_sha256"]:
        raise RuntimeError("Selected configuration changed after CV")
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    if selected["protocol_sha256"] != sha256_file(Path(protocol_path)):
        raise RuntimeError("Selected configuration belongs to another protocol")
    protected_before = _protected_hashes(repo_root)
    development = list(map(str, inner["development_case_ids"]))
    locked_test = list(map(str, inner["locked_test_case_ids"]))
    counts_path = repo_root / protocol["inputs"]["raw_counts"]
    labels = _load_labels(repo_root / protocol["inputs"]["labels"])
    development_counts = _load_counts(counts_path, development)
    started = time.perf_counter()
    prep_started = time.perf_counter()
    prep = _preprocessor(data_config).fit(development_counts)
    X_development = prep.transform(development_counts).astype(np.float32)
    prep_runtime = time.perf_counter() - prep_started
    y_development = labels.loc[development].to_numpy(dtype=int)
    filtered = prep.expression_filter_.transform(development_counts)
    persist_fitted_preprocessor(
        prep, artifacts / "final_preprocessing", prep.tmm_.diagnostics(filtered)
    )
    _atomic_joblib(prep, artifacts / "final_preprocessor.joblib")
    model = build_model(protocol, selected["selected_configuration"])
    train_started = time.perf_counter()
    model.fit(
        X_development,
        y_development,
        sample_weight=np.asarray(balanced_sample_weights(y_development)),
    )
    train_runtime = time.perf_counter() - train_started
    model_path = artifacts / "xgboost_model.json"
    temporary_model = model_path.with_name(model_path.stem + ".part.json")
    model.save_model(temporary_model)
    os.replace(temporary_model, model_path)

    patient_order_frame = pd.read_parquet(repo_root / protocol["inputs"]["bootstrap_patient_order"])
    patient_order = patient_order_frame.sort_values("position")["patient_id"].astype(str).tolist()
    if set(patient_order) != set(locked_test) or len(patient_order) != len(locked_test):
        raise AssertionError("Frozen bootstrap patient order differs from locked test")
    write_json_atomic(
        receipt,
        {
            "access_started_at": utc_now(),
            "purpose": "single_locked_test_inference_xgboost_posthoc_v1",
            "selected_configuration_sha256": sha256_file(selected_path),
            "protocol_sha256": sha256_file(Path(protocol_path)),
            "planned_inference_calls": 1,
            "status": "access_started_before_test_counts_read",
        },
    )
    test_counts = _load_counts(counts_path, patient_order)
    transform_started = time.perf_counter()
    X_test = prep.transform(test_counts).astype(np.float32)
    test_preprocessing_runtime = time.perf_counter() - transform_started
    inference_started = time.perf_counter()
    score = model.predict_proba(X_test)[:, 1]  # The sole locked-test inference call.
    inference_runtime = time.perf_counter() - inference_started
    predicted = (score >= float(protocol["evaluation"]["threshold"])).astype(int)
    y_test = labels.loc[patient_order].to_numpy(dtype=int)
    test_predictions = pd.DataFrame(
        {
            "patient_id": patient_order,
            "y_true": y_test,
            "score": score,
            "predicted_class": predicted,
            "model": "xgboost_posthoc",
            "candidate_id": selected["selected_candidate_id"],
            "fold": "locked_test",
            "score_type": "positive_class_probability",
        }
    )
    prediction_path = repo_root / protocol["outputs"]["locked_test_predictions"]
    _atomic_parquet(test_predictions, prediction_path)

    indices_file = np.load(repo_root / protocol["inputs"]["bootstrap_indices"])
    bootstrap_indices = indices_file["indices"]
    if bootstrap_indices.shape != (5000, len(patient_order)):
        raise AssertionError("Unexpected frozen bootstrap shape")
    metrics = scalar_metrics(y_test, score, predicted)
    bootstrap_frame, intervals = _bootstrap_values(test_predictions, bootstrap_indices)
    _atomic_parquet(bootstrap_frame, processed / "bootstrap_metrics.parquet")
    originals = pd.read_parquet(
        repo_root / "data/processed/phase1c_classical_baseline_v1/locked_test_predictions.parquet"
    )
    paired_frame, paired_summary = _paired_comparisons(
        bootstrap_frame, originals, patient_order, bootstrap_indices
    )
    _atomic_parquet(paired_frame, processed / "paired_bootstrap_differences.parquet")

    importance = _feature_importance(model, X_development.columns.astype(str))
    importance_path = repo_root / protocol["outputs"]["feature_importance"]
    _atomic_parquet(importance, importance_path)
    final_runtime = time.perf_counter() - started
    performance = {
        "generated_at": utc_now(),
        "role": protocol["role"],
        "must_not_replace_primary_reference": True,
        "primary_reference": protocol["primary_reference"],
        "selected_candidate_id": selected["selected_candidate_id"],
        "threshold": 0.5,
        "target_semantics": protocol["target"],
        "point_metrics": metrics,
        "bootstrap_95_percentile_intervals": intervals,
        "bootstrap_method": {
            "mode": "same_frozen_patient_level_stratified_indices_as_phase1c",
            "n_resamples": int(len(bootstrap_indices)),
            "indices_sha256": sha256_file(repo_root / protocol["inputs"]["bootstrap_indices"]),
            "patient_order_sha256": sha256_file(
                repo_root / protocol["inputs"]["bootstrap_patient_order"]
            ),
        },
        "paired_bootstrap_comparisons": paired_summary,
        "runtime_seconds": {
            "development_preprocessing_fit_transform": float(prep_runtime),
            "final_model_training": float(train_runtime),
            "locked_test_transform": float(test_preprocessing_runtime),
            "locked_test_inference": float(inference_runtime),
            "total_final_stage": float(final_runtime),
        },
    }
    performance_path = repo_root / protocol["outputs"]["locked_test_performance"]
    write_json_atomic(performance_path, performance)
    _assert_protected_unchanged(protected_before, repo_root)
    receipt_value = json.loads(receipt.read_text(encoding="utf-8"))
    receipt_value.update(
        {
            "completed_at": utc_now(),
            "status": "complete",
            "actual_inference_calls": 1,
            "prediction_sha256": sha256_file(prediction_path),
            "n_locked_test_patients": len(test_predictions),
        }
    )
    write_json_atomic(receipt, receipt_value)
    qc = {
        "generated_at": utc_now(),
        "status": "pass",
        "checks": {
            "development_locked_test_disjoint": not bool(set(development) & set(locked_test)),
            "cv_locked_test_not_read": True,
            "selection_precedes_locked_test_receipt": True,
            "locked_test_inference_calls": 1,
            "preprocessor_scale_is_false": prep.scaler_ is None,
            "all_test_patients_predicted_once": test_predictions["patient_id"].nunique()
            == len(locked_test)
            == len(test_predictions),
            "finite_scores": bool(np.isfinite(score).all()),
            "finite_transformed_values": bool(np.isfinite(X_test.to_numpy()).all()),
            "feature_order_equal": list(X_development.columns) == list(X_test.columns),
            "bootstrap_indices_reused": True,
            "protected_phase1c_artifacts_unchanged": True,
            "feature_importance_descriptive_only": True,
        },
        "dimensions": {
            "development_patients": len(development),
            "locked_test_patients": len(locked_test),
            "genes_initial": int(prep.n_features_in_),
            "genes_after_expression_filter": int(
                len(prep.expression_filter_.feature_names_out_)
            ),
            "genes_removed_zero_variance": int(
                len(prep.expression_filter_.feature_names_out_) - len(prep.feature_names_out_)
            ),
            "genes_final": int(len(prep.feature_names_out_)),
        },
        "protected_phase1c_sha256": protected_before,
    }
    write_json_atomic(repo_root / protocol["outputs"]["leakage_qc"], qc)
    generated_paths = [
        repo_root / protocol["outputs"]["cv_results"],
        selected_path,
        repo_root / protocol["outputs"]["oof_predictions"],
        prediction_path,
        performance_path,
        importance_path,
        processed / "cv_results.parquet",
        processed / "cv_summary.csv",
        processed / "cv_all_predictions.parquet",
        processed / "bootstrap_metrics.parquet",
        processed / "paired_bootstrap_differences.parquet",
        model_path,
        artifacts / "final_preprocessor.joblib",
        receipt,
    ]
    provenance = {
        "generated_at": utc_now(),
        "role": protocol["role"],
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": sha256_file(Path(protocol_path)),
        "gate_sha256": sha256_file(repo_root / protocol["outputs"]["gate_report"]),
        "selected_configuration_sha256": sha256_file(selected_path),
        "canonical_inputs_sha256": {
            path: sha256_file(repo_root / path)
            for path in (
                protocol["inputs"]["raw_counts"],
                protocol["inputs"]["labels"],
                protocol["inputs"]["canonical_split"],
                protocol["inputs"]["inner_cv"],
                protocol["inputs"]["bootstrap_indices"],
                protocol["inputs"]["bootstrap_patient_order"],
            )
        },
        "protected_phase1c_sha256": protected_before,
        "software": {
            **dependency_versions(["numpy", "pandas", "pyarrow", "scikit-learn", "xgboost"]),
            "platform": platform.platform(),
            "xgboost_build_info": build_info(),
        },
        "git": git_state(repo_root),
        "generated_artifact_sha256": {
            str(path.relative_to(repo_root)): sha256_file(path) for path in generated_paths
        },
        "locked_test_inference_count": 1,
    }
    write_json_atomic(repo_root / protocol["outputs"]["provenance"], provenance)
    return {"performance": performance, "qc": qc, "provenance": provenance}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("cv", "final"), required=True)
    parser.add_argument("--protocol", default="configs/xgboost_posthoc.yaml")
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args(argv)
    result = (
        run_cv(args.protocol, args.config)
        if args.stage == "cv"
        else run_final(args.protocol, args.config)
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
