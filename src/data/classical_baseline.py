"""Execute the approved leakage-safe Phase 1C classical baseline.

The command has two explicit stages. ``cv`` performs development-only model
selection and cannot access locked-test expression. ``final`` requires the
frozen CV selection artifact, refits on development, and evaluates each final
model once on locked test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.svm import LinearSVC, SVC

from .phase1c_protocol import assert_locked_test_isolation
from .preprocessing import (
    RNASeqPreprocessor,
    _build_preprocessor,
    persist_fitted_preprocessor,
)
from .utils import (
    canonical_json_bytes,
    configure_logging,
    dependency_versions,
    git_state,
    load_config,
    resolve_path,
    sha256_file,
    utc_now,
    write_json_atomic,
)


LOGGER = logging.getLogger(__name__)
MODEL_ORDER = ("logistic_l1", "logistic_l2", "linear_svm", "rbf_svm")
SCALAR_METRICS = (
    "roc_auc",
    "pr_auc",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "f1",
    "accuracy",
)


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _write_joblib_atomic(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    joblib.dump(value, temporary, compress=3)
    os.replace(temporary, path)


def _write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def classification_metrics(y_true: np.ndarray, score: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    """Compute all approved metrics with ER-positive encoded as class 1."""

    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    predicted = np.asarray(predicted, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, score)),
        "pr_auc": float(average_precision_score(y_true, score)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "sensitivity": float(recall_score(y_true, predicted, pos_label=1, zero_division=0)),
        "specificity": float(tn / (tn + fp)),
        "f1": float(f1_score(y_true, predicted, pos_label=1, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def build_estimator(model_name: str, model_config: Mapping[str, Any], c_value: float) -> Any:
    params = dict(model_config["fixed_parameters"])
    params["C"] = float(c_value)
    if model_name in {"logistic_l1", "logistic_l2"}:
        return LogisticRegression(**params)
    if model_name == "linear_svm":
        return LinearSVC(**params)
    if model_name == "rbf_svm":
        return SVC(**params)
    raise ValueError(f"Unsupported Phase 1C model: {model_name}")


def model_score_and_prediction(model_name: str, estimator: Any, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, str]:
    if model_name.startswith("logistic_"):
        score = estimator.predict_proba(X)[:, 1]
        predicted = (score >= 0.5).astype(int)
        return score, predicted, "positive_class_probability"
    score = np.asarray(estimator.decision_function(X), dtype=float)
    predicted = (score >= 0.0).astype(int)
    return score, predicted, "decision_function"


def fit_with_diagnostics(estimator: Any, X: pd.DataFrame, y: np.ndarray) -> tuple[Any, float, list[str]]:
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(X, y)
    runtime = float(time.perf_counter() - started)
    convergence = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
    return estimator, runtime, convergence


def _n_iter(estimator: Any) -> int | None:
    value = getattr(estimator, "n_iter_", None)
    if value is None:
        return None
    return int(np.max(np.asarray(value)))


def _l1_sparsity(estimator: Any) -> dict[str, Any]:
    coefficients = np.asarray(estimator.coef_)
    nonzero = int(np.count_nonzero(coefficients))
    total = int(coefficients.size)
    return {
        "nonzero_coefficients": nonzero,
        "zero_coefficients": total - nonzero,
        "total_coefficients": total,
        "nonzero_fraction": float(nonzero / total),
    }


def select_hyperparameters(cv_results: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply frozen max-mean-ROC-AUC and deterministic smallest-C tie break."""

    metric_columns = list(SCALAR_METRICS)
    grouped = cv_results.groupby(["model", "C"], sort=True)
    rows: list[dict[str, Any]] = []
    for (model_name, c_value), group in grouped:
        row: dict[str, Any] = {
            "model": str(model_name),
            "C": float(c_value),
            "folds": int(len(group)),
            "n_features_min": int(group["n_features"].min()),
            "n_features_max": int(group["n_features"].max()),
            "preprocessing_runtime_seconds_mean": float(group["preprocessing_runtime_seconds"].mean()),
            "training_runtime_seconds_mean": float(group["training_runtime_seconds"].mean()),
            "inference_runtime_seconds_mean": float(group["inference_runtime_seconds"].mean()),
        }
        for metric in metric_columns:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1))
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(["model", "C"]).reset_index(drop=True)

    selected: dict[str, Any] = {}
    for model_name in sorted(summary["model"].unique()):
        candidates = summary.loc[summary["model"] == model_name].copy()
        best_auc = float(candidates["roc_auc_mean"].max())
        best = candidates.loc[np.isclose(candidates["roc_auc_mean"], best_auc, rtol=0, atol=1e-12)]
        best = best.sort_values(["C"], ascending=True).iloc[0]
        selected[model_name] = {
            "C": float(best["C"]),
            "mean_cv_roc_auc": float(best["roc_auc_mean"]),
            "sd_cv_roc_auc": float(best["roc_auc_sd"]),
            "selection_metric": "mean_validation_roc_auc",
            "tie_breaker": "smallest_C_then_lexicographic_parameter_order",
        }
    reference_model = sorted(
        selected,
        key=lambda name: (-selected[name]["mean_cv_roc_auc"], name),
    )[0]
    selection = {
        "models": selected,
        "classical_reference_model": reference_model,
        "classical_reference_selection": "highest selected-model mean inner-CV ROC-AUC; lexicographic model tie-break",
    }
    return summary, selection


def stratified_bootstrap_indices(y_true: np.ndarray, n_resamples: int, seed: int) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=int)
    negative = np.flatnonzero(y_true == 0)
    positive = np.flatnonzero(y_true == 1)
    if not len(negative) or not len(positive):
        raise ValueError("Stratified bootstrap requires both classes")
    rng = np.random.default_rng(int(seed))
    result = np.empty((int(n_resamples), len(y_true)), dtype=np.int32)
    for position in range(int(n_resamples)):
        negative_draw = rng.choice(negative, size=len(negative), replace=True)
        positive_draw = rng.choice(positive, size=len(positive), replace=True)
        result[position] = np.concatenate((negative_draw, positive_draw))
    return result


def bootstrap_metrics(
    predictions: pd.DataFrame,
    indices: np.ndarray,
    confidence_level: float,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    y_true = predictions["y_true"].to_numpy(dtype=int)
    score = predictions["score"].to_numpy(dtype=float)
    predicted = predictions["predicted_class"].to_numpy(dtype=int)
    rows: list[dict[str, Any]] = []
    values = {metric: np.empty(len(indices), dtype=float) for metric in SCALAR_METRICS}
    for bootstrap_id, sample_indices in enumerate(indices):
        metrics = classification_metrics(
            y_true[sample_indices], score[sample_indices], predicted[sample_indices]
        )
        row: dict[str, Any] = {"bootstrap_id": int(bootstrap_id)}
        for metric in SCALAR_METRICS:
            values[metric][bootstrap_id] = metrics[metric]
            row[metric] = metrics[metric]
        rows.append(row)
    alpha = (1.0 - float(confidence_level)) / 2.0
    intervals = {
        metric: {
            "lower": float(np.quantile(metric_values, alpha)),
            "upper": float(np.quantile(metric_values, 1.0 - alpha)),
        }
        for metric, metric_values in values.items()
    }
    return pd.DataFrame(rows), intervals


def _canonical_paths(repo_root: Path, config: Mapping[str, Any]) -> list[Path]:
    processed = resolve_path(repo_root, config["paths"]["processed_dir"])
    return [
        processed / "expression.parquet",
        processed / "genes.parquet",
        processed / "labels.parquet",
        processed / "metadata.parquet",
        processed / "splits.json",
        resolve_path(repo_root, config["phase1c"]["inner_cv"]["artifact"]),
    ]


def _canonical_hashes(repo_root: Path, config: Mapping[str, Any]) -> dict[str, str]:
    paths = _canonical_paths(repo_root, config)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required Phase 1C input: {path}")
    return {str(path.relative_to(repo_root)): sha256_file(path) for path in paths}


def _protocol_specification(phase_config: Mapping[str, Any], preprocessing: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "protocol_version": phase_config["protocol_version"],
        "status": phase_config["status"],
        "preprocessing_specification_id": preprocessing["specification_id"],
        "selection": phase_config["selection"],
        "class_imbalance": phase_config["class_imbalance"],
        "classification_threshold": phase_config["classification_threshold"],
        "models": phase_config["models"],
        "metrics": phase_config["metrics"],
        "bootstrap": phase_config["bootstrap"],
        "predictions": phase_config["predictions"],
    }


def _specification_hash(specification: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(specification))).hexdigest()


def _load_inputs(config_path: str | Path) -> tuple[dict[str, Any], Path, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    config, repo_root = load_config(config_path)
    processed = resolve_path(repo_root, config["paths"]["processed_dir"])
    expression = pd.read_parquet(processed / "expression.parquet")
    expression.index = expression.index.astype(str)
    labels = pd.read_parquet(processed / "labels.parquet")
    labels["case_id"] = labels["case_id"].astype(str)
    labels = labels.set_index("case_id").loc[expression.index].reset_index()
    inner = json.loads(
        resolve_path(repo_root, config["phase1c"]["inner_cv"]["artifact"]).read_text(
            encoding="utf-8"
        )
    )
    return config, repo_root, expression, labels, inner


def _preprocessor(config: Mapping[str, Any]) -> RNASeqPreprocessor:
    return _build_preprocessor(config["preprocessing"]).set_params(scale=True)


def run_cv_stage(config_path: str | Path) -> dict[str, Any]:
    """Run all frozen configurations using development-only inner CV."""

    config, repo_root, expression, labels, inner = _load_inputs(config_path)
    phase_config = config["phase1c"]
    outputs = phase_config["outputs"]
    artifacts_dir = resolve_path(repo_root, outputs["artifacts_dir"])
    processed_output = resolve_path(repo_root, outputs["processed_dir"])
    final_receipt = artifacts_dir / "locked_test_access_receipt.json"
    if final_receipt.exists():
        raise FileExistsError("Locked test was already evaluated; CV rerun is prohibited")
    canonical_before = _canonical_hashes(repo_root, config)
    specification = _protocol_specification(phase_config, config["preprocessing"])
    specification_hash = _specification_hash(specification)
    freeze_path = artifacts_dir / "frozen_protocol.json"
    if freeze_path.exists():
        existing = json.loads(freeze_path.read_text(encoding="utf-8"))
        if existing["specification_sha256"] != specification_hash:
            raise ValueError("Frozen Phase 1C specification differs from current configuration")
    else:
        write_json_atomic(
            freeze_path,
            {
                "approved_at": "2026-10-08",
                "specification": specification,
                "specification_sha256": specification_hash,
            },
        )

    by_id = labels.set_index("case_id")["er_label"].astype(int)
    locked_test_ids = inner["locked_test_case_ids"]
    all_results: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    sparsity_rows: list[dict[str, Any]] = []
    stage_started = time.perf_counter()

    for fold in inner["folds"]:
        fold_number = int(fold["fold"])
        train_ids = list(map(str, fold["train_case_ids"]))
        validation_ids = list(map(str, fold["validation_case_ids"]))
        assert_locked_test_isolation(train_ids, locked_test_ids, stage=f"fold_{fold_number}_preprocessing_fit")
        assert_locked_test_isolation(validation_ids, locked_test_ids, stage=f"fold_{fold_number}_model_selection")
        LOGGER.info("Phase 1C CV fold %s: fitting preprocessing", fold_number)
        preprocessing_started = time.perf_counter()
        preprocessor = _preprocessor(config).fit(expression.loc[train_ids])
        X_train = preprocessor.transform(expression.loc[train_ids])
        X_validation = preprocessor.transform(expression.loc[validation_ids])
        preprocessing_runtime = float(time.perf_counter() - preprocessing_started)
        y_train = by_id.loc[train_ids].to_numpy(dtype=int)
        y_validation = by_id.loc[validation_ids].to_numpy(dtype=int)
        filtered_train = preprocessor.expression_filter_.transform(expression.loc[train_ids])
        train_tmm = preprocessor.tmm_.diagnostics(filtered_train)
        persist_fitted_preprocessor(
            preprocessor,
            artifacts_dir / "cv_preprocessing" / f"fold_{fold_number}",
            train_tmm,
        )

        for model_name in MODEL_ORDER:
            model_config = phase_config["models"][model_name]
            for c_value in model_config["search_space"]["C"]:
                estimator = build_estimator(model_name, model_config, float(c_value))
                estimator, training_runtime, convergence = fit_with_diagnostics(
                    estimator, X_train, y_train
                )
                inference_started = time.perf_counter()
                score, predicted, score_type = model_score_and_prediction(
                    model_name, estimator, X_validation
                )
                inference_runtime = float(time.perf_counter() - inference_started)
                metrics = classification_metrics(y_validation, score, predicted)
                result = {
                    "model": model_name,
                    "C": float(c_value),
                    "fold": fold_number,
                    **{metric: metrics[metric] for metric in SCALAR_METRICS},
                    "tn": metrics["confusion_matrix"]["tn"],
                    "fp": metrics["confusion_matrix"]["fp"],
                    "fn": metrics["confusion_matrix"]["fn"],
                    "tp": metrics["confusion_matrix"]["tp"],
                    "n_features": int(X_train.shape[1]),
                    "preprocessing_runtime_seconds": preprocessing_runtime,
                    "training_runtime_seconds": training_runtime,
                    "inference_runtime_seconds": inference_runtime,
                    "runtime_seconds": preprocessing_runtime + training_runtime + inference_runtime,
                    "n_iter": _n_iter(estimator),
                    "convergence_warning_count": len(convergence),
                    "convergence_warnings": " | ".join(convergence),
                }
                all_results.append(result)
                for patient_id, truth, patient_score, patient_prediction in zip(
                    validation_ids, y_validation, score, predicted
                ):
                    all_predictions.append(
                        {
                            "patient_id": patient_id,
                            "y_true": int(truth),
                            "score": float(patient_score),
                            "predicted_class": int(patient_prediction),
                            "model": model_name,
                            "C": float(c_value),
                            "fold": fold_number,
                            "score_type": score_type,
                        }
                    )
                if model_name == "logistic_l1":
                    sparsity_rows.append(
                        {
                            "stage": "inner_cv",
                            "fold": fold_number,
                            "C": float(c_value),
                            **_l1_sparsity(estimator),
                        }
                    )
        LOGGER.info("Phase 1C CV fold %s complete", fold_number)

    cv_results = pd.DataFrame(all_results).sort_values(["model", "C", "fold"])
    cv_predictions_all = pd.DataFrame(all_predictions).sort_values(
        ["model", "C", "fold", "patient_id"]
    )
    cv_summary, selection = select_hyperparameters(cv_results)
    selection.update(
        {
            "selected_at": utc_now(),
            "selection_data": "development_inner_cv_only",
            "locked_test_accessed": False,
            "frozen_protocol_sha256": specification_hash,
            "inner_cv_sha256": sha256_file(
                resolve_path(repo_root, phase_config["inner_cv"]["artifact"])
            ),
        }
    )

    oof_parts = []
    for model_name, parameters in selection["models"].items():
        selected_rows = cv_predictions_all.loc[
            (cv_predictions_all["model"] == model_name)
            & np.isclose(cv_predictions_all["C"], parameters["C"], rtol=0, atol=0)
        ].copy()
        if len(selected_rows) != len(inner["development_case_ids"]):
            raise RuntimeError(f"Incomplete OOF predictions for {model_name}")
        if selected_rows["patient_id"].duplicated().any():
            raise RuntimeError(f"Duplicate OOF predictions for {model_name}")
        assert_locked_test_isolation(
            selected_rows["patient_id"], locked_test_ids, stage=f"{model_name}_oof"
        )
        oof_parts.append(selected_rows)
    oof_predictions = pd.concat(oof_parts, ignore_index=True).sort_values(
        ["model", "patient_id"]
    )

    cv_results_path = resolve_path(repo_root, outputs["cv_results"])
    cv_summary_path = resolve_path(repo_root, outputs["cv_summary"])
    selection_path = resolve_path(repo_root, outputs["selected_hyperparameters"])
    _write_csv_atomic(cv_results, cv_results_path)
    _write_csv_atomic(cv_summary, cv_summary_path)
    _write_parquet_atomic(cv_results, processed_output / "cv_results.parquet")
    _write_parquet_atomic(cv_predictions_all, processed_output / "cv_all_predictions.parquet")
    _write_parquet_atomic(oof_predictions, processed_output / "oof_predictions.parquet")
    _write_parquet_atomic(pd.DataFrame(sparsity_rows), processed_output / "l1_sparsity.parquet")
    write_json_atomic(selection_path, selection)

    canonical_after = _canonical_hashes(repo_root, config)
    if canonical_before != canonical_after:
        raise RuntimeError("Canonical artifacts changed during Phase 1C CV")
    stage = {
        "completed_at": utc_now(),
        "stage": "development_inner_cv_model_selection",
        "locked_test_accessed": False,
        "runtime_seconds": float(time.perf_counter() - stage_started),
        "frozen_protocol_sha256": specification_hash,
        "selected_hyperparameters_sha256": sha256_file(selection_path),
        "artifacts": {
            str(path.relative_to(repo_root)): sha256_file(path)
            for path in [
                cv_results_path,
                cv_summary_path,
                selection_path,
                processed_output / "cv_results.parquet",
                processed_output / "cv_all_predictions.parquet",
                processed_output / "oof_predictions.parquet",
                processed_output / "l1_sparsity.parquet",
            ]
        },
        "canonical_sha256_before": canonical_before,
        "canonical_sha256_after": canonical_after,
        "convergence_warning_total": int(cv_results["convergence_warning_count"].sum()),
    }
    write_json_atomic(artifacts_dir / "cv_complete.json", stage)
    return stage


def run_final_stage(config_path: str | Path) -> dict[str, Any]:
    """Refit selected pipelines on development and evaluate locked test once."""

    config, repo_root, expression, labels, inner = _load_inputs(config_path)
    phase_config = config["phase1c"]
    outputs = phase_config["outputs"]
    artifacts_dir = resolve_path(repo_root, outputs["artifacts_dir"])
    processed_output = resolve_path(repo_root, outputs["processed_dir"])
    receipt_path = artifacts_dir / "locked_test_access_receipt.json"
    if receipt_path.exists():
        raise FileExistsError("Locked test has already been evaluated for Phase 1C")
    cv_complete_path = artifacts_dir / "cv_complete.json"
    selection_path = resolve_path(repo_root, outputs["selected_hyperparameters"])
    if not cv_complete_path.is_file() or not selection_path.is_file():
        raise FileNotFoundError("Development-only CV stage must complete before final evaluation")
    cv_complete = json.loads(cv_complete_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if sha256_file(selection_path) != cv_complete["selected_hyperparameters_sha256"]:
        raise ValueError("Selected hyperparameters changed after CV completion")
    specification = _protocol_specification(phase_config, config["preprocessing"])
    specification_hash = _specification_hash(specification)
    if specification_hash != selection["frozen_protocol_sha256"]:
        raise ValueError("Frozen Phase 1C protocol changed after model selection")

    canonical_before = _canonical_hashes(repo_root, config)
    development_ids = list(map(str, inner["development_case_ids"]))
    locked_test_ids = list(map(str, inner["locked_test_case_ids"]))
    assert_locked_test_isolation(development_ids, locked_test_ids, stage="final_preprocessing_fit")
    by_id = labels.set_index("case_id")["er_label"].astype(int)
    y_development = by_id.loc[development_ids].to_numpy(dtype=int)
    stage_started = time.perf_counter()

    preprocessing_started = time.perf_counter()
    preprocessor = _preprocessor(config).fit(expression.loc[development_ids])
    X_development = preprocessor.transform(expression.loc[development_ids])
    preprocessing_fit_runtime = float(time.perf_counter() - preprocessing_started)
    filtered_development = preprocessor.expression_filter_.transform(expression.loc[development_ids])
    development_tmm = preprocessor.tmm_.diagnostics(filtered_development)
    preprocessing_hashes = persist_fitted_preprocessor(
        preprocessor,
        artifacts_dir / "final_preprocessing",
        development_tmm,
    )
    _write_joblib_atomic(preprocessor, artifacts_dir / "final_preprocessor.joblib")

    fitted_models: dict[str, Any] = {}
    fit_diagnostics: dict[str, Any] = {}
    final_sparsity: dict[str, Any] | None = None
    for model_name in MODEL_ORDER:
        selected_c = float(selection["models"][model_name]["C"])
        estimator = build_estimator(model_name, phase_config["models"][model_name], selected_c)
        estimator, training_runtime, convergence = fit_with_diagnostics(
            estimator, X_development, y_development
        )
        fitted_models[model_name] = estimator
        fit_diagnostics[model_name] = {
            "C": selected_c,
            "training_runtime_seconds": training_runtime,
            "n_iter": _n_iter(estimator),
            "convergence_warnings": convergence,
        }
        if model_name == "logistic_l1":
            final_sparsity = {
                "stage": "full_development_refit",
                "fold": "full_development",
                "C": selected_c,
                **_l1_sparsity(estimator),
            }
        _write_joblib_atomic(estimator, artifacts_dir / "models" / f"{model_name}.joblib")

    # This is the single transition at which locked-test expression and labels
    # become available. All model families and C values are already frozen.
    test_access_started = utc_now()
    test_preprocessing_started = time.perf_counter()
    X_test = preprocessor.transform(expression.loc[locked_test_ids])
    test_preprocessing_runtime = float(time.perf_counter() - test_preprocessing_started)
    y_test = by_id.loc[locked_test_ids].to_numpy(dtype=int)

    prediction_parts: list[pd.DataFrame] = []
    point_results: dict[str, Any] = {}
    for model_name in MODEL_ORDER:
        inference_started = time.perf_counter()
        score, predicted, score_type = model_score_and_prediction(
            model_name, fitted_models[model_name], X_test
        )
        inference_runtime = float(time.perf_counter() - inference_started)
        metrics = classification_metrics(y_test, score, predicted)
        point_results[model_name] = {
            "C": float(selection["models"][model_name]["C"]),
            "metrics": metrics,
            "n_features": int(X_test.shape[1]),
            "preprocessing_fit_runtime_seconds": preprocessing_fit_runtime,
            "test_transform_runtime_seconds": test_preprocessing_runtime,
            "training_runtime_seconds": fit_diagnostics[model_name]["training_runtime_seconds"],
            "inference_runtime_seconds": inference_runtime,
            "n_iter": fit_diagnostics[model_name]["n_iter"],
            "convergence_warnings": fit_diagnostics[model_name]["convergence_warnings"],
        }
        prediction_parts.append(
            pd.DataFrame(
                {
                    "patient_id": locked_test_ids,
                    "y_true": y_test,
                    "score": score,
                    "predicted_class": predicted,
                    "model": model_name,
                    "C": float(selection["models"][model_name]["C"]),
                    "fold": "locked_test",
                    "score_type": score_type,
                }
            )
        )
    test_predictions = pd.concat(prediction_parts, ignore_index=True)

    bootstrap_config = phase_config["bootstrap"]
    bootstrap_index = stratified_bootstrap_indices(
        y_test,
        int(bootstrap_config["n_resamples"]),
        int(bootstrap_config["random_seed"]),
    )
    _write_npz_atomic(
        processed_output / "bootstrap_indices.npz",
        indices=bootstrap_index,
    )
    _write_parquet_atomic(
        pd.DataFrame({"position": np.arange(len(locked_test_ids)), "patient_id": locked_test_ids}),
        processed_output / "bootstrap_patient_order.parquet",
    )
    bootstrap_parts: list[pd.DataFrame] = []
    for model_name in MODEL_ORDER:
        model_predictions = (
            test_predictions.loc[test_predictions["model"] == model_name]
            .set_index("patient_id")
            .loc[locked_test_ids]
            .reset_index()
        )
        bootstrap_frame, intervals = bootstrap_metrics(
            model_predictions,
            bootstrap_index,
            float(bootstrap_config["confidence_level"]),
        )
        bootstrap_frame.insert(0, "model", model_name)
        bootstrap_parts.append(bootstrap_frame)
        point_results[model_name]["confidence_intervals_95"] = intervals
    bootstrap_results = pd.concat(bootstrap_parts, ignore_index=True)

    _write_parquet_atomic(test_predictions, processed_output / "locked_test_predictions.parquet")
    _write_parquet_atomic(bootstrap_results, processed_output / "bootstrap_metrics.parquet")
    if final_sparsity is not None:
        prior_sparsity_path = processed_output / "l1_sparsity.parquet"
        prior_sparsity = pd.read_parquet(prior_sparsity_path)
        # CV folds are integers, whereas the final refit has the semantic label
        # ``full_development``.  Persist one stable string-typed column.
        prior_sparsity["fold"] = prior_sparsity["fold"].astype(str)
        _write_parquet_atomic(
            pd.concat([prior_sparsity, pd.DataFrame([final_sparsity])], ignore_index=True),
            prior_sparsity_path,
        )

    reference_model = selection["classical_reference_model"]
    final_report = {
        "generated_at": utc_now(),
        "phase": "1C leakage-safe classical baseline",
        "protocol_version": phase_config["protocol_version"],
        "locked_test_evaluated_once": True,
        "test_access_started_at": test_access_started,
        "test_patients": len(locked_test_ids),
        "bootstrap": bootstrap_config,
        "classical_reference_model_selected_by_inner_cv": reference_model,
        "P_classical_full": {
            "model": reference_model,
            "primary_metric": "roc_auc",
            "point": point_results[reference_model]["metrics"]["roc_auc"],
            "confidence_interval_95": point_results[reference_model]["confidence_intervals_95"]["roc_auc"],
        },
        "models": point_results,
    }
    performance_path = resolve_path(repo_root, outputs["locked_test_performance"])
    write_json_atomic(performance_path, final_report)
    performance_rows = []
    confusion_rows = []
    for model_name in MODEL_ORDER:
        model_result = point_results[model_name]
        row = {"model": model_name, "C": model_result["C"], "n_features": model_result["n_features"]}
        for metric in SCALAR_METRICS:
            row[metric] = model_result["metrics"][metric]
            row[f"{metric}_ci_lower"] = model_result["confidence_intervals_95"][metric]["lower"]
            row[f"{metric}_ci_upper"] = model_result["confidence_intervals_95"][metric]["upper"]
        performance_rows.append(row)
        confusion_rows.append({"model": model_name, **model_result["metrics"]["confusion_matrix"]})
    _write_csv_atomic(pd.DataFrame(performance_rows), repo_root / "reports/data/phase1c_locked_test_performance.csv")
    _write_csv_atomic(pd.DataFrame(confusion_rows), repo_root / "reports/data/phase1c_confusion_matrices.csv")

    canonical_after = _canonical_hashes(repo_root, config)
    if canonical_before != canonical_after:
        raise RuntimeError("Canonical artifacts changed during final Phase 1C evaluation")
    oof_path = processed_output / "oof_predictions.parquet"
    oof = pd.read_parquet(oof_path)
    leakage_checks = [
        {
            "check": "locked_test_absent_from_oof",
            "passed": set(oof["patient_id"]).isdisjoint(locked_test_ids),
            "evidence": len(set(oof["patient_id"]) & set(locked_test_ids)),
        },
        {
            "check": "one_oof_prediction_per_development_patient_per_model",
            "passed": len(oof) == len(development_ids) * len(MODEL_ORDER)
            and not oof.duplicated(["model", "patient_id"]).any(),
            "evidence": len(oof),
        },
        {
            "check": "one_test_prediction_per_patient_per_model",
            "passed": len(test_predictions) == len(locked_test_ids) * len(MODEL_ORDER)
            and not test_predictions.duplicated(["model", "patient_id"]).any(),
            "evidence": len(test_predictions),
        },
        {
            "check": "selected_hyperparameters_frozen_before_test_access",
            "passed": sha256_file(selection_path) == cv_complete["selected_hyperparameters_sha256"],
            "evidence": sha256_file(selection_path),
        },
        {
            "check": "thresholds_fixed_and_not_optimized",
            "passed": phase_config["classification_threshold"]["optimized"] is False,
            "evidence": phase_config["classification_threshold"],
        },
        {
            "check": "no_calibration",
            "passed": all(not hasattr(model, "calibrated_classifiers_") for model in fitted_models.values()),
            "evidence": None,
        },
        {
            "check": "same_bootstrap_indices_for_all_models",
            "passed": all(len(part) == len(bootstrap_index) for part in bootstrap_parts),
            "evidence": {"resamples": len(bootstrap_index), "models": len(bootstrap_parts)},
        },
        {
            "check": "canonical_inputs_immutable",
            "passed": canonical_before == canonical_after,
            "evidence": canonical_after,
        },
    ]
    failures = [item["check"] for item in leakage_checks if not item["passed"]]
    if failures:
        raise RuntimeError(f"Phase 1C leakage QC failed: {failures}")

    leakage_report = {
        "generated_at": utc_now(),
        "status": "pass",
        "models_fitted": True,
        "locked_test_evaluated_once": True,
        "test_use": "final evaluation and prespecified bootstrap only; no decisions changed",
        "checks": leakage_checks,
        "convergence_warnings": {
            model: diagnostics["convergence_warnings"]
            for model, diagnostics in fit_diagnostics.items()
        },
    }
    leakage_path = resolve_path(repo_root, outputs["leakage_qc"])
    write_json_atomic(leakage_path, leakage_report)

    artifact_paths = [
        resolve_path(repo_root, outputs["cv_results"]),
        resolve_path(repo_root, outputs["cv_summary"]),
        selection_path,
        performance_path,
        repo_root / "reports/data/phase1c_locked_test_performance.csv",
        repo_root / "reports/data/phase1c_confusion_matrices.csv",
        leakage_path,
        processed_output / "cv_results.parquet",
        processed_output / "cv_all_predictions.parquet",
        processed_output / "oof_predictions.parquet",
        processed_output / "locked_test_predictions.parquet",
        processed_output / "bootstrap_metrics.parquet",
        processed_output / "bootstrap_indices.npz",
        processed_output / "bootstrap_patient_order.parquet",
        processed_output / "l1_sparsity.parquet",
        artifacts_dir / "final_preprocessor.joblib",
        *[artifacts_dir / "models" / f"{model}.joblib" for model in MODEL_ORDER],
    ]
    artifact_hashes = {
        str(path.relative_to(repo_root)): sha256_file(path) for path in artifact_paths
    }
    provenance = {
        "generated_at": utc_now(),
        "project": config["project_id"],
        "dataset_version": config["cohort_version"],
        "preprocessing_specification": config["preprocessing"]["specification_id"],
        "phase1c_protocol_version": phase_config["protocol_version"],
        "frozen_protocol_sha256": specification_hash,
        "inner_cv_sha256": sha256_file(
            resolve_path(repo_root, phase_config["inner_cv"]["artifact"])
        ),
        "selected_hyperparameters_sha256": sha256_file(selection_path),
        "canonical_sha256": canonical_after,
        "preprocessing_state_sha256": preprocessing_hashes,
        "artifacts_sha256": artifact_hashes,
        "runtime_seconds": float(time.perf_counter() - stage_started),
        "software": dependency_versions(),
        "code": {
            **git_state(repo_root),
            "classical_baseline_sha256": sha256_file(Path(__file__)),
        },
    }
    provenance_path = resolve_path(repo_root, outputs["provenance"])
    write_json_atomic(provenance_path, provenance)
    receipt = {
        "completed_at": utc_now(),
        "status": "complete",
        "locked_test_evaluation_count_per_model": 1,
        "models": list(MODEL_ORDER),
        "selected_hyperparameters_sha256": sha256_file(selection_path),
        "locked_test_predictions_sha256": sha256_file(
            processed_output / "locked_test_predictions.parquet"
        ),
        "performance_sha256": sha256_file(performance_path),
        "leakage_qc_sha256": sha256_file(leakage_path),
    }
    write_json_atomic(receipt_path, receipt)
    return {
        "status": "pass",
        "runtime_seconds": provenance["runtime_seconds"],
        "reference_model": reference_model,
        "performance": final_report,
        "leakage_qc": leakage_report,
        "artifact_hashes": artifact_hashes,
    }


def recover_finalization_from_checkpoint(
    config_path: str | Path, observed_runtime_seconds: float | None = None
) -> dict[str, Any]:
    """Finalize an interrupted report using persisted predictions only.

    This recovery path deliberately never loads the expression matrix or labels.
    It is valid only after the atomic prediction, bootstrap, preprocessor, and
    fitted-model checkpoints exist, and prevents a second locked-test inference.
    """

    config, repo_root = load_config(config_path)
    phase_config = config["phase1c"]
    outputs = phase_config["outputs"]
    artifacts_dir = resolve_path(repo_root, outputs["artifacts_dir"])
    processed_output = resolve_path(repo_root, outputs["processed_dir"])
    receipt_path = artifacts_dir / "locked_test_access_receipt.json"
    if receipt_path.exists():
        raise FileExistsError("Locked-test finalization has already completed")

    selection_path = resolve_path(repo_root, outputs["selected_hyperparameters"])
    cv_complete_path = artifacts_dir / "cv_complete.json"
    predictions_path = processed_output / "locked_test_predictions.parquet"
    bootstrap_path = processed_output / "bootstrap_metrics.parquet"
    bootstrap_indices_path = processed_output / "bootstrap_indices.npz"
    patient_order_path = processed_output / "bootstrap_patient_order.parquet"
    required = [
        selection_path,
        cv_complete_path,
        predictions_path,
        bootstrap_path,
        bootstrap_indices_path,
        patient_order_path,
        artifacts_dir / "final_preprocessor.joblib",
        artifacts_dir / "final_preprocessing/state.json",
        *[artifacts_dir / "models" / f"{model}.joblib" for model in MODEL_ORDER],
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete final-evaluation checkpoint: {missing}")

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    cv_complete = json.loads(cv_complete_path.read_text(encoding="utf-8"))
    if sha256_file(selection_path) != cv_complete["selected_hyperparameters_sha256"]:
        raise ValueError("Selected hyperparameters changed after CV completion")
    specification = _protocol_specification(phase_config, config["preprocessing"])
    specification_hash = _specification_hash(specification)
    if specification_hash != selection["frozen_protocol_sha256"]:
        raise ValueError("Frozen Phase 1C protocol changed after model selection")

    inner = json.loads(
        resolve_path(repo_root, phase_config["inner_cv"]["artifact"]).read_text(encoding="utf-8")
    )
    development_ids = set(map(str, inner["development_case_ids"]))
    locked_test_ids = list(map(str, inner["locked_test_case_ids"]))
    predictions = pd.read_parquet(predictions_path)
    bootstrap = pd.read_parquet(bootstrap_path)
    patient_order = pd.read_parquet(patient_order_path)
    with np.load(bootstrap_indices_path) as archive:
        bootstrap_indices = archive["indices"]
    if patient_order["patient_id"].astype(str).tolist() != locked_test_ids:
        raise ValueError("Persisted bootstrap patient order differs from frozen locked-test order")
    if len(bootstrap_indices) != int(phase_config["bootstrap"]["n_resamples"]):
        raise ValueError("Persisted bootstrap count differs from frozen protocol")

    state = json.loads(
        (artifacts_dir / "final_preprocessing/state.json").read_text(encoding="utf-8")
    )
    n_features = int(state["dimensions"]["genes_final"])
    alpha = (1.0 - float(phase_config["bootstrap"]["confidence_level"])) / 2.0
    point_results: dict[str, Any] = {}
    performance_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    loaded_models: dict[str, Any] = {}
    for model_name in MODEL_ORDER:
        model_predictions = predictions.loc[predictions["model"] == model_name]
        if len(model_predictions) != len(locked_test_ids):
            raise ValueError(f"Incomplete persisted predictions for {model_name}")
        if set(model_predictions["patient_id"].astype(str)) != set(locked_test_ids):
            raise ValueError(f"Unexpected patient IDs in persisted predictions for {model_name}")
        metrics = classification_metrics(
            model_predictions["y_true"].to_numpy(dtype=int),
            model_predictions["score"].to_numpy(dtype=float),
            model_predictions["predicted_class"].to_numpy(dtype=int),
        )
        model_bootstrap = bootstrap.loc[bootstrap["model"] == model_name]
        if len(model_bootstrap) != len(bootstrap_indices):
            raise ValueError(f"Incomplete bootstrap checkpoint for {model_name}")
        intervals = {
            metric: {
                "lower": float(model_bootstrap[metric].quantile(alpha)),
                "upper": float(model_bootstrap[metric].quantile(1.0 - alpha)),
            }
            for metric in SCALAR_METRICS
        }
        loaded_models[model_name] = joblib.load(
            artifacts_dir / "models" / f"{model_name}.joblib"
        )
        point_results[model_name] = {
            "C": float(selection["models"][model_name]["C"]),
            "metrics": metrics,
            "confidence_intervals_95": intervals,
            "n_features": n_features,
            "preprocessing_fit_runtime_seconds": None,
            "test_transform_runtime_seconds": None,
            "training_runtime_seconds": None,
            "inference_runtime_seconds": None,
            "runtime_recovery_note": (
                "Per-step final runtimes were in memory when report serialization failed; "
                "CV runtimes remain complete and the observed final-stage wall time is recorded."
            ),
        }
        row = {"model": model_name, "C": point_results[model_name]["C"], "n_features": n_features}
        for metric in SCALAR_METRICS:
            row[metric] = metrics[metric]
            row[f"{metric}_ci_lower"] = intervals[metric]["lower"]
            row[f"{metric}_ci_upper"] = intervals[metric]["upper"]
        performance_rows.append(row)
        confusion_rows.append({"model": model_name, **metrics["confusion_matrix"]})

    # Repair the only failed write from the interrupted run without refitting.
    l1_path = processed_output / "l1_sparsity.parquet"
    l1 = pd.read_parquet(l1_path)
    l1["fold"] = l1["fold"].astype(str)
    if not (l1["stage"] == "full_development_refit").any():
        final_sparsity = {
            "stage": "full_development_refit",
            "fold": "full_development",
            "C": float(selection["models"]["logistic_l1"]["C"]),
            **_l1_sparsity(loaded_models["logistic_l1"]),
        }
        _write_parquet_atomic(pd.concat([l1, pd.DataFrame([final_sparsity])], ignore_index=True), l1_path)

    reference_model = selection["classical_reference_model"]
    final_report = {
        "generated_at": utc_now(),
        "phase": "1C leakage-safe classical baseline",
        "protocol_version": phase_config["protocol_version"],
        "locked_test_evaluated_once": True,
        "finalization_mode": "persisted_checkpoint_recovery_without_second_test_inference",
        "observed_final_stage_wall_time_seconds": observed_runtime_seconds,
        "test_patients": len(locked_test_ids),
        "bootstrap": phase_config["bootstrap"],
        "classical_reference_model_selected_by_inner_cv": reference_model,
        "P_classical_full": {
            "model": reference_model,
            "primary_metric": "roc_auc",
            "point": point_results[reference_model]["metrics"]["roc_auc"],
            "confidence_interval_95": point_results[reference_model]["confidence_intervals_95"]["roc_auc"],
        },
        "models": point_results,
    }
    performance_path = resolve_path(repo_root, outputs["locked_test_performance"])
    write_json_atomic(performance_path, final_report)
    _write_csv_atomic(pd.DataFrame(performance_rows), repo_root / "reports/data/phase1c_locked_test_performance.csv")
    _write_csv_atomic(pd.DataFrame(confusion_rows), repo_root / "reports/data/phase1c_confusion_matrices.csv")

    oof = pd.read_parquet(processed_output / "oof_predictions.parquet")
    leakage_checks = [
        {"check": "locked_test_absent_from_oof", "passed": set(oof["patient_id"]).isdisjoint(locked_test_ids), "evidence": len(set(oof["patient_id"]) & set(locked_test_ids))},
        {"check": "one_oof_prediction_per_development_patient_per_model", "passed": len(oof) == len(development_ids) * len(MODEL_ORDER) and not oof.duplicated(["model", "patient_id"]).any(), "evidence": len(oof)},
        {"check": "one_test_prediction_per_patient_per_model", "passed": len(predictions) == len(locked_test_ids) * len(MODEL_ORDER) and not predictions.duplicated(["model", "patient_id"]).any(), "evidence": len(predictions)},
        {"check": "selected_hyperparameters_frozen_before_test_access", "passed": sha256_file(selection_path) == cv_complete["selected_hyperparameters_sha256"], "evidence": sha256_file(selection_path)},
        {"check": "thresholds_fixed_and_not_optimized", "passed": phase_config["classification_threshold"]["optimized"] is False, "evidence": phase_config["classification_threshold"]},
        {"check": "no_calibration", "passed": all(not hasattr(model, "calibrated_classifiers_") for model in loaded_models.values()), "evidence": None},
        {"check": "same_bootstrap_indices_for_all_models", "passed": all(len(bootstrap.loc[bootstrap["model"] == model]) == len(bootstrap_indices) for model in MODEL_ORDER), "evidence": {"resamples": len(bootstrap_indices), "models": len(MODEL_ORDER)}},
        {"check": "checkpoint_recovery_did_not_load_expression_or_labels", "passed": True, "evidence": "recovery function has no expression/labels input path"},
    ]
    failures = [item["check"] for item in leakage_checks if not item["passed"]]
    if failures:
        raise RuntimeError(f"Phase 1C leakage QC failed: {failures}")
    leakage_report = {
        "generated_at": utc_now(), "status": "pass", "models_fitted": True,
        "locked_test_evaluated_once": True,
        "test_use": "single final evaluation and prespecified bootstrap; no decisions changed",
        "recovery": "reports reconstructed from atomic checkpoints after a post-inference serialization error",
        "checks": leakage_checks,
    }
    leakage_path = resolve_path(repo_root, outputs["leakage_qc"])
    write_json_atomic(leakage_path, leakage_report)

    canonical = _canonical_hashes(repo_root, config)
    artifact_paths = [
        resolve_path(repo_root, outputs["cv_results"]), resolve_path(repo_root, outputs["cv_summary"]),
        selection_path, performance_path, repo_root / "reports/data/phase1c_locked_test_performance.csv",
        repo_root / "reports/data/phase1c_confusion_matrices.csv", leakage_path,
        processed_output / "cv_results.parquet", processed_output / "cv_all_predictions.parquet",
        processed_output / "oof_predictions.parquet", predictions_path, bootstrap_path,
        bootstrap_indices_path, patient_order_path, l1_path,
        artifacts_dir / "final_preprocessor.joblib",
        *[artifacts_dir / "models" / f"{model}.joblib" for model in MODEL_ORDER],
    ]
    artifact_hashes = {str(path.relative_to(repo_root)): sha256_file(path) for path in artifact_paths}
    preprocessing_state_paths = [
        artifacts_dir / "final_preprocessing/state.json",
        artifacts_dir / "final_preprocessing/feature_state.parquet",
        artifacts_dir / "final_preprocessing/training_sample_qc.parquet",
    ]
    provenance = {
        "generated_at": utc_now(), "project": config["project_id"],
        "dataset_version": config["cohort_version"],
        "preprocessing_specification": config["preprocessing"]["specification_id"],
        "phase1c_protocol_version": phase_config["protocol_version"],
        "frozen_protocol_sha256": specification_hash,
        "inner_cv_sha256": sha256_file(resolve_path(repo_root, phase_config["inner_cv"]["artifact"])),
        "selected_hyperparameters_sha256": sha256_file(selection_path),
        "canonical_sha256": canonical,
        "preprocessing_state_sha256": {str(path.relative_to(repo_root)): sha256_file(path) for path in preprocessing_state_paths},
        "artifacts_sha256": artifact_hashes,
        "runtime_seconds": observed_runtime_seconds,
        "runtime_note": "Total observed wall time retained; per-step final timings lost in post-inference serialization failure. CV timings are complete.",
        "finalization_mode": "persisted_checkpoint_recovery_without_second_test_inference",
        "software": dependency_versions(),
        "code": {**git_state(repo_root), "classical_baseline_sha256": sha256_file(Path(__file__))},
    }
    provenance_path = resolve_path(repo_root, outputs["provenance"])
    write_json_atomic(provenance_path, provenance)
    receipt = {
        "completed_at": utc_now(), "status": "complete",
        "locked_test_evaluation_count_per_model": 1, "models": list(MODEL_ORDER),
        "recovered_from_atomic_checkpoints": True,
        "second_test_inference_performed": False,
        "selected_hyperparameters_sha256": sha256_file(selection_path),
        "locked_test_predictions_sha256": sha256_file(predictions_path),
        "performance_sha256": sha256_file(performance_path),
        "leakage_qc_sha256": sha256_file(leakage_path),
    }
    write_json_atomic(receipt_path, receipt)
    return {"status": "pass", "reference_model": reference_model, "performance": final_report}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--stage", required=True, choices=["cv", "final", "recover-final"])
    parser.add_argument("--observed-runtime-seconds", type=float)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    if args.stage == "cv":
        result = run_cv_stage(args.config)
        LOGGER.info("Development-only model selection complete in %.2f seconds", result["runtime_seconds"])
    elif args.stage == "final":
        result = run_final_stage(args.config)
        LOGGER.info("Locked-test evaluation complete in %.2f seconds", result["runtime_seconds"])
    else:
        result = recover_finalization_from_checkpoint(args.config, args.observed_runtime_seconds)
        LOGGER.info("Locked-test report recovered without a second test inference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
