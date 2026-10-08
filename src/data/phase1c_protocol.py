"""Create and audit the locked-test/inner-CV protocol for Phase 1C.

This module does not fit classifiers.  It creates deterministic 5-fold
stratified CV partitions inside the canonical development cohort and installs
fail-fast guards that later model-selection code must call before fitting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
from sklearn.model_selection import StratifiedKFold

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


class LockedTestLeakageError(ValueError):
    """Raised when locked-test patients enter a pre-final evaluation stage."""


def _as_unique_ids(values: Iterable[str], name: str) -> list[str]:
    ids = [str(value) for value in values]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{name} contains duplicate patient IDs")
    return ids


def assert_locked_test_isolation(
    patient_ids: Iterable[str],
    locked_test_ids: Iterable[str],
    *,
    stage: str,
) -> None:
    """Fail if fitting, selection, calibration or threshold data include test."""

    observed = set(map(str, patient_ids))
    locked = set(map(str, locked_test_ids))
    overlap = sorted(observed & locked)
    if overlap:
        raise LockedTestLeakageError(
            f"Locked-test leakage at stage={stage!r}: {len(overlap)} patients; "
            f"examples={overlap[:10]}"
        )


def _case_index(labels: pd.DataFrame) -> dict[str, int]:
    return {str(case_id): int(position) for position, case_id in enumerate(labels["case_id"])}


def generate_phase1c_inner_cv(
    labels: pd.DataFrame,
    canonical_splits: Mapping[str, Any],
    inner_config: Mapping[str, Any],
    *,
    dataset_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Generate inner CV using only canonical development patients."""

    required = {"case_id", "er_label"}
    if not required.issubset(labels.columns):
        raise ValueError(f"labels must contain {sorted(required)}")
    if labels["case_id"].astype(str).duplicated().any():
        raise ValueError("labels contains duplicate patient IDs")
    if labels["er_label"].isna().any() or set(labels["er_label"].astype(int)) != {0, 1}:
        raise ValueError("labels must be complete and contain both binary classes")

    outer = canonical_splits["stratified_train_test"]
    development_ids = _as_unique_ids(outer["train_case_ids"], "development split")
    locked_test_ids = _as_unique_ids(outer["test_case_ids"], "locked test split")
    if set(development_ids) & set(locked_test_ids):
        raise ValueError("Canonical development and locked-test splits overlap")
    label_ids = set(labels["case_id"].astype(str))
    if set(development_ids) | set(locked_test_ids) != label_ids:
        raise ValueError("Canonical outer split does not cover the label cohort exactly")

    # The sorted development order is part of the protocol.  Development
    # indices below always refer to this order; canonical indices refer to the
    # immutable labels/expression row order.
    development_ids = sorted(development_ids)
    by_id = labels.assign(case_id=labels["case_id"].astype(str)).set_index("case_id")
    development_labels = by_id.loc[development_ids, "er_label"].astype(int).to_numpy()
    canonical_positions = _case_index(labels)
    seed = int(inner_config["random_seed"])
    n_splits = int(inner_config["n_splits"])
    shuffle = bool(inner_config.get("shuffle", True))
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=seed if shuffle else None,
    )
    folds: list[dict[str, Any]] = []
    for fold_number, (train_index, validation_index) in enumerate(
        splitter.split(development_ids, development_labels)
    ):
        train_ids = [development_ids[int(index)] for index in train_index]
        validation_ids = [development_ids[int(index)] for index in validation_index]
        assert_locked_test_isolation(train_ids, locked_test_ids, stage=f"inner_fold_{fold_number}_fit")
        assert_locked_test_isolation(
            validation_ids, locked_test_ids, stage=f"inner_fold_{fold_number}_validation"
        )
        fold_labels = by_id.loc[validation_ids, "er_label"].astype(int)
        folds.append(
            {
                "fold": fold_number,
                "train_case_ids": train_ids,
                "validation_case_ids": validation_ids,
                "train_development_indices": [int(index) for index in train_index],
                "validation_development_indices": [int(index) for index in validation_index],
                "train_canonical_indices": [canonical_positions[case_id] for case_id in train_ids],
                "validation_canonical_indices": [
                    canonical_positions[case_id] for case_id in validation_ids
                ],
                "validation_class_distribution": {
                    "ER_negative": int(fold_labels.eq(0).sum()),
                    "ER_positive": int(fold_labels.eq(1).sum()),
                },
            }
        )

    return {
        "schema_version": 1,
        "protocol": "Phase 1C development-only inner cross-validation",
        "unit": "GDC case/patient",
        "outer_split_source": "data/processed/splits.json:stratified_train_test",
        "development_order": "lexicographic_case_id",
        "random_seed": seed,
        "n_splits": n_splits,
        "shuffle": shuffle,
        "development_case_ids": development_ids,
        "locked_test_case_ids": sorted(locked_test_ids),
        "folds": folds,
        "dataset_sha256": dict(dataset_hashes or {}),
    }


def validate_phase1c_inner_cv(
    artifact: Mapping[str, Any], labels: pd.DataFrame
) -> dict[str, Any]:
    """Validate coverage, indices, class stratification and locked-test isolation."""

    development = list(map(str, artifact["development_case_ids"]))
    locked_test = set(map(str, artifact["locked_test_case_ids"]))
    development_set = set(development)
    by_id = labels.assign(case_id=labels["case_id"].astype(str)).set_index("case_id")
    canonical_ids = labels["case_id"].astype(str).tolist()
    validation_counts = {case_id: 0 for case_id in development}
    checks: list[dict[str, Any]] = []
    fold_disjoint = True
    fold_exhaustive = True
    locked_test_overlap = 0
    indices_resolve = True
    class_distributions: list[dict[str, Any]] = []

    for fold in artifact["folds"]:
        train_ids = list(map(str, fold["train_case_ids"]))
        validation_ids = list(map(str, fold["validation_case_ids"]))
        train_set, validation_set = set(train_ids), set(validation_ids)
        fold_disjoint &= not bool(train_set & validation_set)
        fold_exhaustive &= train_set | validation_set == development_set
        locked_test_overlap += len((train_set | validation_set) & locked_test)
        for case_id in validation_ids:
            if case_id in validation_counts:
                validation_counts[case_id] += 1

        development_train_from_indices = [
            development[int(index)] for index in fold["train_development_indices"]
        ]
        development_validation_from_indices = [
            development[int(index)] for index in fold["validation_development_indices"]
        ]
        canonical_train_from_indices = [
            canonical_ids[int(index)] for index in fold["train_canonical_indices"]
        ]
        canonical_validation_from_indices = [
            canonical_ids[int(index)] for index in fold["validation_canonical_indices"]
        ]
        indices_resolve &= (
            development_train_from_indices == train_ids
            and development_validation_from_indices == validation_ids
            and canonical_train_from_indices == train_ids
            and canonical_validation_from_indices == validation_ids
        )
        observed = by_id.loc[validation_ids, "er_label"].astype(int)
        expected_distribution = {
            "ER_negative": int(observed.eq(0).sum()),
            "ER_positive": int(observed.eq(1).sum()),
        }
        indices_resolve &= expected_distribution == fold["validation_class_distribution"]
        class_distributions.append(
            {
                "fold": int(fold["fold"]),
                "training_size": len(train_ids),
                "validation_size": len(validation_ids),
                **expected_distribution,
                "ER_positive_prevalence": float(observed.mean()),
            }
        )

    validation_frequency = pd.Series(validation_counts).value_counts().sort_index().to_dict()
    fold_prevalences = [row["ER_positive_prevalence"] for row in class_distributions]
    development_prevalence = float(by_id.loc[development, "er_label"].astype(int).mean())
    max_prevalence_deviation = max(abs(value - development_prevalence) for value in fold_prevalences)
    checks.extend(
        [
            {
                "check": "development_locked_test_disjoint",
                "passed": not bool(development_set & locked_test),
                "evidence": len(development_set & locked_test),
            },
            {
                "check": "locked_test_absent_from_every_inner_fold",
                "passed": locked_test_overlap == 0,
                "evidence": locked_test_overlap,
            },
            {
                "check": "inner_train_validation_disjoint",
                "passed": fold_disjoint,
                "evidence": None,
            },
            {
                "check": "each_inner_fold_exhausts_development",
                "passed": fold_exhaustive,
                "evidence": len(development_set),
            },
            {
                "check": "each_development_patient_validated_exactly_once",
                "passed": set(validation_counts.values()) == {1},
                "evidence": {str(key): int(value) for key, value in validation_frequency.items()},
            },
            {
                "check": "patient_indices_resolve_exactly_to_ids",
                "passed": indices_resolve,
                "evidence": None,
            },
            {
                "check": "stratification_within_one_patient_per_class_rounding",
                "passed": max_prevalence_deviation <= 1.0 / min(
                    row["validation_size"] for row in class_distributions
                ),
                "evidence": {
                    "development_er_positive_prevalence": development_prevalence,
                    "maximum_fold_deviation": max_prevalence_deviation,
                },
            },
        ]
    )
    failures = [item["check"] for item in checks if not item["passed"]]
    if failures:
        raise ValueError(f"Phase 1C inner-CV validation failed: {failures}")
    return {
        "status": "pass",
        "development_patients": len(development),
        "locked_test_patients": len(locked_test),
        "development_class_distribution": {
            "ER_negative": int(by_id.loc[development, "er_label"].eq(0).sum()),
            "ER_positive": int(by_id.loc[development, "er_label"].eq(1).sum()),
        },
        "checks": checks,
        "folds": class_distributions,
    }


def protocol_fingerprint(artifact: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(artifact))).hexdigest()


def run_phase1c_protocol(config_path: str | Path) -> dict[str, Any]:
    """Generate deterministic inner folds and protocol QC; fit no models."""

    config, repo_root = load_config(config_path)
    phase_config = config.get("phase1c")
    if not isinstance(phase_config, dict):
        raise ValueError("Configuration is missing phase1c")
    processed_dir = resolve_path(repo_root, config["paths"]["processed_dir"])
    labels_path = processed_dir / "labels.parquet"
    expression_path = processed_dir / "expression.parquet"
    outer_splits_path = processed_dir / "splits.json"
    canonical_paths = [labels_path, expression_path, outer_splits_path]
    for path in canonical_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing canonical artifact: {path}")
    before_hashes = {str(path.relative_to(repo_root)): sha256_file(path) for path in canonical_paths}
    labels = pd.read_parquet(labels_path)
    outer_splits = json.loads(outer_splits_path.read_text(encoding="utf-8"))
    inner_config = phase_config["inner_cv"]

    artifact = generate_phase1c_inner_cv(
        labels,
        outer_splits,
        inner_config,
        dataset_hashes=before_hashes,
    )
    repeated = generate_phase1c_inner_cv(
        labels,
        outer_splits,
        inner_config,
        dataset_hashes=before_hashes,
    )
    if canonical_json_bytes(artifact) != canonical_json_bytes(repeated):
        raise RuntimeError("Inner-CV generation is not exactly reproducible")
    qc = validate_phase1c_inner_cv(artifact, labels)
    qc["checks"].append(
        {
            "check": "exact_regeneration_reproducibility",
            "passed": True,
            "evidence": protocol_fingerprint(artifact),
        }
    )
    artifact_path = resolve_path(repo_root, inner_config["artifact"])
    write_json_atomic(artifact_path, artifact)
    artifact_hash = sha256_file(artifact_path)

    after_hashes = {str(path.relative_to(repo_root)): sha256_file(path) for path in canonical_paths}
    if before_hashes != after_hashes:
        raise RuntimeError("Canonical artifacts changed while generating Phase 1C protocol")
    report = {
        "generated_at": utc_now(),
        "status": "pass",
        "phase": "1C protocol gate",
        "models_fitted": False,
        "locked_test_evaluated": False,
        "protocol_version": phase_config["protocol_version"],
        "protocol_status": phase_config["status"],
        "inner_cv_artifact": str(artifact_path.relative_to(repo_root)),
        "inner_cv_sha256": artifact_hash,
        "canonical_artifacts_immutable": before_hashes == after_hashes,
        "canonical_sha256_before": before_hashes,
        "canonical_sha256_after": after_hashes,
        "qc": qc,
        "prespecified_experiment": {
            key: phase_config[key]
            for key in [
                "locked_test_policy",
                "selection",
                "class_imbalance",
                "classification_threshold",
                "models",
                "metrics",
                "bootstrap",
                "predictions",
            ]
        },
    }
    qc_path = resolve_path(repo_root, inner_config["qc_report"])
    write_json_atomic(qc_path, report)

    provenance = {
        "generated_at": utc_now(),
        "project": config["project_id"],
        "dataset_version": config["cohort_version"],
        "preprocessing_specification": config["preprocessing"]["specification_id"],
        "phase1c_protocol_version": phase_config["protocol_version"],
        "models_fitted": False,
        "locked_test_evaluated": False,
        "configuration_sha256": sha256_file(Path(config_path).resolve()),
        "inner_cv_artifact": str(artifact_path.relative_to(repo_root)),
        "inner_cv_sha256": artifact_hash,
        "canonical_sha256": after_hashes,
        "qc_report": str(qc_path.relative_to(repo_root)),
        "qc_report_sha256": sha256_file(qc_path),
        "code": {
            **git_state(repo_root),
            "phase1c_protocol_sha256": sha256_file(Path(__file__)),
        },
        "dependencies": dependency_versions(),
        "references": {
            "logistic_regression": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html",
            "linear_svc": "https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html",
            "rbf_svc": "https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html",
            "metrics": "https://scikit-learn.org/stable/modules/model_evaluation.html",
        },
    }
    provenance_path = resolve_path(repo_root, inner_config["provenance_report"])
    write_json_atomic(provenance_path, provenance)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    report = run_phase1c_protocol(args.config)
    LOGGER.info(
        "Phase 1C protocol gate passed: %s; models fitted=%s",
        report["inner_cv_artifact"],
        report["models_fitted"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
