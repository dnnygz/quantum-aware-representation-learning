"""Reusable, patient-level stratified split generation."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def generate_splits(labels: pd.DataFrame, split_config: dict[str, Any]) -> dict[str, Any]:
    """Create deterministic patient-ID splits shared by every future model."""

    required = {"case_id", "er_label"}
    if not required.issubset(labels):
        raise ValueError(f"labels must contain {sorted(required)}")
    if labels["case_id"].duplicated().any():
        raise ValueError("Patient IDs must be unique before splitting")
    if labels["er_label"].isna().any() or not set(labels["er_label"].astype(int)).issubset({0, 1}):
        raise ValueError("Labels must be complete and binary before splitting")

    ordered = labels[["case_id", "er_label"]].sort_values("case_id").reset_index(drop=True)
    ids = ordered["case_id"].astype(str).to_numpy()
    y = ordered["er_label"].astype(int).to_numpy()
    seed = int(split_config["random_seed"])
    test_size = float(split_config["test_size"])
    n_splits = int(split_config["n_splits"])
    shuffle = bool(split_config.get("shuffle", True))

    train_ids, test_ids = train_test_split(
        ids, test_size=test_size, random_state=seed, stratify=y, shuffle=True
    )
    folds = []
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=shuffle, random_state=seed if shuffle else None
    )
    for fold, (train_index, validation_index) in enumerate(splitter.split(ids, y)):
        folds.append(
            {
                "fold": fold,
                "train_case_ids": ids[train_index].tolist(),
                "validation_case_ids": ids[validation_index].tolist(),
            }
        )

    return {
        "unit": "GDC case/patient",
        "random_seed": seed,
        "stratified_train_test": {
            "test_size": test_size,
            "train_case_ids": sorted(train_ids.tolist()),
            "test_case_ids": sorted(test_ids.tolist()),
        },
        "stratified_kfold": {"n_splits": n_splits, "shuffle": shuffle, "folds": folds},
    }


def validate_splits(splits: dict[str, Any], labels: pd.DataFrame) -> dict[str, Any]:
    """Validate patient coverage, disjointness and stratification artifacts."""

    universe = set(labels["case_id"].astype(str))
    by_id = labels.set_index(labels["case_id"].astype(str))["er_label"].astype(int)
    train = set(splits["stratified_train_test"]["train_case_ids"])
    test = set(splits["stratified_train_test"]["test_case_ids"])
    checks = [
        {"check": "train_test_disjoint", "passed": not bool(train & test), "evidence": len(train & test)},
        {"check": "train_test_exhaustive", "passed": train | test == universe, "evidence": len(train | test)},
    ]
    validation_counts = {case_id: 0 for case_id in universe}
    fold_disjoint = True
    fold_exhaustive = True
    fold_distribution = []
    for fold in splits["stratified_kfold"]["folds"]:
        fold_train = set(fold["train_case_ids"])
        validation = set(fold["validation_case_ids"])
        fold_disjoint &= not bool(fold_train & validation)
        fold_exhaustive &= fold_train | validation == universe
        for case_id in validation:
            validation_counts[case_id] += 1
        fold_distribution.append(
            {
                "fold": fold["fold"],
                "validation_size": len(validation),
                "validation_er_positive": int(by_id.loc[list(validation)].eq(1).sum()),
                "validation_er_negative": int(by_id.loc[list(validation)].eq(0).sum()),
            }
        )
    checks.extend(
        [
            {"check": "kfold_train_validation_disjoint", "passed": fold_disjoint, "evidence": None},
            {"check": "kfold_each_fold_exhaustive", "passed": fold_exhaustive, "evidence": None},
            {
                "check": "each_patient_once_in_kfold_validation",
                "passed": set(validation_counts.values()) == {1},
                "evidence": {
                    str(int(key)): int(value)
                    for key, value in pd.Series(validation_counts).value_counts().sort_index().items()
                },
            },
        ]
    )
    failures = [check["check"] for check in checks if not check["passed"]]
    if failures:
        raise ValueError(f"Split QC failed: {failures}")
    return {
        "status": "pass",
        "checks": checks,
        "train_test_distribution": {
            "train": {
                str(int(key)): int(value)
                for key, value in by_id.loc[list(train)].value_counts().sort_index().items()
            },
            "test": {
                str(int(key)): int(value)
                for key, value in by_id.loc[list(test)].value_counts().sort_index().items()
            },
        },
        "fold_distribution": fold_distribution,
    }
