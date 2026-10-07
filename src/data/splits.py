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
