import pandas as pd

from src.data.splits import generate_splits


CONFIG = {"random_seed": 42, "test_size": 0.25, "n_splits": 4, "shuffle": True}


def labels():
    return pd.DataFrame(
        {
            "case_id": [f"case-{index:02d}" for index in range(24)],
            "er_label": [0, 1] * 12,
        }
    )


def test_splits_are_reproducible_and_order_invariant():
    first = generate_splits(labels(), CONFIG)
    second = generate_splits(labels().sample(frac=1, random_state=9), CONFIG)
    assert first == second


def test_no_patient_crosses_train_test_boundary():
    result = generate_splits(labels(), CONFIG)
    split = result["stratified_train_test"]
    assert set(split["train_case_ids"]).isdisjoint(split["test_case_ids"])
    assert set(split["train_case_ids"]) | set(split["test_case_ids"]) == set(labels()["case_id"])
    for fold in result["stratified_kfold"]["folds"]:
        assert set(fold["train_case_ids"]).isdisjoint(fold["validation_case_ids"])
