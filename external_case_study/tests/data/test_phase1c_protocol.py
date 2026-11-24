import copy

import pandas as pd
import pytest

from src.data.phase1c_protocol import (
    LockedTestLeakageError,
    assert_locked_test_isolation,
    generate_phase1c_inner_cv,
    protocol_fingerprint,
    validate_phase1c_inner_cv,
)


INNER_CONFIG = {"n_splits": 5, "shuffle": True, "random_seed": 1729}


def labels() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": [f"case-{index:03d}" for index in range(60)],
            "er_label": [0] * 20 + [1] * 40,
        }
    )


def canonical_splits() -> dict:
    # Development contains both classes and locked test is never passed to CV.
    development = [f"case-{index:03d}" for index in list(range(15)) + list(range(20, 50))]
    test = sorted(set(labels()["case_id"]) - set(development))
    return {
        "stratified_train_test": {
            "train_case_ids": development,
            "test_case_ids": test,
        }
    }


def test_inner_cv_is_exactly_reproducible_and_label_row_order_invariant():
    first = generate_phase1c_inner_cv(labels(), canonical_splits(), INNER_CONFIG)
    second = generate_phase1c_inner_cv(
        labels().sample(frac=1, random_state=99).reset_index(drop=True),
        canonical_splits(),
        INNER_CONFIG,
    )
    # Canonical row indices intentionally differ after row shuffling; all
    # patient assignments and development-relative indices must not.
    for artifact in (first, second):
        for fold in artifact["folds"]:
            fold.pop("train_canonical_indices")
            fold.pop("validation_canonical_indices")
    assert first == second


def test_locked_test_is_absent_and_each_development_patient_validates_once():
    artifact = generate_phase1c_inner_cv(labels(), canonical_splits(), INNER_CONFIG)
    qc = validate_phase1c_inner_cv(artifact, labels())
    assert qc["status"] == "pass"
    locked = set(artifact["locked_test_case_ids"])
    validation_counts = {case_id: 0 for case_id in artifact["development_case_ids"]}
    for fold in artifact["folds"]:
        assert locked.isdisjoint(fold["train_case_ids"])
        assert locked.isdisjoint(fold["validation_case_ids"])
        assert set(fold["train_case_ids"]).isdisjoint(fold["validation_case_ids"])
        for case_id in fold["validation_case_ids"]:
            validation_counts[case_id] += 1
    assert set(validation_counts.values()) == {1}


def test_patient_indices_resolve_exactly_to_ids():
    artifact = generate_phase1c_inner_cv(labels(), canonical_splits(), INNER_CONFIG)
    development = artifact["development_case_ids"]
    canonical = labels()["case_id"].tolist()
    for fold in artifact["folds"]:
        assert [development[i] for i in fold["train_development_indices"]] == fold[
            "train_case_ids"
        ]
        assert [development[i] for i in fold["validation_development_indices"]] == fold[
            "validation_case_ids"
        ]
        assert [canonical[i] for i in fold["train_canonical_indices"]] == fold[
            "train_case_ids"
        ]
        assert [canonical[i] for i in fold["validation_canonical_indices"]] == fold[
            "validation_case_ids"
        ]


@pytest.mark.parametrize(
    "stage",
    ["preprocessing_fit", "model_fit", "model_selection", "threshold_selection", "calibration"],
)
def test_locked_test_guard_rejects_every_pre_final_use(stage):
    split = canonical_splits()["stratified_train_test"]
    with pytest.raises(LockedTestLeakageError, match=stage):
        assert_locked_test_isolation(
            [split["train_case_ids"][0], split["test_case_ids"][0]],
            split["test_case_ids"],
            stage=stage,
        )


def test_locked_test_guard_accepts_development_only():
    split = canonical_splits()["stratified_train_test"]
    assert_locked_test_isolation(
        split["train_case_ids"], split["test_case_ids"], stage="model_selection"
    )


def test_validation_fails_if_locked_test_is_injected_into_fold():
    artifact = generate_phase1c_inner_cv(labels(), canonical_splits(), INNER_CONFIG)
    tampered = copy.deepcopy(artifact)
    tampered["folds"][0]["train_case_ids"][0] = tampered["locked_test_case_ids"][0]
    with pytest.raises(ValueError, match="validation failed"):
        validate_phase1c_inner_cv(tampered, labels())


def test_protocol_fingerprint_is_stable_and_seed_sensitive():
    first = generate_phase1c_inner_cv(labels(), canonical_splits(), INNER_CONFIG)
    repeated = generate_phase1c_inner_cv(labels(), canonical_splits(), INNER_CONFIG)
    changed = generate_phase1c_inner_cv(
        labels(), canonical_splits(), {**INNER_CONFIG, "random_seed": 1730}
    )
    assert protocol_fingerprint(first) == protocol_fingerprint(repeated)
    assert protocol_fingerprint(first) != protocol_fingerprint(changed)
