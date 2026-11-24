import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import (
    ExactZeroVarianceFilter,
    FrozenTMMLogCPM,
    RNASeqPreprocessor,
    TrainingExpressionFilter,
    TrainingStandardScaler,
)


GENES = ["g1", "g2", "g3", "g4", "g5", "g6"]


def training_counts() -> pd.DataFrame:
    # g4 is always zero, g5 is expressed in only one sample and g6 is a
    # proportional constant after library normalization.
    return pd.DataFrame(
        [
            [100, 50, 20, 0, 0, 10],
            [120, 55, 25, 0, 0, 12],
            [90, 45, 18, 0, 0, 9],
            [110, 52, 22, 0, 5, 11],
            [105, 48, 19, 0, 0, 10],
        ],
        index=[f"train-{i}" for i in range(5)],
        columns=GENES,
        dtype=np.int64,
    )


def validation_counts(multiplier: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        [
            [80, 40, 15, 1000 * multiplier, 900 * multiplier, 8],
            [140, 60, 30, 2000 * multiplier, 800 * multiplier, 14],
        ],
        index=["validation-a", "validation-b"],
        columns=GENES,
        dtype=np.int64,
    )


def permissive_preprocessor(scale: bool = False) -> RNASeqPreprocessor:
    # Synthetic libraries are tiny, so a low CPM threshold keeps ordinary genes.
    return RNASeqPreprocessor(
        cpm_threshold=1.0,
        min_training_prevalence=0.40,
        prior_count=2.0,
        scale=scale,
    )


def test_modifying_validation_cannot_change_any_fitted_parameter():
    train = training_counts()
    fitted = permissive_preprocessor(scale=True).fit(train)
    before = fitted.parameter_fingerprint()
    fitted.transform(validation_counts(multiplier=1))
    after_first = fitted.parameter_fingerprint()
    fitted.transform(validation_counts(multiplier=100))
    after_second = fitted.parameter_fingerprint()
    assert before == after_first == after_second


def test_expression_gene_mask_depends_only_on_training():
    train = training_counts()
    filter_ = TrainingExpressionFilter(cpm_threshold=1.0, min_training_prevalence=0.40).fit(
        train
    )
    before = filter_.keep_mask_.copy()
    # g4 is enormous in validation but absent from training and must remain excluded.
    transformed = filter_.transform(validation_counts(multiplier=100))
    assert np.array_equal(filter_.keep_mask_, before)
    assert "g4" not in transformed.columns


def test_tmm_reference_and_calibration_depend_only_on_training():
    train = training_counts().drop(columns=["g4"])
    normalizer = FrozenTMMLogCPM().fit(train)
    reference_before = normalizer.reference_counts_.copy()
    calibration_before = normalizer.calibration_
    reference_id_before = normalizer.reference_sample_id_
    normalizer.transform(validation_counts(multiplier=100).drop(columns=["g4"]))
    assert np.array_equal(normalizer.reference_counts_, reference_before)
    assert normalizer.calibration_ == calibration_before
    assert normalizer.reference_sample_id_ == reference_id_before


def test_zero_variance_filter_depends_only_on_training():
    train = pd.DataFrame(
        {"variable": [1.0, 2.0, 3.0], "constant": [7.0, 7.0, 7.0]},
        index=["a", "b", "c"],
    )
    validation = pd.DataFrame(
        {"variable": [5.0], "constant": [999.0]}, index=["new"]
    )
    filter_ = ExactZeroVarianceFilter().fit(train)
    assert list(filter_.transform(validation).columns) == ["variable"]
    assert list(filter_.feature_names_out_) == ["variable"]


def test_standard_scaler_never_observes_validation_during_fit():
    train = pd.DataFrame({"a": [1.0, 3.0], "b": [10.0, 14.0]}, index=["t1", "t2"])
    validation = pd.DataFrame({"a": [10_000.0], "b": [-10_000.0]}, index=["v1"])
    scaler = TrainingStandardScaler().fit(train)
    expected_mean = train.mean(axis=0).to_numpy()
    before = scaler.mean_.copy()
    scaler.transform(validation)
    assert np.array_equal(before, expected_mean)
    assert np.array_equal(scaler.mean_, expected_mean)


def test_transform_restores_exact_fitted_feature_order():
    train = training_counts()
    fitted = permissive_preprocessor().fit(train)
    shuffled = validation_counts().loc[:, list(reversed(GENES))]
    transformed = fitted.transform(shuffled)
    assert list(transformed.columns) == list(fitted.get_feature_names_out())


def test_new_patient_transform_is_independent_of_other_test_patients():
    fitted = permissive_preprocessor().fit(training_counts())
    validation = validation_counts()
    alone = fitted.transform(validation.iloc[[0]])
    together = fitted.transform(validation)
    pd.testing.assert_frame_equal(alone, together.loc[["validation-a"]], check_exact=True)


def test_identical_composition_libraries_have_unit_tmm_factors():
    counts = pd.DataFrame(
        [[10, 20, 30], [20, 40, 60], [30, 60, 90]],
        index=["a", "b", "c"],
        columns=["g1", "g2", "g3"],
    )
    fitted = FrozenTMMLogCPM().fit(counts)
    assert fitted.training_factors_ == pytest.approx(np.ones(3))


def test_transform_rejects_missing_or_extra_genes():
    fitted = permissive_preprocessor().fit(training_counts())
    with pytest.raises(ValueError, match="gene schema differs"):
        fitted.transform(validation_counts().drop(columns=["g1"]))
