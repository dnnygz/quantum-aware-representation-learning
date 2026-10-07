import numpy as np
import pandas as pd
import pytest

from src.data.validation import CohortValidationError, validate_cohort


def valid_frames():
    expression = pd.DataFrame(
        [[1, 2], [3, 4]], index=pd.Index(["case-1", "case-2"], name="case_id"), columns=["g1", "g2"]
    )
    labels = pd.DataFrame(
        {"case_id": ["case-1", "case-2"], "er_label": [0, 1]}
    )
    metadata = pd.DataFrame(
        {
            "case_id": ["case-1", "case-2"],
            "sample_id": ["sample-1", "sample-2"],
            "file_id": ["file-1", "file-2"],
        }
    )
    genes = pd.DataFrame({"gene_id": ["g1", "g2"], "gene_name": ["A", "B"]})
    return expression, labels, metadata, genes


def test_valid_cohort_passes_all_checks():
    report = validate_cohort(*valid_frames())
    assert report["status"] == "pass"
    assert all(check["passed"] for check in report["checks"])


def test_misaligned_join_fails():
    expression, labels, metadata, genes = valid_frames()
    labels = labels.iloc[::-1].reset_index(drop=True)
    with pytest.raises(CohortValidationError, match="case_alignment"):
        validate_cohort(expression, labels, metadata, genes)


def test_nan_and_infinity_fail():
    expression, labels, metadata, genes = valid_frames()
    expression = expression.astype(float)
    expression.iloc[0, 0] = np.nan
    expression.iloc[1, 1] = np.inf
    with pytest.raises(CohortValidationError):
        validate_cohort(expression, labels, metadata, genes)
