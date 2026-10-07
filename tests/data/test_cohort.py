import pandas as pd

from src.data.cohort import flatten_file_hits, select_patient_files


def candidates():
    return pd.DataFrame(
        [
            {
                "case_id": "case-1",
                "file_id": "file-z",
                "sample_id": "sample-z",
                "sample_submitter_id": "TCGA-AA-0001-01B",
                "portion_number": "04",
                "aliquot_submitter_id": "aliquot-z",
            },
            {
                "case_id": "case-1",
                "file_id": "file-a",
                "sample_id": "sample-a",
                "sample_submitter_id": "TCGA-AA-0001-01A",
                "portion_number": "11",
                "aliquot_submitter_id": "aliquot-a",
            },
            {
                "case_id": "case-2",
                "file_id": "file-2",
                "sample_id": "sample-2",
                "sample_submitter_id": "TCGA-AA-0002-01A",
                "portion_number": "1",
                "aliquot_submitter_id": "aliquot-2",
            },
        ]
    )


SELECTION_CONFIG = {
    "strategy": "tcga_barcode_lexicographic",
    "sort_keys": [
        "sample_submitter_id",
        "portion_number",
        "aliquot_submitter_id",
        "sample_id",
        "file_id",
    ],
}


def test_sample_selection_is_deterministic_and_patient_level():
    first, audit_first = select_patient_files(candidates(), SELECTION_CONFIG)
    second, audit_second = select_patient_files(
        candidates().sample(frac=1, random_state=99), SELECTION_CONFIG
    )
    assert first["file_id"].tolist() == ["file-a", "file-2"]
    assert second["file_id"].tolist() == first["file_id"].tolist()
    assert audit_first.loc[audit_first.case_id == "case-1", "case_candidate_count"].eq(2).all()
    pd.testing.assert_frame_equal(audit_first, audit_second)


def test_flatten_preserves_multiple_primary_tumor_candidates():
    hits = [
        {
            "file_id": "file-1",
            "file_name": "counts.tsv",
            "file_size": 10,
            "md5sum": "abc",
            "analysis": {"workflow_type": "STAR - Counts"},
            "cases": [
                {
                    "case_id": "case-1",
                    "submitter_id": "TCGA-AA-0001",
                    "samples": [
                        {
                            "sample_id": "sample-1",
                            "submitter_id": "TCGA-AA-0001-01A",
                            "sample_type": "Primary Tumor",
                            "portions": [
                                {
                                    "portion_number": "11",
                                    "analytes": [
                                        {
                                            "analyte_type": "RNA",
                                            "aliquots": [
                                                {
                                                    "aliquot_id": "aliquot-1",
                                                    "submitter_id": "TCGA-AA-0001-01A-11R-X",
                                                }
                                            ],
                                        }
                                    ],
                                }
                            ],
                        },
                        {
                            "sample_id": "normal-1",
                            "submitter_id": "TCGA-AA-0001-11A",
                            "sample_type": "Solid Tissue Normal",
                        },
                    ],
                }
            ],
        }
    ]
    result = flatten_file_hits(hits)
    assert len(result) == 1
    assert result.iloc[0]["sample_id"] == "sample-1"
    assert result.iloc[0]["aliquot_id"] == "aliquot-1"


def test_unknown_selection_strategy_is_rejected():
    config = {**SELECTION_CONFIG, "strategy": "largest_file"}
    try:
        select_patient_files(candidates(), config)
    except ValueError as exc:
        assert "strategy" in str(exc)
    else:
        raise AssertionError("Expected unsupported strategy to fail")
