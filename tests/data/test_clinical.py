import pytest

from src.data.clinical import parse_er_status, resolve_case_er


CLINICAL_CONFIG = {
    "er_gene_symbol": "ESR1",
    "er_method": "IHC",
    "positive_values": ["Positive"],
    "negative_values": ["Negative"],
    "ambiguous_values": ["Equivocal", "Unknown"],
    "conflict_policy": "exclude",
}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Positive", 1),
        (" positive ", 1),
        ("Negative", 0),
        ("NEGATIVE", 0),
        ("Equivocal", None),
        ("Not Evaluated", None),
        (None, None),
        ("", None),
    ],
)
def test_parse_er_status_is_strict(raw, expected):
    assert parse_er_status(raw, ["Positive"], ["Negative"]) == expected


def make_case(*results):
    return {
        "case_id": "case-1",
        "submitter_id": "TCGA-AA-0001",
        "follow_ups": [
            {
                "follow_up_id": f"fu-{index}",
                "molecular_tests": [
                    {
                        "molecular_test_id": f"test-{index}",
                        "submitter_id": f"submitted-{index}",
                        "gene_symbol": "ESR1",
                        "molecular_analysis_method": "IHC",
                        "test_result": result,
                    }
                ],
            }
            for index, result in enumerate(results)
        ],
    }


def test_duplicate_concordant_results_are_retained():
    resolved = resolve_case_er(make_case("Positive", "Positive"), CLINICAL_CONFIG)
    assert resolved["er_label"] == 1
    assert resolved["er_record_count"] == 2
    assert resolved["er_exclusion_reason"] is None


def test_conflicting_results_are_excluded_not_majority_voted():
    resolved = resolve_case_er(
        make_case("Negative", "Positive", "Negative"), CLINICAL_CONFIG
    )
    assert resolved["er_label"] is None
    assert resolved["er_exclusion_reason"] == "conflicting_er_results"


def test_known_plus_ambiguous_is_excluded():
    resolved = resolve_case_er(make_case("Positive", "Equivocal"), CLINICAL_CONFIG)
    assert resolved["er_label"] is None
    assert resolved["er_exclusion_reason"] == "known_and_ambiguous_er_results"


def test_non_esr1_record_does_not_become_er_label():
    case = make_case("Positive")
    case["follow_ups"][0]["molecular_tests"][0]["gene_symbol"] = "PGR"
    resolved = resolve_case_er(case, CLINICAL_CONFIG)
    assert resolved["er_label"] is None
    assert resolved["er_exclusion_reason"] == "no_er_ihc_record"


def test_nonconservative_conflict_policy_is_rejected():
    config = {**CLINICAL_CONFIG, "conflict_policy": "majority_vote"}
    with pytest.raises(ValueError, match="conflict_policy"):
        resolve_case_er(make_case("Positive"), config)
