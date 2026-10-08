import pandas as pd

from src.data.baseline_er import build_baseline_clinical_table


CONFIG = {
    "er_gene_symbol": "ESR1",
    "er_method": "IHC",
    "positive_values": ["Positive"],
    "negative_values": ["Negative"],
    "conflict_policy": "exclude",
}


def _case(case_id: str, barcode: str, result: str | None) -> dict:
    tests = [] if result is None else [
        {
            "molecular_test_id": f"test-{case_id}",
            "gene_symbol": "ESR1",
            "molecular_analysis_method": "IHC",
            "test_result": result,
        }
    ]
    return {
        "case_id": case_id,
        "submitter_id": barcode,
        "follow_ups": [{"follow_up_id": f"follow-{case_id}", "molecular_tests": tests}],
    }


def test_baseline_xml_is_primary_and_later_harmonized_result_does_not_impute() -> None:
    cases = [
        _case("case-1", "TCGA-XX-0001", "Negative"),
        _case("case-2", "TCGA-XX-0002", "Negative"),
    ]
    sources = pd.DataFrame(
        [
            {
                "case_id": "case-1",
                "case_submitter_id": "TCGA-XX-0001",
                "xml_primary_label": 1,
                "xml_primary_reason": None,
                "biotab_primary_label": 1,
            },
            {
                "case_id": "case-2",
                "case_submitter_id": "TCGA-XX-0002",
                "xml_primary_label": None,
                "xml_primary_reason": "xml_primary_unknown",
                "biotab_primary_label": None,
            },
        ]
    )

    labels, _ = build_baseline_clinical_table(cases, sources, CONFIG)
    first = labels.set_index("case_id").loc["case-1"]
    second = labels.set_index("case_id").loc["case-2"]

    assert first["er_label"] == 1
    assert first["harmonized_validation"] == "discordant"
    assert pd.isna(second["er_label"])
    assert second["harmonized_validation"] == "harmonized_only_known"
    assert second["er_exclusion_reason"] == "xml_primary_unknown"
