"""Strict extraction and resolution of clinical estrogen-receptor status."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Any, Iterable, Mapping

import pandas as pd


DEFAULT_POSITIVE = {"positive", "er positive", "er+"}
DEFAULT_NEGATIVE = {"negative", "er negative", "er-"}


def _normalise_text(value: Any) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).strip().split()).casefold()
    return text or None


def parse_er_status(
    value: Any,
    positive_values: Iterable[str] = DEFAULT_POSITIVE,
    negative_values: Iterable[str] = DEFAULT_NEGATIVE,
) -> int | None:
    """Map an original ER value to {0, 1}; never impute unknown values."""

    normalised = _normalise_text(value)
    positive = {_normalise_text(item) for item in positive_values}
    negative = {_normalise_text(item) for item in negative_values}
    if normalised in positive:
        return 1
    if normalised in negative:
        return 0
    return None


@dataclass(frozen=True)
class ERRecord:
    case_id: str
    case_submitter_id: str
    molecular_test_id: str | None
    molecular_test_submitter_id: str | None
    follow_up_id: str | None
    gene_symbol: str | None
    method: str | None
    raw_result: str | None
    days_to_test: int | None
    created_datetime: str | None
    updated_datetime: str | None


def extract_er_records(
    case: Mapping[str, Any], gene_symbol: str = "ESR1", method: str = "IHC"
) -> list[ERRecord]:
    """Extract configured ESR1 assay records from the harmonized case graph."""

    records: list[ERRecord] = []
    target_gene = gene_symbol.casefold()
    target_method = method.casefold()
    for follow_up in case.get("follow_ups") or []:
        for test in follow_up.get("molecular_tests") or []:
            test_gene = _normalise_text(test.get("gene_symbol"))
            test_method = _normalise_text(test.get("molecular_analysis_method"))
            if test_gene != target_gene or test_method != target_method:
                continue
            records.append(
                ERRecord(
                    case_id=str(case["case_id"]),
                    case_submitter_id=str(case["submitter_id"]),
                    molecular_test_id=test.get("molecular_test_id"),
                    molecular_test_submitter_id=test.get("submitter_id"),
                    follow_up_id=follow_up.get("follow_up_id"),
                    gene_symbol=test.get("gene_symbol"),
                    method=test.get("molecular_analysis_method"),
                    raw_result=test.get("test_result"),
                    days_to_test=test.get("days_to_test"),
                    created_datetime=test.get("created_datetime"),
                    updated_datetime=test.get("updated_datetime"),
                )
            )
    return records


def resolve_case_er(
    case: Mapping[str, Any], clinical_config: Mapping[str, Any]
) -> dict[str, Any]:
    """Resolve a case-level label conservatively and retain the audit evidence."""

    if clinical_config.get("conflict_policy") != "exclude":
        raise ValueError("Only the conservative ER conflict_policy='exclude' is supported")

    records = extract_er_records(
        case,
        gene_symbol=str(clinical_config["er_gene_symbol"]),
        method=str(clinical_config["er_method"]),
    )
    positive = clinical_config["positive_values"]
    negative = clinical_config["negative_values"]
    parsed = [parse_er_status(record.raw_result, positive, negative) for record in records]
    known = {label for label in parsed if label is not None}
    unknown_present = any(label is None for label in parsed)

    if not records:
        label = None
        reason = "no_er_ihc_record"
    elif not known:
        label = None
        reason = "er_ambiguous_or_unknown"
    elif len(known) > 1:
        label = None
        reason = "conflicting_er_results"
    elif unknown_present:
        label = None
        reason = "known_and_ambiguous_er_results"
    else:
        label = next(iter(known))
        reason = None

    return {
        "case_id": str(case["case_id"]),
        "case_submitter_id": str(case["submitter_id"]),
        "er_label": label,
        "er_status": "ER_positive" if label == 1 else "ER_negative" if label == 0 else None,
        "er_exclusion_reason": reason,
        "er_record_count": len(records),
        "er_raw_values": json.dumps([record.raw_result for record in records]),
        "er_molecular_test_ids": json.dumps([record.molecular_test_id for record in records]),
    }


def build_clinical_table(
    cases: Iterable[Mapping[str, Any]], clinical_config: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one-row-per-case labels and one-row-per-test audit tables."""

    cases = list(cases)
    labels = pd.DataFrame(resolve_case_er(case, clinical_config) for case in cases)
    record_rows = [
        asdict(record)
        for case in cases
        for record in extract_er_records(
            case,
            gene_symbol=str(clinical_config["er_gene_symbol"]),
            method=str(clinical_config["er_method"]),
        )
    ]
    records = pd.DataFrame(record_rows, columns=[field.name for field in ERRecord.__dataclass_fields__.values()])
    if not labels.empty:
        labels = labels.sort_values("case_submitter_id").reset_index(drop=True)
    return labels, records
