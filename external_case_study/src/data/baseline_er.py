"""Acquire and resolve baseline TCGA-BRCA ER from official Clinical Supplements."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .clinical import build_clinical_table
from .er_source_audit import (
    PATIENT_BIOTAB_NAME,
    _clinical_file_payload,
    _flatten_files,
    parse_clinical_xml,
    read_patient_biotab,
    resolve_primary_xml,
)
from .gdc_client import GDCClient
from .utils import release_slug, write_json_immutable


def acquire_baseline_er_sources(
    client: GDCClient,
    config: Mapping[str, Any],
    raw_dir: Path,
    status: Mapping[str, Any],
    workers: int = 16,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Download/verify XML and Biotab and return source-resolved baseline labels."""

    payload = _clinical_file_payload(str(config["project_id"]))
    response = client.search_all(
        str(config["gdc"]["files_endpoint"]), payload, int(config["gdc"]["page_size"])
    )
    # Keep the fingerprint identical to the independently reviewed source audit.
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    snapshot = raw_dir / release_slug(status) / f"er_source_audit_{fingerprint}"
    write_json_immutable(snapshot / "metadata" / "status.json", status)
    write_json_immutable(snapshot / "metadata" / "clinical_supplement_query.json", payload)
    write_json_immutable(snapshot / "metadata" / "clinical_supplement_response.json", response)

    files = _flatten_files(response["data"]["hits"])
    xml_files = files.loc[files["data_format"].eq("BCR XML")].copy()
    if xml_files["case_id"].duplicated().any():
        raise ValueError("Multiple released BCR XML files map to one case")
    biotab_files = files.loc[files["file_name"].eq(PATIENT_BIOTAB_NAME)].drop_duplicates("file_id")
    if len(biotab_files) != 1:
        raise ValueError(f"Expected one patient BCR Biotab, found {len(biotab_files)}")

    def download(row: Mapping[str, Any], subdir: str) -> Path:
        destination = snapshot / subdir / str(row["file_id"]) / str(row["file_name"])
        local_client = GDCClient(
            base_url=str(config["gdc"]["api_base_url"]),
            timeout_seconds=int(config["gdc"]["timeout_seconds"]),
            max_retries=int(config["gdc"]["max_retries"]),
        )
        local_client.download_file(
            str(row["file_id"]), destination, str(row["md5sum"]), int(row["file_size"]),
            endpoint=str(config["gdc"]["data_endpoint"]),
        )
        return destination

    xml_rows = xml_files.to_dict("records")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        xml_paths = list(pool.map(lambda row: download(row, "clinical_xml"), xml_rows))
    biotab_row = biotab_files.iloc[0].to_dict()
    biotab_path = download(biotab_row, "biotab")

    observations = []
    for row, path in zip(xml_rows, xml_paths, strict=True):
        observations.extend(
            parse_clinical_xml(
                path,
                file_id=str(row["file_id"]),
                file_name=str(row["file_name"]),
                expected_case_id=str(row["case_id"]),
            )
        )
    xml_primary = resolve_primary_xml(observations)
    biotab = read_patient_biotab(biotab_path)
    comparison = xml_primary.merge(
        biotab, on=["case_id", "case_submitter_id"], how="outer", validate="one_to_one"
    )
    known_both = comparison[["xml_primary_label", "biotab_primary_label"]].notna().all(axis=1)
    mismatch = known_both & comparison["xml_primary_label"].ne(comparison["biotab_primary_label"])
    if mismatch.any():
        raise ValueError(
            "Baseline XML and BCR Biotab disagree for cases: "
            + ", ".join(comparison.loc[mismatch, "case_submitter_id"].astype(str))
        )
    comparison["xml_biotab_crosscheck"] = "neither_known"
    comparison.loc[known_both, "xml_biotab_crosscheck"] = "concordant"
    comparison.loc[
        comparison["xml_primary_label"].notna() & comparison["biotab_primary_label"].isna(),
        "xml_biotab_crosscheck",
    ] = "xml_only_known"
    comparison.loc[
        comparison["xml_primary_label"].isna() & comparison["biotab_primary_label"].notna(),
        "xml_biotab_crosscheck",
    ] = "biotab_only_known"
    context = {
        "snapshot_dir": str(snapshot.resolve()),
        "query_fingerprint": fingerprint,
        "bcr_xml_files": len(xml_files),
        "patient_biotab_file_id": str(biotab_row["file_id"]),
        "md5_verified_file_count": len(xml_files) + 1,
    }
    return comparison, pd.DataFrame(vars(item) for item in observations), context


def build_baseline_clinical_table(
    cases: list[Mapping[str, Any]],
    source_comparison: pd.DataFrame,
    clinical_config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create one baseline target per GDC case and retain secondary validation."""

    case_frame = pd.DataFrame(
        {"case_id": str(case["case_id"]), "case_submitter_id": str(case["submitter_id"])}
        for case in cases
    )
    harmonized, harmonized_records = build_clinical_table(cases, clinical_config)
    harmonized = harmonized.rename(
        columns={
            "er_label": "harmonized_er_label",
            "er_status": "harmonized_er_status",
            "er_exclusion_reason": "harmonized_er_exclusion_reason",
            "er_record_count": "harmonized_er_record_count",
            "er_raw_values": "harmonized_er_raw_values",
            "er_molecular_test_ids": "harmonized_er_molecular_test_ids",
        }
    )
    table = case_frame.merge(
        source_comparison, on=["case_id", "case_submitter_id"], how="left", validate="one_to_one"
    ).merge(harmonized, on=["case_id", "case_submitter_id"], how="left", validate="one_to_one")
    table["er_label"] = table["xml_primary_label"]
    table["er_status"] = table["er_label"].map({1: "ER_positive", 0: "ER_negative"})
    table["er_exclusion_reason"] = table["xml_primary_reason"].fillna("xml_file_missing")
    table.loc[table["er_label"].notna(), "er_exclusion_reason"] = None
    table["er_source"] = "TCGA Clinical Supplement BCR XML"
    table["er_source_field"] = "er_status_by_ihc"
    table["er_source_cde"] = "2957359"
    both = table[["er_label", "harmonized_er_label"]].notna().all(axis=1)
    table["harmonized_validation"] = "neither_known"
    table.loc[both & table["er_label"].eq(table["harmonized_er_label"]), "harmonized_validation"] = "concordant"
    table.loc[both & table["er_label"].ne(table["harmonized_er_label"]), "harmonized_validation"] = "discordant"
    table.loc[table["er_label"].notna() & table["harmonized_er_label"].isna(), "harmonized_validation"] = "baseline_only_known"
    table.loc[table["er_label"].isna() & table["harmonized_er_label"].notna(), "harmonized_validation"] = "harmonized_only_known"
    return table.sort_values("case_submitter_id").reset_index(drop=True), harmonized_records
