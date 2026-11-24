"""Audit TCGA-BRCA ER labels across official GDC representations.

This module is deliberately read-only with respect to ``data/processed``.  It
compares the harmonized GDC case graph with the primary diagnostic field in
TCGA BCR Clinical Supplement XML and its aggregate BCR Biotab rendering.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from .clinical import build_clinical_table, parse_er_status
from .gdc_client import GDCClient
from .utils import (
    configure_logging,
    load_config,
    release_slug,
    resolve_path,
    sha256_file,
    utc_now,
    write_json_atomic,
    write_json_immutable,
)


LOGGER = logging.getLogger(__name__)
PRIMARY_ER_LOCAL_NAME = "breast_carcinoma_estrogen_receptor_status"
METASTATIC_ER_LOCAL_NAME = "metastatic_breast_carcinoma_estrogen_receptor_status"
ER_LOCAL_NAMES = {PRIMARY_ER_LOCAL_NAME, METASTATIC_ER_LOCAL_NAME}
PATIENT_BIOTAB_NAME = "nationwidechildrens.org_clinical_patient_brca.txt"


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = " ".join(value.split())
    return value or None


@dataclass(frozen=True)
class XMLERObservation:
    case_id: str
    case_submitter_id: str
    file_id: str
    file_name: str
    value: str | None
    label: int | None
    context: str
    xml_path: str
    cde: str | None
    preferred_name: str | None
    procurement_status: str | None


def parse_clinical_xml(
    path: Path, *, file_id: str, file_name: str, expected_case_id: str | None = None
) -> list[XMLERObservation]:
    """Extract every ER field and distinguish the direct patient diagnostic field.

    The primary field is *only* a direct child of ``patient``.  Nested fields in
    ``follow_up`` or ``new_tumor_event`` are retained for audit, never silently
    collapsed into the baseline label.
    """

    root = ET.parse(path).getroot()
    patient = next((node for node in root.iter() if _local_name(node.tag) == "patient"), None)
    if patient is None:
        raise ValueError(f"No patient element in {path}")

    direct = {_local_name(node.tag): _clean(node.text) for node in patient}
    case_id = direct.get("bcr_patient_uuid")
    submitter_id = direct.get("bcr_patient_barcode")
    if not case_id or not submitter_id:
        raise ValueError(f"Missing patient UUID/barcode in {path}")
    case_id = case_id.lower()
    if expected_case_id and case_id != expected_case_id.lower():
        raise ValueError(f"Case UUID mismatch for {path}: {case_id} != {expected_case_id}")

    observations: list[XMLERObservation] = []

    def visit(node: ET.Element, ancestors: tuple[str, ...]) -> None:
        name = _local_name(node.tag)
        current_path = (*ancestors, name)
        if name in ER_LOCAL_NAMES:
            is_primary = (
                name == PRIMARY_ER_LOCAL_NAME
                and len(ancestors) > 0
                and ancestors[-1] == "patient"
            )
            if is_primary:
                context = "primary_patient"
            elif "new_tumor_event" in ancestors:
                context = "new_tumor_event"
            elif "follow_up" in ancestors:
                context = "follow_up"
            elif name == METASTATIC_ER_LOCAL_NAME and ancestors[-1:] == ("patient",):
                context = "new_tumor_event_patient_template"
            else:
                context = "other_nested"
            value = _clean(node.text)
            observations.append(
                XMLERObservation(
                    case_id=case_id,
                    case_submitter_id=submitter_id,
                    file_id=file_id,
                    file_name=file_name,
                    value=value,
                    label=parse_er_status(value),
                    context=context,
                    xml_path="/".join(current_path),
                    cde=node.attrib.get("cde"),
                    preferred_name=node.attrib.get("preferred_name"),
                    procurement_status=node.attrib.get("procurement_status"),
                )
            )
        for child in node:
            visit(child, current_path)

    visit(patient, ())
    return observations


def resolve_primary_xml(observations: Iterable[XMLERObservation]) -> pd.DataFrame:
    """Resolve direct-patient XML values conservatively to one row per case."""

    rows: list[dict[str, Any]] = []
    grouped: dict[str, list[XMLERObservation]] = {}
    for observation in observations:
        grouped.setdefault(observation.case_id, []).append(observation)
    for case_id, case_observations in grouped.items():
        primary = [item for item in case_observations if item.context == "primary_patient"]
        known = {item.label for item in primary if item.label is not None}
        unknown = any(item.label is None for item in primary)
        if not primary:
            label, reason = None, "xml_primary_field_missing"
        elif not known:
            label, reason = None, "xml_primary_unknown"
        elif len(known) > 1:
            label, reason = None, "xml_primary_conflict"
        elif unknown:
            label, reason = None, "xml_primary_known_and_unknown"
        else:
            label, reason = next(iter(known)), None
        exemplar = primary[0] if primary else case_observations[0]
        rows.append(
            {
                "case_id": case_id,
                "case_submitter_id": exemplar.case_submitter_id,
                "xml_primary_label": label,
                "xml_primary_raw_values": json.dumps([item.value for item in primary]),
                "xml_primary_reason": reason,
                "xml_primary_record_count": len(primary),
                "xml_nested_er_record_count": len(case_observations) - len(primary),
            }
        )
    return pd.DataFrame(rows).sort_values("case_submitter_id").reset_index(drop=True)


def read_patient_biotab(path: Path) -> pd.DataFrame:
    """Read the TCGA patient Biotab, whose first three rows describe the schema."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    if len(rows) < 4:
        raise ValueError(f"Unexpectedly short BCR Biotab: {path}")
    header = rows[0]
    required = {"bcr_patient_uuid", "bcr_patient_barcode", "er_status_by_ihc"}
    if not required.issubset(header):
        raise ValueError(f"BCR Biotab lacks fields: {sorted(required - set(header))}")
    records = [dict(zip(header, row, strict=False)) for row in rows[3:] if row]
    frame = pd.DataFrame(records)
    frame["case_id"] = frame["bcr_patient_uuid"].str.lower()
    frame["case_submitter_id"] = frame["bcr_patient_barcode"]
    frame["biotab_primary_raw"] = frame["er_status_by_ihc"]
    frame["biotab_primary_label"] = frame["biotab_primary_raw"].map(parse_er_status)
    return frame[["case_id", "case_submitter_id", "biotab_primary_raw", "biotab_primary_label"]]


def classify_comparison(harmonized: Any, xml: Any) -> str:
    h_known, x_known = pd.notna(harmonized), pd.notna(xml)
    if h_known and x_known:
        return "concordant" if int(harmonized) == int(xml) else "discordant"
    if h_known:
        return "harmonized_only_known"
    if x_known:
        return "xml_only_known"
    return "neither_known"


def _clinical_file_payload(project_id: str) -> dict[str, Any]:
    return {
        "filters": {
            "op": "and",
            "content": [
                {"op": "=", "content": {"field": "cases.project.project_id", "value": [project_id]}},
                {"op": "=", "content": {"field": "data_category", "value": ["Clinical"]}},
                {"op": "=", "content": {"field": "data_type", "value": ["Clinical Supplement"]}},
                {"op": "=", "content": {"field": "access", "value": ["open"]}},
            ],
        },
        "fields": ",".join(
            [
                "file_id", "file_name", "file_size", "md5sum", "data_format", "data_type",
                "state", "created_datetime", "updated_datetime", "cases.case_id", "cases.submitter_id",
            ]
        ),
        "format": "JSON",
    }


def _flatten_files(hits: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for hit in hits:
        for case in hit.get("cases") or [{}]:
            rows.append(
                {
                    "file_id": hit.get("file_id"), "file_name": hit.get("file_name"),
                    "file_size": hit.get("file_size"), "md5sum": hit.get("md5sum"),
                    "data_format": hit.get("data_format"), "case_id": case.get("case_id"),
                    "case_submitter_id": case.get("submitter_id"),
                }
            )
    return pd.DataFrame(rows)


def _processed_hashes(processed_dir: Path) -> dict[str, str]:
    return {path.name: sha256_file(path) for path in sorted(processed_dir.glob("*")) if path.is_file()}


def run_audit(config_path: Path, workers: int = 12) -> dict[str, Any]:
    config, repo_root = load_config(config_path)
    raw_dir = resolve_path(repo_root, config["paths"]["raw_dir"])
    interim_dir = resolve_path(repo_root, config["paths"]["interim_dir"])
    processed_dir = resolve_path(repo_root, config["paths"]["processed_dir"])
    reports_dir = resolve_path(repo_root, config["paths"]["reports_dir"])
    before_hashes = _processed_hashes(processed_dir)

    client = GDCClient(
        base_url=config["gdc"]["api_base_url"],
        timeout_seconds=int(config["gdc"]["timeout_seconds"]),
        max_retries=int(config["gdc"]["max_retries"]),
    )
    status = client.status(config["gdc"]["status_endpoint"])
    payload = _clinical_file_payload(config["project_id"])
    response = client.search_all(config["gdc"]["files_endpoint"], payload, int(config["gdc"]["page_size"]))
    hits = response["data"]["hits"]
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    snapshot = raw_dir / release_slug(status) / f"er_source_audit_{fingerprint}"
    write_json_immutable(snapshot / "metadata" / "status.json", status)
    write_json_immutable(snapshot / "metadata" / "clinical_supplement_query.json", payload)
    write_json_immutable(snapshot / "metadata" / "clinical_supplement_response.json", response)

    files = _flatten_files(hits)
    xml_files = files.loc[files["data_format"].eq("BCR XML")].copy()
    if xml_files["case_id"].duplicated().any():
        raise ValueError("Multiple released BCR XML files map to one case; version resolution required")
    biotab = files.loc[files["file_name"].eq(PATIENT_BIOTAB_NAME)].drop_duplicates("file_id")
    if len(biotab) != 1:
        raise ValueError(f"Expected one patient BCR Biotab, found {len(biotab)}")

    def download(row: Mapping[str, Any], subdir: str) -> Path:
        destination = snapshot / subdir / str(row["file_id"]) / str(row["file_name"])
        local_client = GDCClient(
            base_url=config["gdc"]["api_base_url"],
            timeout_seconds=int(config["gdc"]["timeout_seconds"]),
            max_retries=int(config["gdc"]["max_retries"]),
        )
        local_client.download_file(
            str(row["file_id"]), destination, str(row["md5sum"]), int(row["file_size"]),
            endpoint=config["gdc"]["data_endpoint"],
        )
        return destination

    xml_rows = xml_files.to_dict("records")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        xml_paths = list(pool.map(lambda row: download(row, "clinical_xml"), xml_rows))
    biotab_row = biotab.iloc[0].to_dict()
    biotab_path = download(biotab_row, "biotab")

    observations: list[XMLERObservation] = []
    for row, path in zip(xml_rows, xml_paths, strict=True):
        observations.extend(
            parse_clinical_xml(
                path, file_id=str(row["file_id"]), file_name=str(row["file_name"]),
                expected_case_id=str(row["case_id"]),
            )
        )
    xml_primary = resolve_primary_xml(observations)
    biotab_primary = read_patient_biotab(biotab_path)

    clinical_response = raw_dir / release_slug(status) / "metadata" / "clinical_response.json"
    cases = json.loads(clinical_response.read_text(encoding="utf-8"))["data"]["hits"]
    harmonized, _ = build_clinical_table(cases, config["clinical"])
    comparison = harmonized.rename(
        columns={"er_label": "harmonized_label", "er_exclusion_reason": "harmonized_reason"}
    ).merge(xml_primary, on=["case_id", "case_submitter_id"], how="outer", validate="one_to_one")
    comparison = comparison.merge(
        biotab_primary, on=["case_id", "case_submitter_id"], how="outer", validate="one_to_one"
    )
    comparison["harmonized_vs_xml"] = [
        classify_comparison(h, x)
        for h, x in zip(comparison["harmonized_label"], comparison["xml_primary_label"], strict=True)
    ]
    comparison["xml_vs_biotab"] = [
        classify_comparison(x, b)
        for x, b in zip(comparison["xml_primary_label"], comparison["biotab_primary_label"], strict=True)
    ]
    labels = pd.read_parquet(processed_dir / "labels.parquet")
    frozen_ids = set(labels["case_id"].astype(str))
    selection_audit = pd.read_parquet(interim_dir / "sample_selection_audit.parquet")
    rna_ids = set(selection_audit["case_id"].astype(str))
    comparison["in_frozen_primary_cohort"] = comparison["case_id"].isin(frozen_ids)
    comparison = comparison.sort_values("case_submitter_id").reset_index(drop=True)

    interim_path = interim_dir / "er_source_comparison.parquet"
    temporary = interim_path.with_suffix(".parquet.part")
    comparison.to_parquet(temporary, index=False)
    os.replace(temporary, interim_path)
    observation_path = interim_dir / "er_xml_observations.parquet"
    observation_frame = pd.DataFrame(asdict(item) for item in observations)
    temporary = observation_path.with_suffix(".parquet.part")
    observation_frame.to_parquet(temporary, index=False)
    os.replace(temporary, observation_path)

    categories = comparison["harmonized_vs_xml"].value_counts(dropna=False).to_dict()
    frozen = comparison.loc[comparison["in_frozen_primary_cohort"]]
    conflict_cases = comparison.loc[comparison["harmonized_reason"].eq("conflicting_er_results")]
    xml_only_rna = comparison.loc[
        comparison["harmonized_vs_xml"].eq("xml_only_known") & comparison["case_id"].isin(rna_ids)
    ]
    harmonized_only_rna = comparison.loc[
        comparison["harmonized_vs_xml"].eq("harmonized_only_known") & comparison["case_id"].isin(rna_ids)
    ]
    report = {
        "generated_at": utc_now(),
        "project_id": config["project_id"],
        "gdc_release": status.get("data_release_version"),
        "scope": {
            "project_cases": len(cases), "clinical_supplement_files": int(files["file_id"].nunique()),
            "bcr_xml_files": len(xml_files), "bcr_xml_cases": int(xml_files["case_id"].nunique()),
            "bcr_biotab_files": int(files.loc[files["data_format"].eq("BCR Biotab"), "file_id"].nunique()),
            "bcr_omf_xml_files": int(files.loc[files["data_format"].eq("BCR OMF XML"), "file_id"].nunique()),
        },
        "retrieval": {
            "files_endpoint": f"{config['gdc']['api_base_url']}{config['gdc']['files_endpoint']}",
            "data_endpoint": f"{config['gdc']['api_base_url']}{config['gdc']['data_endpoint']}",
            "query_fingerprint": fingerprint,
            "downloaded_files_md5_verified": len(xml_files) + 1,
        },
        "patient_level_comparison": {
            "harmonized_known": int(comparison["harmonized_label"].notna().sum()),
            "xml_primary_known": int(comparison["xml_primary_label"].notna().sum()),
            "biotab_primary_known": int(comparison["biotab_primary_label"].notna().sum()),
            "harmonized_distribution": {
                str(int(k)): int(v)
                for k, v in comparison["harmonized_label"].value_counts().to_dict().items()
            },
            "xml_primary_distribution": {
                str(int(k)): int(v)
                for k, v in comparison["xml_primary_label"].value_counts().to_dict().items()
            },
            "categories": {str(k): int(v) for k, v in categories.items()},
            "known_in_both": int(comparison[["harmonized_label", "xml_primary_label"]].notna().all(axis=1).sum()),
            "concordance_among_known_both": float(
                (comparison["harmonized_vs_xml"] == "concordant").sum()
                / max(1, comparison[["harmonized_label", "xml_primary_label"]].notna().all(axis=1).sum())
            ),
        },
        "frozen_cohort_comparison": {
            "cases": len(frozen),
            "concordant": int(frozen["harmonized_vs_xml"].eq("concordant").sum()),
            "discordant": int(frozen["harmonized_vs_xml"].eq("discordant").sum()),
            "xml_missing": int(frozen["xml_primary_label"].isna().sum()),
        },
        "harmonized_conflicts": {
            "cases": len(conflict_cases),
            "xml_primary_known": int(conflict_cases["xml_primary_label"].notna().sum()),
            "xml_primary_values": conflict_cases[
                ["case_id", "case_submitter_id", "er_raw_values", "xml_primary_raw_values"]
            ].to_dict("records"),
        },
        "harmonized_only_known": comparison.loc[
            comparison["harmonized_vs_xml"].eq("harmonized_only_known"),
            ["case_id", "case_submitter_id", "er_raw_values", "xml_primary_raw_values", "biotab_primary_raw"],
        ].to_dict("records"),
        "xml_vs_biotab_categories": {
            str(k): int(v) for k, v in comparison["xml_vs_biotab"].value_counts().to_dict().items()
        },
        "potential_cohort_impact_not_applied": {
            "xml_only_known_with_eligible_rna": len(xml_only_rna),
            "harmonized_only_known_with_eligible_rna": len(harmonized_only_rna),
            "projected_case_count_if_baseline_xml_were_adopted": len(labels) + len(xml_only_rna) - len(harmonized_only_rna),
            "projected_er_positive_if_baseline_xml_were_adopted": int(labels["er_label"].eq(1).sum())
            + int(xml_only_rna["xml_primary_label"].eq(1).sum())
            - int(harmonized_only_rna["harmonized_label"].eq(1).sum()),
            "projected_er_negative_if_baseline_xml_were_adopted": int(labels["er_label"].eq(0).sum())
            + int(xml_only_rna["xml_primary_label"].eq(0).sum())
            - int(harmonized_only_rna["harmonized_label"].eq(0).sum()),
            "applied": False,
        },
        "xml_context_counts": {str(k): int(v) for k, v in Counter(item.context for item in observations).items()},
        "frozen_processed_sha256_before": before_hashes,
        "frozen_processed_sha256_after": _processed_hashes(processed_dir),
        "artifacts": {
            "comparison": str(interim_path.relative_to(repo_root)),
            "xml_observations": str(observation_path.relative_to(repo_root)),
            "raw_snapshot": str(snapshot.relative_to(repo_root)),
        },
    }
    report["frozen_cohort_unchanged"] = (
        report["frozen_processed_sha256_before"] == report["frozen_processed_sha256_after"]
    )
    write_json_atomic(reports_dir / "er_source_audit.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)
    report = run_audit(args.config, args.workers)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
