"""Acquire release-scoped TCGA-BRCA metadata and raw STAR Counts files."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .clinical import build_clinical_table
from .cohort import flatten_file_hits, select_patient_files
from .gdc_client import GDCClient
from .utils import (
    canonical_json_bytes,
    configure_logging,
    load_config,
    release_slug,
    resolve_path,
    utc_now,
    write_json_atomic,
    write_json_immutable,
)


LOGGER = logging.getLogger(__name__)


FILE_FIELDS = [
    "file_id",
    "file_name",
    "file_size",
    "md5sum",
    "data_category",
    "data_type",
    "experimental_strategy",
    "access",
    "state",
    "created_datetime",
    "updated_datetime",
    "analysis.workflow_type",
    "cases.case_id",
    "cases.submitter_id",
    "cases.samples.sample_id",
    "cases.samples.submitter_id",
    "cases.samples.sample_type",
    "cases.samples.created_datetime",
    "cases.samples.preservation_method",
    "cases.samples.portions.portion_number",
    "cases.samples.portions.analytes.analyte_type",
    "cases.samples.portions.analytes.aliquots.aliquot_id",
    "cases.samples.portions.analytes.aliquots.submitter_id",
]


def equality_filter(field: str, value: str) -> dict[str, Any]:
    return {"op": "=", "content": {"field": field, "value": [value]}}


def build_file_query(config: dict[str, Any]) -> dict[str, Any]:
    query = config["query"]
    filters = [
        equality_filter("cases.project.project_id", config["project_id"]),
        equality_filter("data_category", query["data_category"]),
        equality_filter("data_type", query["data_type"]),
        equality_filter("experimental_strategy", query["experimental_strategy"]),
        equality_filter("analysis.workflow_type", query["workflow_type"]),
        equality_filter("cases.samples.sample_type", query["sample_type"]),
        equality_filter("access", query["access"]),
    ]
    return {
        "filters": {"op": "and", "content": filters},
        "fields": ",".join(FILE_FIELDS),
        "format": "JSON",
    }


def build_case_query(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "filters": equality_filter("project.project_id", config["project_id"]),
        "fields": "case_id,submitter_id,project.project_id,created_datetime,updated_datetime,state",
        "expand": "follow_ups,follow_ups.molecular_tests",
        "format": "JSON",
    }


def acquire_metadata(
    client: GDCClient, config: dict[str, Any], raw_dir: Path, interim_dir: Path
) -> tuple[Path, list[dict[str, Any]]]:
    """Query live metadata and create/reuse an immutable release snapshot."""

    endpoints = config["gdc"]
    status = client.status(endpoints["status_endpoint"])
    file_query = build_file_query(config)
    case_query = build_case_query(config)
    query_fingerprint = hashlib.sha256(
        canonical_json_bytes({"files": file_query, "cases": case_query, "config": config})
    ).hexdigest()[:12]
    snapshot_dir = raw_dir / release_slug(status) / f"query_{query_fingerprint}"
    metadata_dir = snapshot_dir / "metadata"
    file_response = client.search_all(
        endpoints["files_endpoint"], file_query, int(endpoints["page_size"])
    )
    case_response = client.search_all(
        endpoints["cases_endpoint"], case_query, int(endpoints["page_size"])
    )

    write_json_immutable(metadata_dir / "gdc_status.json", status)
    write_json_immutable(metadata_dir / "file_query.json", file_query)
    write_json_immutable(metadata_dir / "case_query.json", case_query)
    write_json_immutable(metadata_dir / "file_response.json", file_response)
    write_json_immutable(metadata_dir / "clinical_response.json", case_response)
    write_json_immutable(metadata_dir / "configuration.json", config)

    candidates = flatten_file_hits(file_response["data"]["hits"], config["query"]["sample_type"])
    selected, audit = select_patient_files(candidates, config["sample_selection"])
    clinical_labels, _ = build_clinical_table(
        case_response["data"]["hits"], config["clinical"]
    )
    selected = selected.merge(
        clinical_labels[["case_id", "case_submitter_id", "er_label", "er_exclusion_reason"]],
        on=["case_id", "case_submitter_id"],
        how="left",
        validate="one_to_one",
    )
    selected = selected.loc[selected["er_label"].notna()].copy()
    selected["er_label"] = selected["er_label"].astype(int)
    interim_dir.mkdir(parents=True, exist_ok=True)
    audit.to_parquet(interim_dir / "sample_selection_audit.parquet", index=False)
    selected.to_parquet(interim_dir / "selected_download_manifest.parquet", index=False)
    clinical_labels.to_parquet(interim_dir / "clinical_labels_preview.parquet", index=False)
    selected_records = selected.to_dict("records")

    context = {
        "created_at": utc_now(),
        "snapshot_dir": str(snapshot_dir.resolve()),
        "release_slug": snapshot_dir.parent.name,
        "query_fingerprint": query_fingerprint,
        "metadata_only": True,
        "selected_file_count": len(selected_records),
    }
    write_json_atomic(interim_dir / "run_context.json", context)
    return snapshot_dir, selected_records


def download_selected_files(
    client: GDCClient,
    selected_records: list[dict[str, Any]],
    snapshot_dir: Path,
    data_endpoint: str,
    workers: int,
) -> list[dict[str, Any]]:
    def download(record: dict[str, Any]) -> dict[str, Any]:
        destination = snapshot_dir / "files" / record["file_id"] / record["file_name"]
        return client.download_file(
            file_id=record["file_id"],
            destination=destination,
            expected_md5=record["md5sum"],
            expected_size=int(record["file_size"]),
            endpoint=data_endpoint,
        )

    receipts: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download, record): record for record in selected_records}
        for completed, future in enumerate(as_completed(futures), start=1):
            record = futures[future]
            try:
                receipts.append(future.result())
            except Exception as exc:
                raise RuntimeError(f"Failed to acquire GDC file {record['file_id']}") from exc
            if completed % 25 == 0 or completed == len(futures):
                LOGGER.info("Verified %d/%d selected raw files", completed, len(futures))
    return sorted(receipts, key=lambda row: row["file_id"])


def run(config_path: str | Path, metadata_only: bool = False, workers: int = 4) -> None:
    config, repo_root = load_config(config_path)
    raw_dir = resolve_path(repo_root, config["paths"]["raw_dir"])
    interim_dir = resolve_path(repo_root, config["paths"]["interim_dir"])
    client = GDCClient(
        base_url=config["gdc"]["api_base_url"],
        timeout_seconds=int(config["gdc"]["timeout_seconds"]),
        max_retries=int(config["gdc"]["max_retries"]),
    )
    snapshot_dir, selected_records = acquire_metadata(client, config, raw_dir, interim_dir)
    if metadata_only:
        LOGGER.info("Metadata snapshot ready at %s", snapshot_dir)
        return

    receipts = download_selected_files(
        client,
        selected_records,
        snapshot_dir,
        config["gdc"]["data_endpoint"],
        workers,
    )
    write_json_atomic(interim_dir / "download_receipts.json", receipts)
    context_path = interim_dir / "run_context.json"
    context = json.loads(context_path.read_text(encoding="utf-8"))
    context.update({"metadata_only": False, "download_completed_at": utc_now()})
    write_json_atomic(context_path, context)
    LOGGER.info("Raw acquisition complete: %d files", len(receipts))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml", help="Path to data YAML")
    parser.add_argument(
        "--metadata-only", action="store_true", help="Query and snapshot metadata without large files"
    )
    parser.add_argument("--workers", type=int, default=4, help="Concurrent public-file downloads")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.verbose)
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    run(args.config, args.metadata_only, args.workers)


if __name__ == "__main__":
    main()
