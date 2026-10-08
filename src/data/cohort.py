"""Build the patient-level TCGA-BRCA ER cohort and all audit artifacts."""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from .baseline_er import build_baseline_clinical_table
from .expression import assemble_expression_matrix
from .splits import generate_splits, validate_splits
from .utils import (
    configure_logging,
    dependency_versions,
    git_state,
    load_config,
    resolve_path,
    sha256_file,
    utc_now,
    write_json_atomic,
)
from .validation import validate_cohort


LOGGER = logging.getLogger(__name__)


FILE_COLUMNS = [
    "file_id",
    "file_name",
    "file_size",
    "md5sum",
    "data_release",
    "file_created_datetime",
    "file_updated_datetime",
    "workflow_type",
    "case_id",
    "case_submitter_id",
    "sample_id",
    "sample_submitter_id",
    "sample_type",
    "sample_created_datetime",
    "is_ffpe",
    "preservation_method",
    "portion_number",
    "analyte_type",
    "aliquot_id",
    "aliquot_submitter_id",
]


def flatten_file_hits(
    hits: Iterable[Mapping[str, Any]], required_sample_type: str = "Primary Tumor"
) -> pd.DataFrame:
    """Flatten nested GDC file metadata without silently discarding candidates."""

    rows: list[dict[str, Any]] = []
    for hit in hits:
        for case in hit.get("cases") or []:
            for sample in case.get("samples") or []:
                if sample.get("sample_type") != required_sample_type:
                    continue
                aliquots: list[dict[str, Any]] = []
                for portion in sample.get("portions") or []:
                    for analyte in portion.get("analytes") or []:
                        for aliquot in analyte.get("aliquots") or []:
                            aliquots.append(
                                {
                                    "portion_number": portion.get("portion_number"),
                                    "analyte_type": analyte.get("analyte_type"),
                                    "aliquot_id": aliquot.get("aliquot_id"),
                                    "aliquot_submitter_id": aliquot.get("submitter_id"),
                                }
                            )
                if not aliquots:
                    aliquots = [{}]
                for aliquot in aliquots:
                    rows.append(
                        {
                            "file_id": hit.get("file_id") or hit.get("id"),
                            "file_name": hit.get("file_name"),
                            "file_size": hit.get("file_size"),
                            "md5sum": hit.get("md5sum"),
                            "data_release": hit.get("data_release"),
                            "file_created_datetime": hit.get("created_datetime"),
                            "file_updated_datetime": hit.get("updated_datetime"),
                            "workflow_type": (hit.get("analysis") or {}).get("workflow_type"),
                            "case_id": case.get("case_id"),
                            "case_submitter_id": case.get("submitter_id"),
                            "sample_id": sample.get("sample_id"),
                            "sample_submitter_id": sample.get("submitter_id"),
                            "sample_type": sample.get("sample_type"),
                            "sample_created_datetime": sample.get("created_datetime"),
                            "is_ffpe": sample.get("is_ffpe"),
                            "preservation_method": sample.get("preservation_method"),
                            **aliquot,
                        }
                    )
    frame = pd.DataFrame(rows, columns=FILE_COLUMNS)
    if frame.empty:
        raise ValueError("The GDC file query produced no eligible sample/file relationships")
    required = ["file_id", "file_name", "file_size", "md5sum", "case_id", "sample_id"]
    missing = frame[required].isna().any()
    if missing.any():
        raise ValueError(f"Required GDC metadata are missing: {missing[missing].index.tolist()}")
    return frame.drop_duplicates().reset_index(drop=True)


def _sortable_portion(value: Any) -> tuple[int, str]:
    text = "" if value is None or pd.isna(value) else str(value)
    try:
        return int(text), text
    except ValueError:
        return 10**9, text


def select_patient_files(
    candidates: pd.DataFrame, selection_config: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select one file per case using an explicit, outcome-blind ordering."""

    if selection_config.get("strategy") != "tcga_barcode_lexicographic":
        raise ValueError("Unsupported sample selection strategy")

    required = {
        "case_id",
        "file_id",
        "sample_id",
        "sample_submitter_id",
        "portion_number",
        "aliquot_submitter_id",
    }
    if not required.issubset(candidates.columns):
        raise ValueError(f"Candidate metadata missing fields: {sorted(required - set(candidates))}")
    if candidates["case_id"].isna().any():
        raise ValueError("Cannot select samples with missing case IDs")

    audit = candidates.copy()
    portion_keys = audit["portion_number"].map(_sortable_portion)
    audit["_portion_numeric"] = portion_keys.map(lambda item: item[0])
    audit["_portion_text"] = portion_keys.map(lambda item: item[1])
    requested = list(selection_config.get("sort_keys", []))
    supported = {
        "sample_submitter_id",
        "portion_number",
        "aliquot_submitter_id",
        "sample_id",
        "file_id",
    }
    if set(requested) - supported:
        raise ValueError(f"Unsupported sample selection keys: {sorted(set(requested) - supported)}")
    sort_columns = ["case_id"]
    for key in requested:
        sort_columns.extend(["_portion_numeric", "_portion_text"] if key == "portion_number" else [key])
    # Stable fallbacks make even a shortened user configuration deterministic.
    for fallback in ["sample_submitter_id", "_portion_numeric", "_portion_text", "aliquot_submitter_id", "sample_id", "file_id"]:
        if fallback not in sort_columns:
            sort_columns.append(fallback)
    for column in sort_columns:
        if column in {"case_id", "_portion_numeric"}:
            continue
        audit[column] = audit[column].fillna("").astype(str)
    audit = audit.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    audit["selection_rank"] = audit.groupby("case_id").cumcount() + 1
    audit["selected"] = audit["selection_rank"].eq(1)
    candidate_counts = audit.groupby("case_id")["file_id"].transform("size")
    audit["selection_reason"] = audit["selected"].map(
        {True: "selected_by_configured_outcome_blind_order", False: "not_first_by_configured_order"}
    )
    audit["case_candidate_count"] = candidate_counts
    audit = audit.drop(columns=["_portion_numeric", "_portion_text"])
    selected = audit.loc[audit["selected"]].drop(columns=["selected"]).copy()
    if selected["case_id"].duplicated().any():
        raise AssertionError("Deterministic selection did not produce one row per case")
    return selected.reset_index(drop=True), audit


def _write_parquet_atomic(frame: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_parquet(temporary, index=index)
    os.replace(temporary, path)


def _build_summary(
    all_cases: pd.DataFrame,
    rna_candidates: pd.DataFrame,
    final_labels: pd.DataFrame,
    audit: pd.DataFrame,
) -> dict[str, Any]:
    rna_case_ids = set(rna_candidates["case_id"].astype(str))
    clinical_by_case = all_cases.set_index("case_id")
    exclusion_counts: Counter[str] = Counter()
    exclusion_rows: list[dict[str, Any]] = []
    for case_id in sorted(rna_case_ids):
        if case_id not in clinical_by_case.index:
            reason = "case_missing_from_clinical_response"
            submitter_id = None
        else:
            row = clinical_by_case.loc[case_id]
            reason = row["er_exclusion_reason"]
            submitter_id = row["case_submitter_id"]
        if pd.notna(reason):
            exclusion_counts[str(reason)] += 1
            exclusion_rows.append(
                {"case_id": case_id, "case_submitter_id": submitter_id, "reason": str(reason)}
            )

    final_distribution = final_labels["er_label"].astype(int).value_counts().to_dict()
    project_known = all_cases["er_label"].notna()
    return {
        "generated_at": utc_now(),
        "flow": {
            "total_project_cases": int(len(all_cases)),
            "cases_with_eligible_primary_tumor_rna_seq": int(len(rna_case_ids)),
            "project_cases_with_known_baseline_er": int(project_known.sum()),
            "rna_cases_with_valid_er": int(len(final_labels)),
            "cases_with_multiple_expression_candidates": int(
                audit.loc[audit["case_candidate_count"] > 1, "case_id"].nunique()
            ),
            "final_patient_cohort": int(len(final_labels)),
        },
        "project_er_distribution": {
            "ER_positive": int((all_cases["er_label"] == 1).sum()),
            "ER_negative": int((all_cases["er_label"] == 0).sum()),
        },
        "final_er_distribution": {
            "ER_positive": int(final_distribution.get(1, 0)),
            "ER_negative": int(final_distribution.get(0, 0)),
        },
        "exclusions_from_rna_cases": {
            "total": int(sum(exclusion_counts.values())),
            "by_reason": dict(sorted(exclusion_counts.items())),
            "cases": exclusion_rows,
        },
    }


def _build_provenance(
    config: dict[str, Any],
    repo_root: Path,
    config_path: Path,
    snapshot_dir: Path,
    selected_metadata: pd.DataFrame,
    summary: dict[str, Any],
) -> dict[str, Any]:
    metadata_dir = snapshot_dir / "metadata"
    status = json.loads((metadata_dir / "gdc_status.json").read_text(encoding="utf-8"))
    file_query = json.loads((metadata_dir / "file_query.json").read_text(encoding="utf-8"))
    case_query = json.loads((metadata_dir / "case_query.json").read_text(encoding="utf-8"))
    interim_dir = resolve_path(repo_root, config["paths"]["interim_dir"])
    receipts_path = interim_dir / "download_receipts.json"
    receipts = json.loads(receipts_path.read_text(encoding="utf-8"))
    run_context = json.loads((interim_dir / "run_context.json").read_text(encoding="utf-8"))
    baseline_context = run_context["baseline_er"]
    baseline_snapshot = Path(baseline_context["snapshot_dir"])
    baseline_metadata_dir = baseline_snapshot / "metadata"
    receipts_by_id = {receipt["file_id"]: receipt for receipt in receipts}
    selected_ids = set(selected_metadata["file_id"].astype(str))
    if set(receipts_by_id) != selected_ids:
        raise ValueError("Download receipts do not match the selected final files")
    source_paths = sorted((repo_root / "src").rglob("*.py"))
    source_paths.extend([repo_root / "pyproject.toml", config_path])
    code_state = git_state(repo_root)
    code_state["source_files_sha256"] = {
        str(path.relative_to(repo_root)): sha256_file(path) for path in source_paths
    }
    return {
        "timestamp": utc_now(),
        "project": config["project_id"],
        "cohort_version": config["cohort_version"],
        "cohort_status": config["cohort_status"],
        "gdc_status": status,
        "gdc_endpoints": {
            key: f"{config['gdc']['api_base_url']}{value}"
            for key, value in config["gdc"].items()
            if key.endswith("_endpoint")
        },
        "queries": {"files": file_query, "cases": case_query},
        "filters": config["query"],
        "workflow": config["query"]["workflow_type"],
        "sample_type": config["query"]["sample_type"],
        "expression_measure": config["expression"]["value_column"],
        "er_source": {
            "primary": {
                "source": config["clinical"]["primary_source"],
                "field": config["clinical"]["primary_field"],
                "xml_element": config["clinical"]["primary_xml_element"],
                "cde": config["clinical"]["primary_cde"],
                "timepoint": config["clinical"]["clinical_timepoint"],
                "later_event_imputation_prohibited": config["clinical"]["prohibit_later_event_imputation"],
            },
            "crosscheck": "TCGA BCR patient Biotab er_status_by_ihc",
            "secondary_validation": {
                "location": "cases.follow_ups.molecular_tests",
                "gene_symbol": config["clinical"]["er_gene_symbol"],
                "method": config["clinical"]["er_method"],
            },
        },
        "methodological_change": {
            "decision": "Approved baseline ER ground truth from Clinical Supplement XML",
            "supersedes": "v0_harmonized_esr1_ihc",
            "superseded_manifest": "data/processed/versions/superseded/v0_harmonized_esr1_ihc/manifest.json",
            "audit_report": "reports/data/er_source_audit.md",
            "audit_machine_readable_prechange": "reports/data/audit/er_source_audit_release46_prechange.json",
            "comparison_report": "reports/data/old_cohort_vs_baseline_er_cohort.json",
        },
        "counts": summary["flow"],
        "files": [
            {
                "file_id": row.file_id,
                "file_name": row.file_name,
                "case_id": row.case_id,
                "sample_id": row.sample_id,
                "file_size": int(row.file_size),
                "gdc_md5": row.md5sum,
                "data_release": row.data_release,
                "downloaded_at": receipts_by_id[row.file_id]["downloaded_at"],
                "reused_on_acquisition_run": receipts_by_id[row.file_id]["reused"],
                "integrity_verified": True,
            }
            for row in selected_metadata.itertuples(index=False)
        ],
        "configuration": config,
        "configuration_sha256": sha256_file(config_path),
        "raw_metadata_sha256": {
            path.name: sha256_file(path)
            for path in sorted(metadata_dir.glob("*.json"))
        },
        "baseline_clinical_supplement": {
            **baseline_context,
            "metadata_sha256": {
                path.name: sha256_file(path)
                for path in sorted(baseline_metadata_dir.glob("*.json"))
            },
            "source_comparison_sha256": sha256_file(
                interim_dir / "baseline_er_source_comparison.parquet"
            ),
            "xml_observations_sha256": sha256_file(
                interim_dir / "baseline_er_xml_observations.parquet"
            ),
        },
        "download_receipts_sha256": sha256_file(receipts_path),
        "code": code_state,
        "dependencies": dependency_versions(),
        "documentation": {
            "gdc_search": "https://docs.gdc.cancer.gov/API/Users_Guide/Search_and_Retrieval/",
            "gdc_download": "https://docs.gdc.cancer.gov/API/Users_Guide/Downloading_Files/",
            "gdc_mrna": "https://docs.gdc.cancer.gov/Data/Bioinformatics_Pipelines/Expression_mRNA_Pipeline/",
            "gdc_star_counts": "https://docs.gdc.cancer.gov/Encyclopedia/pages/STAR_Counts/",
        },
    }


def run(config_path: str | Path) -> None:
    config, repo_root = load_config(config_path)
    config_path = Path(config_path).expanduser().resolve()
    interim_dir = resolve_path(repo_root, config["paths"]["interim_dir"])
    processed_dir = resolve_path(repo_root, config["paths"]["processed_dir"])
    reports_dir = resolve_path(repo_root, config["paths"]["reports_dir"])
    context_path = interim_dir / "run_context.json"
    if not context_path.is_file():
        raise FileNotFoundError("Missing acquisition context. Run python -m src.data.download first.")
    context = json.loads(context_path.read_text(encoding="utf-8"))
    if context.get("metadata_only"):
        raise RuntimeError("Raw expression download is incomplete; rerun src.data.download without --metadata-only")
    snapshot_dir = Path(context["snapshot_dir"])
    metadata_dir = snapshot_dir / "metadata"
    file_response = json.loads((metadata_dir / "file_response.json").read_text(encoding="utf-8"))
    clinical_response = json.loads(
        (metadata_dir / "clinical_response.json").read_text(encoding="utf-8")
    )
    if config["clinical"].get("primary_source") != "clinical_supplement_bcr_xml":
        raise ValueError("Canonical v1 requires clinical.primary_source=clinical_supplement_bcr_xml")
    baseline_source_path = interim_dir / "baseline_er_source_comparison.parquet"
    if not baseline_source_path.is_file():
        raise FileNotFoundError("Missing baseline ER source table; rerun src.data.download")
    baseline_sources = pd.read_parquet(baseline_source_path)

    candidates = flatten_file_hits(file_response["data"]["hits"], config["query"]["sample_type"])
    selected, audit = select_patient_files(candidates, config["sample_selection"])
    all_clinical, er_records = build_baseline_clinical_table(
        clinical_response["data"]["hits"], baseline_sources, config["clinical"]
    )
    merged = selected.merge(all_clinical, on=["case_id", "case_submitter_id"], how="left", validate="one_to_one")
    final_metadata = merged.loc[merged["er_label"].notna()].copy()
    final_metadata["er_label"] = final_metadata["er_label"].astype(int)
    final_metadata = final_metadata.sort_values("case_id").reset_index(drop=True)

    expression, genes, expression_qc = assemble_expression_matrix(
        final_metadata, snapshot_dir, config["expression"]
    )
    expression = expression.loc[final_metadata["case_id"].tolist()]
    labels = final_metadata[["case_id", "case_submitter_id", "er_status", "er_label"]].copy()
    metadata = final_metadata.drop(columns=["er_label"]).copy()
    qc_report = validate_cohort(expression, labels, metadata, genes)
    qc_report["expression_file_qc"] = expression_qc.to_dict("records")
    splits = generate_splits(labels, config["splits"])
    qc_report["split_qc"] = validate_splits(splits, labels)
    summary = _build_summary(all_clinical, candidates, labels, audit)
    provenance = _build_provenance(
        config, repo_root, config_path, snapshot_dir, final_metadata, summary
    )

    _write_parquet_atomic(expression, processed_dir / "expression.parquet", index=True)
    _write_parquet_atomic(labels, processed_dir / "labels.parquet")
    _write_parquet_atomic(metadata, processed_dir / "metadata.parquet")
    _write_parquet_atomic(genes, processed_dir / "genes.parquet")
    _write_parquet_atomic(audit, interim_dir / "sample_selection_audit.parquet")
    _write_parquet_atomic(er_records, interim_dir / "er_molecular_tests.parquet")
    _write_parquet_atomic(expression_qc, interim_dir / "expression_qc.parquet")
    write_json_atomic(processed_dir / "splits.json", splits)
    write_json_atomic(reports_dir / "cohort_summary.json", summary)
    write_json_atomic(reports_dir / "qc_report.json", qc_report)
    write_json_atomic(reports_dir / "provenance.json", provenance)
    provenance["processed_artifacts_sha256"] = {
        path.name: sha256_file(path)
        for path in sorted(processed_dir.glob("*"))
        if path.is_file() and path.name != ".gitkeep"
    }
    write_json_atomic(reports_dir / "provenance.json", provenance)
    LOGGER.info("Final patient cohort: %d cases x %d genes", *expression.shape)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml", help="Path to data YAML")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.verbose)
    run(args.config)


if __name__ == "__main__":
    main()
