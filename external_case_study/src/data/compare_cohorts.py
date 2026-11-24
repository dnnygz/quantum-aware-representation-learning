"""Compare superseded v0 and canonical baseline-ER v1 cohorts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import load_config, resolve_path, sha256_file, utc_now, write_json_atomic


def _hashes(directory: Path, names: list[str]) -> dict[str, str]:
    return {name: sha256_file(directory / name) for name in names}


def compare(config_path: str | Path) -> dict[str, Any]:
    config, repo_root = load_config(config_path)
    processed = resolve_path(repo_root, config["paths"]["processed_dir"])
    interim = resolve_path(repo_root, config["paths"]["interim_dir"])
    reports = resolve_path(repo_root, config["paths"]["reports_dir"])
    old_root = processed / "versions" / "superseded" / "v0_harmonized_esr1_ihc"
    old_processed = old_root / "processed"

    old_labels = pd.read_parquet(old_processed / "labels.parquet")
    new_labels = pd.read_parquet(processed / "labels.parquet")
    new_metadata = pd.read_parquet(processed / "metadata.parquet")
    all_new_clinical = pd.read_parquet(interim / "clinical_labels_preview.parquet")
    old_by_id = old_labels.set_index("case_id")
    new_by_id = new_labels.set_index("case_id")
    old_ids, new_ids = set(old_by_id.index), set(new_by_id.index)
    entered_ids, exited_ids = sorted(new_ids - old_ids), sorted(old_ids - new_ids)
    common_ids = sorted(old_ids & new_ids)
    changed_ids = [
        case_id
        for case_id in common_ids
        if int(old_by_id.loc[case_id, "er_label"]) != int(new_by_id.loc[case_id, "er_label"])
    ]

    metadata_by_id = new_metadata.set_index("case_id")
    clinical_by_id = all_new_clinical.set_index("case_id")
    entered = []
    for case_id in entered_ids:
        row = metadata_by_id.loc[case_id]
        entered.append(
            {
                "case_id": case_id,
                "case_submitter_id": row["case_submitter_id"],
                "new_er_label": int(new_by_id.loc[case_id, "er_label"]),
                "reason": "baseline_er_known_but_prior_harmonized_policy_excluded_case",
                "prior_harmonized_exclusion_reason": row.get("harmonized_er_exclusion_reason"),
                "baseline_raw_values": row.get("xml_primary_raw_values"),
                "harmonized_raw_values": row.get("harmonized_er_raw_values"),
            }
        )
    exited = []
    for case_id in exited_ids:
        row = clinical_by_id.loc[case_id]
        exited.append(
            {
                "case_id": case_id,
                "case_submitter_id": old_by_id.loc[case_id, "case_submitter_id"],
                "old_er_label": int(old_by_id.loc[case_id, "er_label"]),
                "reason": "baseline_er_missing_or_unknown_later_event_not_used_for_imputation",
                "baseline_exclusion_reason": row.get("er_exclusion_reason"),
                "baseline_raw_values": row.get("xml_primary_raw_values"),
                "harmonized_raw_values": row.get("harmonized_er_raw_values"),
            }
        )
    changed = [
        {
            "case_id": case_id,
            "case_submitter_id": new_by_id.loc[case_id, "case_submitter_id"],
            "old_er_label": int(old_by_id.loc[case_id, "er_label"]),
            "new_er_label": int(new_by_id.loc[case_id, "er_label"]),
        }
        for case_id in changed_ids
    ]
    artifact_names = ["expression.parquet", "genes.parquet", "labels.parquet", "metadata.parquet", "splits.json"]
    result = {
        "generated_at": utc_now(),
        "old": {
            "version": "v0_harmonized_esr1_ihc",
            "status": "superseded",
            "patients": len(old_labels),
            "class_distribution": {
                "ER_positive": int(old_labels["er_label"].eq(1).sum()),
                "ER_negative": int(old_labels["er_label"].eq(0).sum()),
            },
            "sha256": _hashes(old_processed, artifact_names),
            "manifest": str((old_root / "manifest.json").relative_to(repo_root)),
        },
        "new": {
            "version": config["cohort_version"],
            "status": config["cohort_status"],
            "target": "baseline_er_status_primary_tumor_cde_2957359",
            "patients": len(new_labels),
            "class_distribution": {
                "ER_positive": int(new_labels["er_label"].eq(1).sum()),
                "ER_negative": int(new_labels["er_label"].eq(0).sum()),
            },
            "sha256": _hashes(processed, artifact_names),
        },
        "patient_changes": {
            "common": len(common_ids),
            "entered": entered,
            "exited": exited,
            "label_changed": changed,
        },
        "methodological_reason": (
            "Baseline Clinical Supplement ER is the estimand; recurrence, metastasis and new-tumour-event ER cannot impute baseline."
        ),
    }
    write_json_atomic(reports / "old_cohort_vs_baseline_er_cohort.json", result)

    lines = [
        "# Old cohort vs baseline-ER canonical cohort",
        "",
        f"Generated: {result['generated_at']}",
        "",
        "| Cohort | Status | Patients | ER+ | ER− |",
        "|---|---|---:|---:|---:|",
        f"| v0 harmonized ESR1/IHC | superseded | {result['old']['patients']} | {result['old']['class_distribution']['ER_positive']} | {result['old']['class_distribution']['ER_negative']} |",
        f"| v1 baseline ER CDE 2957359 | canonical | {result['new']['patients']} | {result['new']['class_distribution']['ER_positive']} | {result['new']['class_distribution']['ER_negative']} |",
        "",
        f"Common patients: {len(common_ids)}. Entered: {len(entered)}. Exited: {len(exited)}. Label changes among common patients: {len(changed)}.",
        "",
        "## Entered patients",
        "",
        "| Patient | New label | Previous-policy reason |",
        "|---|---:|---|",
    ]
    lines.extend(
        f"| {row['case_submitter_id']} | {row['new_er_label']} | {row['prior_harmonized_exclusion_reason']} |"
        for row in entered
    )
    lines.extend(["", "## Exited patients", "", "| Patient | Old label | Baseline reason |", "|---|---:|---|"])
    lines.extend(
        f"| {row['case_submitter_id']} | {row['old_er_label']} | {row['baseline_exclusion_reason']} |"
        for row in exited
    )
    lines.extend(
        [
            "",
            "No later-event ER value was used to fill a missing baseline label. Full UUIDs, raw values, reasons and artifact hashes are in the JSON report.",
            "",
        ]
    )
    markdown_path = reports / "old_cohort_vs_baseline_er_cohort.md"
    temporary = markdown_path.with_suffix(".md.part")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, markdown_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    print(json.dumps(compare(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
