"""Immutable local version snapshots for processed cohort artifacts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from .utils import load_config, resolve_path, sha256_file, utc_now, write_json_immutable


PROCESSED_ARTIFACTS = [
    "expression.parquet",
    "labels.parquet",
    "metadata.parquet",
    "genes.parquet",
    "splits.json",
]
REPORT_ARTIFACTS = ["cohort_summary.json", "qc_report.json", "provenance.json"]


def archive_current_cohort(
    config_path: str | Path,
    version_id: str,
    target_definition: str,
) -> Path:
    """Archive the current top-level cohort without deleting or rewriting it."""

    config, repo_root = load_config(config_path)
    processed_dir = resolve_path(repo_root, config["paths"]["processed_dir"])
    reports_dir = resolve_path(repo_root, config["paths"]["reports_dir"])
    archive_dir = processed_dir / "versions" / "superseded" / version_id
    manifest_path = archive_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for relative, expected in manifest["sha256"].items():
            path = archive_dir / relative
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f"Superseded cohort archive integrity failure: {path}")
        return archive_dir

    missing = [name for name in PROCESSED_ARTIFACTS if not (processed_dir / name).is_file()]
    missing += [f"reports/{name}" for name in REPORT_ARTIFACTS if not (reports_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot archive incomplete current cohort: {missing}")

    (archive_dir / "processed").mkdir(parents=True, exist_ok=False)
    (archive_dir / "reports").mkdir(parents=True, exist_ok=False)
    for name in PROCESSED_ARTIFACTS:
        os.link(processed_dir / name, archive_dir / "processed" / name)
    for name in REPORT_ARTIFACTS:
        shutil.copy2(reports_dir / name, archive_dir / "reports" / name)
    hashes = {
        str(path.relative_to(archive_dir)): sha256_file(path)
        for path in sorted(archive_dir.rglob("*"))
        if path.is_file()
    }
    manifest: dict[str, Any] = {
        "version_id": version_id,
        "status": "superseded",
        "archived_at": utc_now(),
        "target_definition": target_definition,
        "superseded_by": "v1_baseline_er_cde_2957359",
        "reason": "Primary ER target changed from unrestricted harmonized ESR1/IHC aggregation to baseline Clinical Supplement ER IHC.",
        "sha256": hashes,
    }
    write_json_immutable(manifest_path, manifest)
    return archive_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--version-id", required=True)
    parser.add_argument("--target-definition", required=True)
    args = parser.parse_args()
    path = archive_current_cohort(args.config, args.version_id, args.target_definition)
    print(path)


if __name__ == "__main__":
    main()
