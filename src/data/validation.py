"""Fail-fast cohort quality checks with a machine-readable report."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class CohortValidationError(ValueError):
    """Raised when the assembled cohort is unsafe for downstream modeling."""


def _check(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"check": name, "passed": bool(passed), "evidence": evidence}


def validate_cohort(
    expression: pd.DataFrame,
    labels: pd.DataFrame,
    metadata: pd.DataFrame,
    genes: pd.DataFrame,
) -> dict[str, Any]:
    """Validate grain, alignment, domains, finiteness and dimensions."""

    checks: list[dict[str, Any]] = []
    case_ids = expression.index.astype(str)
    label_ids = labels["case_id"].astype(str)
    metadata_ids = metadata["case_id"].astype(str)

    checks.extend(
        [
            _check("expression_case_ids_unique", not case_ids.duplicated().any(), int(case_ids.duplicated().sum())),
            _check("label_case_ids_unique", not label_ids.duplicated().any(), int(label_ids.duplicated().sum())),
            _check("metadata_case_ids_unique", not metadata_ids.duplicated().any(), int(metadata_ids.duplicated().sum())),
            _check(
                "sample_ids_unique_after_selection",
                not metadata["sample_id"].duplicated().any(),
                int(metadata["sample_id"].duplicated().sum()),
            ),
            _check("file_ids_unique", not metadata["file_id"].duplicated().any(), int(metadata["file_id"].duplicated().sum())),
            _check("labels_not_missing", not labels["er_label"].isna().any(), int(labels["er_label"].isna().sum())),
            _check(
                "labels_binary",
                set(labels["er_label"].astype(int).unique()).issubset({0, 1}),
                sorted(labels["er_label"].dropna().astype(int).unique().tolist()),
            ),
            _check("gene_ids_unique", not genes["gene_id"].duplicated().any(), int(genes["gene_id"].duplicated().sum())),
            _check("expression_no_nan", not expression.isna().any().any(), int(expression.isna().sum().sum())),
            _check(
                "expression_finite",
                bool(np.isfinite(expression.to_numpy(dtype=float)).all()),
                int((~np.isfinite(expression.to_numpy(dtype=float))).sum()),
            ),
            _check("expression_nonnegative", bool((expression.to_numpy() >= 0).all()), None),
            _check(
                "dimensions_match",
                len(expression) == len(labels) == len(metadata) and expression.shape[1] == len(genes),
                {
                    "expression": list(expression.shape),
                    "labels": len(labels),
                    "metadata": len(metadata),
                    "genes": len(genes),
                },
            ),
            _check(
                "case_alignment",
                list(case_ids) == list(label_ids) == list(metadata_ids),
                {
                    "expression_only": sorted(set(case_ids) - set(label_ids))[:10],
                    "labels_only": sorted(set(label_ids) - set(case_ids))[:10],
                },
            ),
        ]
    )

    distribution = {
        str(int(key)): int(value)
        for key, value in labels["er_label"].value_counts(dropna=False).sort_index().items()
    }
    checks.append(_check("both_er_classes_present", set(map(int, distribution)) == {0, 1}, distribution))
    failures = [item for item in checks if not item["passed"]]
    report = {
        "status": "pass" if not failures else "fail",
        "grain": "one row per GDC case/patient",
        "dimensions": {"patients": len(expression), "genes": expression.shape[1]},
        "class_distribution": distribution,
        "checks": checks,
    }
    if failures:
        names = [item["check"] for item in failures]
        raise CohortValidationError(f"Cohort QC failed: {names}")
    return report
