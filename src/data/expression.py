"""Parser and patient-by-gene assembler for GDC augmented STAR counts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


class ExpressionFormatError(ValueError):
    """Raised when a STAR Counts file violates the expected schema."""


def read_star_counts(
    path: Path, expression_config: Mapping[str, Any]
) -> tuple[pd.Series, pd.DataFrame, dict[str, int]]:
    """Read raw unstranded counts and gene annotations from one GDC file."""

    value_column = str(expression_config["value_column"])
    gene_id_column = str(expression_config["gene_id_column"])
    gene_name_column = str(expression_config["gene_name_column"])
    gene_type_column = str(expression_config["gene_type_column"])
    technical_prefix = str(expression_config["technical_row_prefix"])
    output_dtype = np.dtype(str(expression_config["output_dtype"]))
    if not np.issubdtype(output_dtype, np.integer):
        raise ExpressionFormatError("expression.output_dtype must be an integer dtype")

    with path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline().rstrip("\n")
    if not first_line.startswith("# gene-model:"):
        raise ExpressionFormatError(f"Missing '# gene-model:' header in {path}")
    observed_model = first_line.partition(":")[2].strip()
    expected_model = expression_config.get("expected_gene_model")
    if expected_model and observed_model != expected_model:
        raise ExpressionFormatError(
            f"Unexpected gene model in {path}: expected {expected_model!r}, got {observed_model!r}"
        )

    table = pd.read_csv(path, sep="\t", comment="#", low_memory=False)
    required = {gene_id_column, gene_name_column, gene_type_column, value_column}
    missing = required - set(table.columns)
    if missing:
        raise ExpressionFormatError(f"Missing columns in {path}: {sorted(missing)}")

    technical_mask = table[gene_id_column].astype(str).str.startswith(technical_prefix)
    technical_rows = table.loc[technical_mask, [gene_id_column, value_column]].copy()
    genes = table.loc[~technical_mask, [gene_id_column, gene_name_column, gene_type_column]].copy()
    values = pd.to_numeric(table.loc[~technical_mask, value_column], errors="raise")

    if genes[gene_id_column].isna().any() or genes[gene_id_column].duplicated().any():
        duplicates = genes.loc[genes[gene_id_column].duplicated(), gene_id_column].tolist()[:10]
        raise ExpressionFormatError(f"Missing or duplicate gene IDs in {path}: {duplicates}")
    array = values.to_numpy(dtype=float)
    if not np.isfinite(array).all():
        raise ExpressionFormatError(f"Non-finite expression values in {path}")
    if (array < 0).any() or not np.equal(array, np.floor(array)).all():
        raise ExpressionFormatError(f"Raw counts must be non-negative integers in {path}")

    index = pd.Index(genes[gene_id_column].astype(str), name="gene_id")
    counts = pd.Series(array.astype(output_dtype), index=index, name=value_column)
    annotations = genes.set_index(gene_id_column)
    annotations.index = annotations.index.astype(str)
    annotations.index.name = "gene_id"
    technical = {
        str(row[gene_id_column]): int(row[value_column])
        for _, row in technical_rows.iterrows()
        if pd.notna(row[value_column])
    }
    return counts, annotations, technical


def assemble_expression_matrix(
    selected_metadata: pd.DataFrame,
    raw_snapshot_dir: Path,
    expression_config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Assemble a case-level matrix and verify a consistent gene schema."""

    rows: list[pd.Series] = []
    qc_rows: list[dict[str, Any]] = []
    canonical_annotations: pd.DataFrame | None = None
    canonical_gene_ids: pd.Index | None = None

    for record in selected_metadata.sort_values("case_submitter_id").to_dict("records"):
        path = raw_snapshot_dir / "files" / record["file_id"] / record["file_name"]
        if not path.is_file():
            raise FileNotFoundError(
                f"Required raw expression file is missing: {path}. Run python -m src.data.download."
            )
        counts, annotations, technical = read_star_counts(path, expression_config)
        if canonical_gene_ids is None:
            canonical_gene_ids = counts.index
            canonical_annotations = annotations
        else:
            if not counts.index.equals(canonical_gene_ids):
                missing = canonical_gene_ids.difference(counts.index).tolist()[:10]
                extra = counts.index.difference(canonical_gene_ids).tolist()[:10]
                raise ExpressionFormatError(
                    f"Gene schema mismatch in {path}; missing={missing}, extra={extra}"
                )
            if not annotations.equals(canonical_annotations):
                raise ExpressionFormatError(f"Gene annotations differ in {path}")

        counts.name = record["case_id"]
        rows.append(counts)
        qc_rows.append(
            {
                "case_id": record["case_id"],
                "file_id": record["file_id"],
                "gene_count": int(counts.size),
                "library_size_unstranded": int(counts.sum()),
                "zero_gene_count": int((counts == 0).sum()),
                **{f"star_{key}": value for key, value in technical.items()},
            }
        )

    if not rows or canonical_annotations is None:
        raise ExpressionFormatError("No expression files were selected for the final cohort")
    matrix = pd.DataFrame(rows)
    matrix.index.name = "case_id"
    genes = canonical_annotations.reset_index()
    return matrix, genes, pd.DataFrame(qc_rows)
