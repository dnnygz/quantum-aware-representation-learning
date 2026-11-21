from pathlib import Path

import pandas as pd
import pytest

from src.data.expression import ExpressionFormatError, read_star_counts


CONFIG = {
    "value_column": "unstranded",
    "gene_id_column": "gene_id",
    "gene_name_column": "gene_name",
    "gene_type_column": "gene_type",
    "technical_row_prefix": "N_",
    "expected_gene_model": "GENCODE v36",
    "output_dtype": "int64",
}


def write_counts(path: Path, rows: str) -> None:
    path.write_text(
        "# gene-model: GENCODE v36\n"
        "gene_id\tgene_name\tgene_type\tunstranded\tstranded_first\n"
        + rows,
        encoding="utf-8",
    )


def test_read_star_counts_keeps_gene_ids_and_separates_technical_rows(tmp_path):
    path = tmp_path / "counts.tsv"
    write_counts(
        path,
        "N_unmapped\t\t\t5\t5\n"
        "ENSG000001.1\tGENE1\tprotein_coding\t10\t5\n"
        "ENSG000002.2\tGENE2\tlncRNA\t0\t0\n",
    )
    counts, genes, technical = read_star_counts(path, CONFIG)
    assert counts.to_dict() == {"ENSG000001.1": 10, "ENSG000002.2": 0}
    assert genes.loc["ENSG000001.1", "gene_name"] == "GENE1"
    assert technical == {"N_unmapped": 5}
    assert counts.dtype == "int64"


def test_duplicate_gene_ids_fail(tmp_path):
    path = tmp_path / "counts.tsv"
    write_counts(
        path,
        "ENSG000001.1\tGENE1\tprotein_coding\t10\t5\n"
        "ENSG000001.1\tGENE1\tprotein_coding\t11\t5\n",
    )
    with pytest.raises(ExpressionFormatError, match="duplicate gene IDs"):
        read_star_counts(path, CONFIG)


def test_non_integer_raw_counts_fail(tmp_path):
    path = tmp_path / "counts.tsv"
    write_counts(path, "ENSG000001.1\tGENE1\tprotein_coding\t1.5\t1\n")
    with pytest.raises(ExpressionFormatError, match="non-negative integers"):
        read_star_counts(path, CONFIG)
