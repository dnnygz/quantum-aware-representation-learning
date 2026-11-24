#!/usr/bin/env python3
"""Compares a re-run (e.g. results_rerun/) with the run reported in the paper (results/); both the flat and the sorted layout are read.
Rows are matched on the non-metric columns (dataset, seed, model, ...); a smoke run (2 seeds) is compared on the seeds it has."""
import argparse
from pathlib import Path
import pandas as pd, numpy as np
ROOT = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument("--mine", default="results_rerun"); ap.add_argument("--reference", default="results"); ap.add_argument("--tol", type=float, default=1e-3)
a = ap.parse_args(); mine, ref = ROOT / a.mine, ROOT / a.reference
METRICS = ["auc_tr", "auc_te", "brier_te", "brier_tr", "bce_tr"]
lines = [f"# Comparison of `{a.mine}` against `{a.reference}` (tolerance {a.tol})", "", "| file | matched rows | max abs diff (test AUC) | mean abs diff (test AUC) | max abs diff (any metric) | verdict |", "|---|---|---|---|---|---|"]
bad = 0
for f in sorted(ref.rglob("*_results.csv")):
    g = next(iter(mine.rglob(f.name)), None)      # works for the flat folder of run_all.py and for the sorted one
    if g is None:
        lines.append(f"| {f.name} | - | - | - | - | missing in `{a.mine}` |"); continue
    A, B = pd.read_csv(f), pd.read_csv(g)
    met = [c for c in METRICS if c in A.columns and c in B.columns]; keys = [c for c in A.columns if c not in METRICS and c in B.columns]
    m = A.merge(B, on=keys, suffixes=("_ref", "_mine"))
    if len(m) == 0:
        lines.append(f"| {f.name} | 0 | - | - | - | no common rows |"); continue
    d = {c: (m[c + "_ref"] - m[c + "_mine"]).abs() for c in met}
    mx = max(float(v.max()) for v in d.values()); au = d["auc_te"]
    ok = mx <= a.tol; bad += (not ok)
    lines.append(f"| {f.name} | {len(m)} | {au.max():.2e} | {au.mean():.2e} | {mx:.2e} | {'OK' if ok else 'DIFFERENT'} |")
lines += ["", "Interpretation: `OK` means the re-run reproduces the reference to within the tolerance (small differences come from different CPUs, BLAS or library versions).",
          "`DIFFERENT` means some rows differ by more than the tolerance: check `environment.json` (library versions) before trusting either run, and do not tune anything to make them match."]
txt = "\n".join(lines); print(txt); (mine / "compare_report.md").write_text(txt, encoding="utf-8")
print("\nfiles different:", bad)
