#!/usr/bin/env python3
"""Sorts the flat output folder written by run_all.py into one folder per experiment (this is how results/ is laid out).

  python tidy_results.py --dir results_rerun            # moves the files of results_rerun/ into subfolders
  python tidy_results.py --dir results_rerun --dry-run  # only prints what would move

Nothing is changed in the files themselves, only their location. Run it after the experiments have finished: the scripts read each
other's CSV files from one flat folder, so a folder that has been sorted can no longer be resumed or extended by run_all.py.
compare_results.py reads both layouts.
"""
import argparse, re, shutil
from pathlib import Path

# (regex on the file name, folder). The first match wins. Log files of an experiment go to <folder>/logs/.
FOLDERS = [
    (r"^(e1_|fig_e1|tables_sec42\.tex|run_e1\.log|analyze_e1\.log|gen_sec42\.log)", "e1_baseline"),
    (r"^(h1_|sec42_h1\.tex|run_h1\.log|analyze_h1\.log|gen_sec42_h4h1\.log)", "h1_representation"),
    (r"^(h2|fig_h2|sec42_h2\.tex|run_h2_|analyze_h2\.log|gen_sec42_h2\.log)", "h2_encoding"),
    (r"^(h3_|eff_params|sec42_h3\.tex|run_h3\.log|analyze_h3\.log|gen_sec42_h3\.log)", "h3_capacity_readout"),
    (r"^(h4_|fig_h4|sec42_h4\.tex|run_h4\.log|analyze_h4\.log)", "h4_training_size"),
    (r"^(h5_|fig_h5|tab_h5\.tex|run_h5\.log|analyze_h5\.log|gen_sec42_h5\.log)", "h5_optimization"),
    (r"^(h6_|sec42_h6\.tex|run_h6\.log|analyze_h6\.log|gen_sec42_h6\.log)", "h6_classical_teacher"),
    (r"^(pilot)", "pilot"),
    (r"^(check_)", "validation"),
    (r"^(environment\.|timings|run_all_)", "run_info"),
]

def target(name):
    for pat, folder in FOLDERS:
        if re.match(pat, name):
            return Path(folder) / ("logs" if name.endswith(".log") and folder[0] in "eh" else "") / name
    return None

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(); root = Path(a.dir)
    files = [f for f in list(root.glob("*")) + list((root / "logs").glob("*")) if f.is_file() and f.name != "README.md"]
    left = []
    for f in sorted(files):
        t = target(f.name)
        if t is None: left.append(f.name); continue
        print(f"{f.relative_to(root)}  ->  {t}")
        if not a.dry_run:
            (root / t).parent.mkdir(parents=True, exist_ok=True); shutil.move(str(f), str(root / t))
    if (root / "logs").exists() and not a.dry_run and not any((root / "logs").iterdir()): (root / "logs").rmdir()
    if left: print("left in place (no rule):", ", ".join(left))

if __name__ == "__main__":
    main()
