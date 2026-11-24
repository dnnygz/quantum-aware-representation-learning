#!/usr/bin/env python3
"""Runs the whole study (or part of it) in the order the paper needs it.

  python run_all.py --smoke              # 2 seeds per experiment, writes to results_smoke/ (about 10-20 min)
  python run_all.py --jobs 6             # full run (30 seeds), writes to results/
  python run_all.py --only h2,h6         # only some stages (comma separated)
  python run_all.py --skip h2            # everything except some stages
  python run_all.py --validate           # also run the simulator checks (check_pennylane.py needs `pip install pennylane`)

Each experiment script writes its CSV only when it finishes. A stage whose main CSV already exists is skipped
unless --force is given, so an interrupted run can be resumed (the interrupted script restarts from its first seed).
Hyper-parameters are frozen (lr=0.05, 200 epochs, 30 seeds); do not change them to "improve" results.
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORE = ROOT / "core"                       # shared modules (vqc.py, vqc_enc.py, common.py), put on PYTHONPATH for every script
SCRIPT_DIRS = ["experiments", "analysis", "tables", "checks"]

def locate(script):
    for d in SCRIPT_DIRS:
        if (ROOT / d / script).exists(): return ROOT / d / script
    raise FileNotFoundError(script)

# stage -> list of (script, args, main output that proves the step is done, or None)
STAGES = {
    "e1": [("run_e1.py", [], "e1_results.csv"), ("analyze_e1.py", [], "e1_summary.csv"), ("gen_sec42.py", [], "tables_sec42.tex")],
    "h5": [("run_h5.py", [], "h5_results.csv"), ("analyze_h5.py", [], "h5_summary.csv"), ("gen_sec42_h5.py", [], "tab_h5.tex")],
    "h3": [("eff_params.py", [], "eff_params.csv"), ("run_h3.py", [], "h3_results.csv"), ("analyze_h3.py", [], "h3_paired.csv"), ("gen_sec42_h3.py", [], "sec42_h3.tex")],
    "h4": [("run_h4.py", [], "h4_results.csv"), ("analyze_h4.py", [], "h4_summary.csv")],
    "h1": [("run_h1.py", [], "h1_results.csv"), ("analyze_h1.py", [], "h1_summary.csv"), ("gen_sec42_h4h1.py", [], "sec42_h1.tex")],
    "h2": [("run_h2.py", ["B"], "h2B_results.csv"), ("run_h2.py", ["A"], "h2A_results.csv"), ("analyze_h2.py", [], "h2_paired.csv"), ("gen_sec42_h2.py", [], "sec42_h2.tex")],
    "h6": [("run_h6.py", [], "h6_results.csv"), ("analyze_h6.py", [], "h6_paired.csv"), ("gen_sec42_h6.py", [], "sec42_h6.tex")],
}
OPTIONAL = {
    "validate": [("check_enc.py", [], None), ("check_pennylane.py", [], None)],
    "pilot": [("pilot.py", [], "pilot.csv")],      # re-selects lr/epochs on a validation split; NOT needed (config is frozen)
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 2), help="worker processes per experiment (default: min(4, cores))")
    ap.add_argument("--seeds", type=int, default=None, help="number of seeds (default 30, the paper setting)")
    ap.add_argument("--smoke", action="store_true", help="2 seeds, output in results_smoke/")
    ap.add_argument("--out", default=None, help="output directory (default results/ or results_smoke/)")
    ap.add_argument("--only", default=None); ap.add_argument("--skip", default="")
    ap.add_argument("--force", action="store_true"); ap.add_argument("--validate", action="store_true"); ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    seeds = a.seeds or (2 if a.smoke else 30)
    out = Path(a.out).resolve() if a.out else ROOT / ("results_smoke" if a.smoke else "results")   # absolute: the scripts run with cwd=out
    (out / "logs").mkdir(parents=True, exist_ok=True)
    names = [s for s in STAGES if (a.only is None or s in a.only.split(",")) and s not in a.skip.split(",")]
    plan = [(n, STAGES[n]) for n in names]
    if a.validate: plan = [("validate", OPTIONAL["validate"])] + plan
    if a.pilot: plan = plan + [("pilot", OPTIONAL["pilot"])]
    env = dict(os.environ, PYTHONPATH=str(CORE) + os.pathsep + os.environ.get("PYTHONPATH", ""), VQC_JOBS=str(a.jobs), VQC_NSEEDS=str(seeds),
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONHASHSEED="0", MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
    # environment description (hardware, versions, data source) for Section 4.1
    subprocess.run([sys.executable, str(ROOT / "collect_env.py"), "--out", str(out), "--jobs", str(a.jobs), "--seeds", str(seeds)], env=env, cwd=out)
    timings = {}; t_all = time.time()
    print(f"output: {out} | seeds: {seeds} | jobs: {a.jobs} | stages: {[n for n, _ in plan]}", flush=True)
    for stage, steps in plan:
        for script, args, done in steps:
            tag = f"{stage}:{script}{' ' + ' '.join(args) if args else ''}"
            if done and (out / done).exists() and not a.force:
                print(f"[skip ] {tag} ({done} exists)", flush=True); continue
            t0 = time.time(); log = out / "logs" / (Path(script).stem + ("_" + "".join(args) if args else "") + ".log")
            print(f"[start] {tag}", flush=True)
            with open(log, "w", encoding="utf-8") as fh:
                rc = subprocess.run([sys.executable, str(locate(script)), *args], cwd=out, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
            dt = time.time() - t0; timings[tag] = round(dt)
            print(f"[{'done ' if rc == 0 else 'FAIL '}] {tag} in {dt/60:.1f} min (log: {log})", flush=True)
            if rc != 0:
                print(f"Stopped. Look at the end of {log}, fix the ENVIRONMENT problem (not the experiment) and re-run the same command; finished steps are skipped.")
                json.dump(timings, open(out / "timings.json", "w"), indent=1); sys.exit(rc)
    json.dump(dict(timings, total_minutes=round((time.time() - t_all) / 60, 1)), open(out / "timings.json", "w"), indent=1)
    print(f"ALL DONE in {(time.time() - t_all) / 60:.1f} min. Next: python compare_results.py --mine {out.name} --reference results (optional afterwards: python tidy_results.py --dir {out.name})", flush=True)

if __name__ == "__main__":
    main()
