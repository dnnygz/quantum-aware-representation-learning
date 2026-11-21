Run reported in the paper (laptop, UCI data). One folder per experiment, each with its CSV files, figures, LaTeX fragments and `logs/`:

| Folder | Experiment |
|---|---|
| `e1_baseline/` | VQC against LR, SVM, Random Forest, XGBoost and MLP; label randomization |
| `h1_representation/` | H1: input representation |
| `h2_encoding/` | H2: encoding (synthetic tasks and real datasets) |
| `h3_capacity_readout/` | H3: capacity and readout, effective parameters |
| `h4_training_size/` | H4: training-set size |
| `h5_optimization/` | H5: epochs, learning rate, initialization |
| `h6_classical_teacher/` | H6: classical teacher |
| `pilot/` | hyper-parameter pilot (validation split, seeds 1000-1004), see "Setup that must not change" in `../README.md` |
| `validation/` | logs of the simulator checks (`check_enc.py`, `check_pennylane.py`) |
| `run_info/` | `environment.json` / `environment.md` (hardware and versions), run logs and timings |

Inside an experiment folder: `*_results.csv` has one row per model, seed and condition; `*_summary.csv` and `*_paired.csv` hold group means and
paired bootstrap intervals; `fig_*` are the figures.

`run_all.py` writes everything into one flat folder; `tidy_results.py` sorted this one into the layout above without touching the contents.
Timings: `run_info/timings_all_sessions.json` has the minutes per stage of the full run (it was interrupted three times and resumed);
`timings_full_run_last_session.json` is the last of those sessions; `timings.json` belongs to the later run of the simulator checks and the
pilot (`run_all_pilot_validate.log`).
