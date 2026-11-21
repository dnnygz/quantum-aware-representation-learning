# Generalization gap of variational quantum classifiers (noise-free simulation)

Code, data and results for the controlled study in the paper *A Controlled Analysis of the Factors Behind the Generalization Gap of
Variational Quantum Classifiers in Simulated Environments* (see the [repository overview](../README.md) and the
[mapping from paper tables to files](../docs/paper_mapping.md)). A four-qubit VQC is compared with LR, SVM (RBF), Random Forest,
XGBoost and an MLP under equal inputs, comparable capacity and the same training budget, with 30 random stratified 70/30 splits per
dataset, and the factors behind its train-test gap are varied one at a time (H1 representation, H2 encoding, H3 capacity/readout,
H4 training-set size, H5 optimization, H6 classical teacher).

## Layout

```
run_all.py           runs everything in order (see below)
collect_env.py       writes environment.json / environment.md (hardware and versions)
compare_results.py   compares a re-run with the run reported in the paper (CSV by CSV, tolerance 1e-3)
tidy_results.py      sorts the flat output folder of run_all.py into one folder per experiment
core/                simulator (vqc.py, vqc_enc.py) and shared helpers (common.py: data, preprocessing, models)
experiments/         run_*.py (one per experiment), eff_params.py, pilot.py
analysis/            analyze_*.py: summaries, paired statistics and figures
tables/              gen_sec42*.py: LaTeX fragments (see below)
checks/              check_enc.py, check_pennylane.py: simulator checks
data/                mirror/ (copies of the three UCI files, for offline use), download_data.py (UCI originals)
results/             run reported in the paper (laptop, see results/run_info/environment.md), one folder per experiment
```

`tables/gen_sec42*.py` write LaTeX fragments of the Results section of a longer draft (`sec42_*.tex`, `tables_sec42.tex` in each experiment folder of `results/`); the
numbers quoted in the paper come from the CSV files, see [docs/paper_mapping.md](../docs/paper_mapping.md).

## Reproduce

```bash
cd controlled_study
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python data/download_data.py                              # UCI originals (checked against the copies in data/mirror/)
python run_all.py --smoke --jobs 2                        # 2 seeds per experiment, about 10-20 minutes: checks that everything works
python run_all.py --jobs 6 --out results_rerun            # full run, 30 seeds; use about half of your cores
python run_all.py --validate --pilot --out results_rerun  # optional: simulator check against PennyLane (pip install pennylane) and the hyper-parameter pilot
python compare_results.py --mine results_rerun --reference results   # difference with the run reported in the paper
python tidy_results.py --dir results_rerun                   # optional: sort the output into one folder per experiment, like results/
```

Always pass `--out` for a full re-run: a stage whose main CSV already exists is skipped, so running with the default `results/`
would reuse the files of the paper. Stages: `e1` (baseline and label randomization), `h5`, `h3`, `h4`, `h1`, `h2`, `h6`. Each experiment
script writes its CSV when it finishes, then `analyze_*.py` computes summaries, paired statistics (bootstrap intervals, Wilcoxon, Holm
for the baseline) and figures. Use `--only h2,h6` or `--skip h2` to select stages and `--force` to redo one. To run a single script by hand, put `core/` on the path and work from the output folder, e.g. `PYTHONPATH=core python experiments/run_h4.py`. The next section says what to
expect when you compare your run with the one in `results/`.

## Reproducibility and validation

Runs are deterministic given the seeds on the same software. With other versions of scikit-learn or XGBoost (or another CPU or BLAS)
the results agree closely but not bit by bit, XGBoost most of all; `compare_results.py` reports the difference file by file and
`results/run_info/environment.json` records the versions of the run reported in the paper.

Simulator checks (run on the same laptop, logs in `results/validation/`): `checks/check_enc.py` checks the gradients of the re-uploading circuits
against finite differences (5.6e-11) and that the extended simulator reproduces the base VQC (logits differ by 2e-14); `checks/check_pennylane.py` compares outputs and parameter-shift gradients
with PennyLane 0.42.3 (outputs differ by 4e-16, gradients against finite differences of PennyLane by 3.5e-11, circuit without
entanglement by 8e-16).

## Setup that must not change

Frozen from a pilot on a validation split of the training part (`results/pilot/pilot.csv`): learning rate 0.05, 200 epochs (shared by VQC
and MLP), full-batch Adam. The pilot uses seeds 1000-1004, disjoint from the evaluation seeds 0-29; the grid is learning rate
{0.02, 0.05, 0.1} x epochs {100, 200}, five datasets (WDBC, Ionosphere, Sonar, Synthetic, Blood Transfusion), 300 rows. The setting was
chosen on the first four datasets (Blood Transfusion joined the design later): learning rate 0.05 with 200 epochs has the highest
mean validation AUC when VQC and MLP share one setting (0.8558). It is also the best shared setting on all five datasets (0.8329).
On their own, the MLP prefers 0.02 / 200 epochs, and the VQC prefers 0.05 / 200 on the first four datasets and 0.10 / 200 on all five
(0.8242 against 0.8224). The test split is never used to choose a configuration.

VQC: 4 qubits, angle encoding RY(x), two layers of RY/RZ with a CNOT chain, readout of all four Pauli-Z expectations followed by a
linear head, exact noise-free statevector, parameter-shift gradients (validated against PennyLane, see above). Preprocessing is fitted
on the training split only: standardization, PCA to four components (skipped for Blood Transfusion), rescaling to [-pi, pi].

## Known limitations

- The 30 seeds are re-splits of the same dataset, so intervals are optimistic (H2 on synthetic data draws a new sample per seed).
- Daggers in the tables of H1-H6 mark bootstrap intervals that exclude zero, without multiple-comparison correction.
- Scaling the angles to [-pi, pi] maps both ends of a feature to the same quantum state (see H2 and H6).
- Sonar appears in the pilot and the baseline and is excluded from the final design; Blood Transfusion joined the design after the other datasets had been analyzed (four native features, so PCA is skipped).
- Ionosphere, Sonar and Blood Transfusion come from UCI (`data/uci/`, downloaded by `data/download_data.py`); `data/mirror/` holds copies for offline use, and the script checks that the two are identical.
