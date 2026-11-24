# The generalization gap of variational quantum classifiers

Code, data pipelines and results behind the paper *A Controlled Analysis of the Factors Behind the Generalization Gap of Variational
Quantum Classifiers in Simulated Environments*. Everything runs in noise-free classical simulation.

The repository has two parts. Each is a self-contained project with its own dependencies and instructions; run the commands of each
part from its own folder.

| Folder | What it contains |
|---|---|
| [`controlled_study/`](controlled_study/README.md) | **Controlled study (main contribution).** A four-qubit VQC against logistic regression, SVM (RBF), Random Forest, XGBoost and an MLP on four tabular datasets (WDBC, Ionosphere, Blood Transfusion, synthetic), 30 random stratified splits per dataset. Six factors behind the train-test gap are varied one at a time: representation (H1), encoding (H2), capacity and readout (H3), training-set size (H4), optimization (H5) and a classical teacher (H6). Includes our own NumPy statevector simulator, validated against PennyLane. |
| [`external_case_study/`](external_case_study/README.md) | **External case study.** Quantum-aware representation learning for ER-status classification on TCGA-BRCA gene-expression data, with fixed PCA and trainable encoders, VQCs and classical controls. It motivated the question and is reported descriptively in the paper; it does not enter the verdicts on H1-H6. |

The two parts use different circuits and protocols, so their numbers are not directly comparable. The external study measures only the
last qubit, uses angles pi*tanh(z/2), trains with AdamW for 50 epochs and evaluates on validation folds of the development set;
the controlled study reads out all four qubits, uses full-batch Adam for 200 epochs and evaluates on held-out test splits.

[`docs/paper_mapping.md`](docs/paper_mapping.md) lists which file backs each table and figure of the paper.

## Quick start

```bash
# controlled study (smoke test: 10-20 minutes; full run: about 3.5 hours with 4 parallel jobs)
cd controlled_study
pip install -r requirements.txt
python run_all.py --smoke --jobs 2

# external case study (needs the GDC download, several GB; see its README)
cd external_case_study
python -m pip install -e '.[dev]'
python -m pytest
```

## Layout

```
controlled_study/        core/, experiments/, analysis/, tables/, checks/, data/, results/ (run reported in the paper), run_all.py
external_case_study/     src/, configs/, tests/, docs/, reports/, notebooks/, pyproject.toml, Makefile
docs/                    paper_mapping.md
```
