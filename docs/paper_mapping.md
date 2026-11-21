# From the paper to the files

Paths are relative to the repository root. In the first table, `results/` is `controlled_study/results/` (run reported in the paper, one folder per experiment).

## Controlled study (`controlled_study/`)

| Paper | Content | Files |
|---|---|---|
| Table I, Baseline text | VQC and five classical models, test AUC and gap, paired tests (Wilcoxon, Holm), label randomization | `results/e1_baseline/`: `e1_summary.csv`, `e1_paired.csv`, `e1_shuffled.csv`, `e1_results.csv` |
| H1 text | Input representation (all features, PCA-4, ANOVA-4) | `results/h1_representation/`: `h1_summary.csv`, `h1_paired.csv`, `h1_results.csv` |
| Table II, H2 text | Encoding: synthetic tasks (part B) and real datasets (part A) | `results/h2_encoding/`: `h2B_summary.csv`, `h2A_summary.csv`, `h2_paired.csv`, `fig_h2.png` |
| Table III, H3 text | Capacity and readout, effective parameters, shuffled labels | `results/h3_capacity_readout/`: `h3_summary_ep200.csv`, `eff_params.csv`, `h3_paired.csv`, `h3_shuffled.csv` |
| Fig. 3, H4 text | Training-set size | `results/h4_training_size/`: `h4_summary.csv`, `h4_paired.csv`, `fig_h4.png` |
| H5 text | Optimization: epochs, learning rate, initialization | `results/h5_optimization/`: `h5_summary.csv`, `h5_paired.csv`, `fig_h5.png` |
| Table IV, H6 text | Classical teacher: distillation and learned representation | `results/h6_classical_teacher/`: `h6_summary.csv`, `h6_paired.csv` |
| Section IV-A | Hyper-parameter pilot, environment | `results/pilot/` (`pilot.csv`, `pilot.log`), `results/validation/` (`check_pennylane.log`, `check_enc.log`), `results/run_info/` (`environment.md`, `environment.json`) |

Figures 1 and 2 (circuit and pipeline) are drawings and have no data file.

## External case study (TCGA-BRCA, ER status)

| Paper | Content | Files |
|---|---|---|
| Table V, external study text | Five pipelines, train and validation AUC, gap, Brier, paired counts over 5 folds x 3 seeds | `external_case_study/reports/phase2/cv_5fold_50e_v1/` (`generalization.csv`, `metrics.csv`, `paired.csv`, `paired_fold_means.csv`, `summary.csv`, `report.md`) |
| Protocol | A/B experiments, folds, training settings | `external_case_study/docs/phase2_ab.md`, `external_case_study/docs/phase2_ab_results.md`, `external_case_study/configs/phase2_*.yaml` |
| Cohort | Case selection and ER label | `external_case_study/README.md` (cohort section), `external_case_study/docs/data_decisions.md`, `external_case_study/configs/label_semantics.yaml` |

The two parts use different circuits and protocols (the external study measures only the last qubit, uses angles pi*tanh(z/2),
AdamW for 50 epochs and five development folds), so their numbers are not directly comparable. The external study uses the validation
folds of the development set, not the locked test set, and its three seeds repeat the same patients; it is descriptive and does not
enter the verdicts of the paper.
