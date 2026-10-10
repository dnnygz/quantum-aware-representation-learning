# Phase 1C — leakage-safe classical reference baseline

Status: **complete and frozen (2026-10-08)**.

## Design

All model selection used five stratified folds within the 835-patient
development set. The 209-patient locked test was excluded from preprocessing,
feature decisions, model-family and hyperparameter selection, thresholds and
calibration. Every fold started from raw counts and fitted the complete Phase
1B transform on inner training patients only. The complete 115-row model × C ×
fold table is `phase1c_cv_results.csv`; all configuration summaries are in
`phase1c_cv_summary.csv`.

ROC-AUC was the sole primary selection criterion. The classical reference
family was also selected before test access from mean inner-CV ROC-AUC. Fixed
thresholds were 0.5 for logistic regression and 0 for SVM; no calibration or
threshold tuning was performed.

## Selected configurations and inner-CV performance

Values are mean ± sample SD over five validation folds.

| Model | C | ROC-AUC | ER+ PR-AUC | Balanced accuracy | ER+ sensitivity/recall | ER− specificity/recall | ER+ F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic L1 | 0.01 | 0.9605 ± 0.0209 | 0.9851 ± 0.0073 | 0.9173 ± 0.0344 | 0.8977 ± 0.0149 | 0.9368 ± 0.0577 | 0.9369 ± 0.0152 | 0.9066 ± 0.0226 |
| Logistic L2 | 0.0001 | 0.9537 ± 0.0233 | 0.9817 ± 0.0102 | 0.9151 ± 0.0194 | 0.9039 ± 0.0322 | 0.9263 ± 0.0506 | 0.9386 ± 0.0143 | 0.9090 ± 0.0196 |
| Linear SVM | 0.0001 | 0.9530 ± 0.0194 | 0.9826 ± 0.0077 | 0.9235 ± 0.0249 | 0.9364 ± 0.0168 | 0.9105 ± 0.0440 | 0.9542 ± 0.0118 | 0.9305 ± 0.0178 |
| RBF-SVM | 1 | 0.9490 ± 0.0295 | 0.9797 ± 0.0132 | 0.9141 ± 0.0368 | 0.9597 ± 0.0168 | 0.8684 ± 0.0617 | 0.9604 ± 0.0158 | 0.9389 ± 0.0245 |

Logistic L1 was frozen as the definition of `P_classical-full` because it had
the highest selected-configuration mean CV ROC-AUC. This choice was made before
the locked test was accessed.

Fold-local final feature counts were 18,480, 18,451, 18,424, 18,442 and 18,455.
The development refit retained 18,438 features. No zero-variance feature was
removed in that refit.

## Locked-test performance

Intervals are paired, patient-level, class-stratified percentile bootstrap 95%
intervals from the same 5,000 saved resamples (seed 2718) for every model.

| Model | ROC-AUC (95% CI) | ER+ PR-AUC (95% CI) | Balanced accuracy (95% CI) | ER+ sensitivity/recall (95% CI) | ER− specificity/recall (95% CI) | ER+ F1 (95% CI) | Accuracy (95% CI) |
|---|---|---|---|---|---|---|---|
| Logistic L1 | 0.9660 (0.9372–0.9896) | 0.9908 (0.9822–0.9974) | 0.9122 (0.8741–0.9444) | 0.8457 (0.7840–0.9012) | 0.9787 (0.9362–1.0000) | 0.9133 (0.8767–0.9450) | 0.8756 (0.8278–0.9187) |
| Logistic L2 | 0.9626 (0.9320–0.9871) | 0.9894 (0.9793–0.9967) | 0.9276 (0.8923–0.9568) | 0.8765 (0.8210–0.9259) | 0.9787 (0.9362–1.0000) | 0.9311 (0.8986–0.9585) | 0.8995 (0.8565–0.9378) |
| Linear SVM | 0.9518 (0.9057–0.9854) | 0.9811 (0.9546–0.9963) | 0.9324 (0.8930–0.9660) | 0.9074 (0.8580–0.9506) | 0.9574 (0.8936–1.0000) | 0.9453 (0.9180–0.9686) | 0.9187 (0.8804–0.9522) |
| RBF-SVM | 0.9670 (0.9345–0.9904) | 0.9895 (0.9772–0.9974) | 0.9403 (0.8978–0.9753) | 0.9444 (0.9074–0.9753) | 0.9362 (0.8511–1.0000) | 0.9623 (0.9397–0.9814) | 0.9426 (0.9091–0.9713) |

The convention is `ER_positive=1` and `ER_negative=0`. Confusion cells mean
`TN=true ER− predicted ER−`, `FP=true ER− predicted ER+`, `FN=true ER+
predicted ER−`, and `TP=true ER+ predicted ER+`. Values were: Logistic L1
`(46, 1, 25, 137)`, Logistic L2 `(46, 1, 20, 142)`, Linear SVM
`(45, 2, 15, 147)`, and RBF-SVM `(44, 3, 9, 153)`.

RBF-SVM has the numerically largest test ROC-AUC, but it was not selected by
the preregistered development-only rule. Its difference from Logistic L1 is
0.0011, with strongly overlapping uncertainty. It is therefore not interpreted
as evidence of a unique winner.

ER+ prevalence is 77.30% in the canonical cohort and 77.51% in locked test;
these are the corresponding no-skill ER+ average-precision baselines. Frozen
legacy PR-AUC is ER+ average precision, not a class-symmetric measure. Locked
test ER− average precision, after reversing both label and score, is 0.8166
for Logistic L1, 0.8266 for Logistic L2, 0.8192 for Linear SVM and 0.8908 for
RBF-SVM. High ER+ PR-AUC must be interpreted with prevalence and minority-class
performance. See the [semantic audit](phase1c_label_metric_semantic_audit.md).

## Sparsity and runtime

At selected `C=0.01`, Logistic L1 retained 3–6 nonzero coefficients across the
five CV folds (mean 5.0) and seven in the full-development refit. This is a
model property only; these genes are not an authorized feature selector for
Phase 2 or a VQC.

The development-only CV stage took 131.20 s wall time. Mean per selected-model
fold preprocessing time was 5.99 s. Mean training/inference seconds per fold
were respectively: L1 0.260/0.025, L2 0.247/0.029, Linear SVM 0.527/0.024 and
RBF-SVM 0.422/0.350. The final stage took 58.244 s observed wall time.

After all predictions and bootstrap draws had been atomically persisted, the
first report write failed because the mixed integer/string `fold` field in the
L1 sparsity table was not Arrow-compatible. Reports were reconstructed from
the existing prediction/bootstrap/model checkpoints without loading raw
expression or labels and without a second test inference. Exact final-stage
per-step timings existed only in memory and were lost; they are recorded as
null with this limitation in the machine-readable report. The serialization
type is corrected for clean future runs, and the access receipt records the
recovery explicitly.

## Quality and interpretation

All leakage checks pass: the test is absent from OOF predictions, every
development patient has exactly one OOF prediction per final model, every test
patient has exactly one final prediction per model, selected-parameter hashes
match the pre-test CV checkpoint, thresholds are fixed, calibration is absent,
all models use the same bootstrap indices, and canonical cohort hashes are
unchanged. No convergence warning occurred in 115 CV fits.

`P_classical-full` is the Logistic-L1 locked-test ROC-AUC 0.9660 (95% CI
0.9372–0.9896), selected by development-only CV. This is the frozen reference
for later `Delta_representation`; it is not a claim that L1 is inherently
superior to the other models.

Primary machine-readable artifacts are `phase1c_locked_test_performance.json`,
`phase1c_leakage_qc.json`, `phase1c_provenance.json`, and the data files under
`data/processed/phase1c_classical_baseline_v1/`.
