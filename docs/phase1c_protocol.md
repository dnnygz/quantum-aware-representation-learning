# Phase 1C protocol gate: leakage-safe classical baseline

Status: **proposed and implemented for review; benchmark not executed**.

## Scientific question

How much predictive signal for baseline ER status can conventional
machine-learning models extract from the usable transcriptomic representation
before qubit-constrained feature reduction?

The canonical 80/20 patient split is preserved. The development cohort alone
is divided into five stratified inner folds. The locked test cannot participate
in preprocessing fitting, hyperparameter/model selection, threshold selection,
or calibration. It is evaluated once after all choices are frozen.

## Pipeline evaluated inside every inner fold

`raw counts -> training CPM filter -> frozen-reference training TMM -> log2-CPM prior=2 -> exact training zero-variance filter -> training StandardScaler -> classifier`

Every patient-distribution-dependent operation implements
`fit(inner_train) / transform(inner_validation)`. Phase 1B remains the frozen
preprocessing specification `phase1b_primary_v1`.

## Prespecified models and search spaces

All models use `class_weight="balanced"`; scikit-learn therefore derives class
weights only from the labels supplied to each call to `fit`.

| Model | Solver / fixed choices | Search space |
|---|---|---|
| Logistic Regression L2 | `liblinear`, `dual=True`, `max_iter=10000` | `C = 10^-4, 10^-3, 10^-2, 10^-1, 1, 10` |
| Logistic Regression L1 | `liblinear`, `dual=False`, `max_iter=10000` | same `C` grid |
| Linear SVM | `LinearSVC`, L2, squared hinge, `dual="auto"`, `max_iter=20000` | same `C` grid |
| RBF-SVM | `SVC`, `gamma="scale"`, no probability calibration, 1024 MB cache | `C = 10^-3, 10^-2, 10^-1, 1, 10` |

`liblinear` supports both L1 and L2 binary logistic regression and is suitable
for this small-sample setting. The L2 formulation is dual because each fold
has far fewer patients than genes. `LinearSVC(dual="auto")` makes the analogous
choice from fold dimensions.

RBF-SVM is retained as a secondary prespecified baseline. Its training cost is
at least quadratic in sample count, but the development cohort has fewer than
one thousand patients and its kernel matrix is modest. The high-dimensional
distance calculation is still materially more expensive than the linear
models, so gamma is fixed to the scale-derived value and only five C values are
tested. This avoids a broad two-dimensional kernel search. No test observation
or performance result informed this decision.

L1 sparsity is reported only as a property of that classifier. Its nonzero
features cannot be reused as a VQC feature selector in Phase 1C.

## Selection, imbalance and classification thresholds

- Primary selection metric: mean validation ROC-AUC across the five inner
  folds.
- Secondary metrics: average precision (reported as PR-AUC), balanced
  accuracy, ER+ sensitivity, ER- specificity, ER+ F1, accuracy and confusion
  matrix.
- Deterministic tie break: smallest C, then lexicographic parameter order.
- Class imbalance: fixed `class_weight="balanced"`, calculated independently
  within every training fold.
- Logistic threshold: probability `>= 0.5`.
- Linear and RBF SVM threshold: decision value `>= 0`.
- Thresholds are not optimized. Calibration is not performed in the primary
  baseline.

After selection, the entire raw-count pipeline is refit on all development
patients and evaluated once on locked test. Inner-CV metrics for the selected
hyperparameter are tuning estimates and can be optimistic from selection; the
locked test is the primary unbiased generalization estimate.

## Predictions and uncertainty

The approved experiment will persist one out-of-fold row per development
patient and one final row per locked-test patient with:

`patient_id, y_true, score, predicted_class, model, fold`.

Logistic models use positive-class probability as score; SVMs use the decision
function. No probability calibration is introduced solely to harmonize scores.

Locked-test confidence intervals use 5,000 patient-level stratified bootstrap
resamples with seed 2718. ER+ and ER- patients are sampled with replacement
within class, preserving test class counts. Percentile 95% intervals are
reported for scalar metrics. The same saved bootstrap indices are reused
across models so future comparisons remain paired. The confusion matrix itself
is reported as observed counts; sensitivity and specificity receive intervals.

## Leakage guard and stopping rule

The protocol artifact stores patient IDs plus indices relative to both the
development ordering and canonical cohort ordering. Automated checks require:

- zero locked-test patients in every inner train/validation set;
- train/validation disjointness inside each fold;
- exhaustive coverage of development in every fold;
- exactly one validation appearance per development patient;
- exact ID/index resolution;
- stratified ER distributions;
- byte-equivalent regeneration from the same inputs and seed.

The code exposes a fail-fast locked-test guard for preprocessing fit, model
fit, model selection, threshold selection and calibration. Benchmark execution
remains blocked pending approval of this protocol.

## Out of scope

No PCA, MI, ANOVA, F-score, LASSO feature-budget selection, 4/6/8-feature
reduction, VQC, quantum circuit, noise model or QEM is part of Phase 1C.

## Method references

- [scikit-learn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [scikit-learn LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)
- [scikit-learn SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- [scikit-learn model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
