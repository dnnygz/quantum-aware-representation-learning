# Phase 1C label and metric semantic audit

Status: **pass — no numerical discrepancies**.

This audit reads only canonical labels, frozen locked-test predictions and the
existing performance report. It does not load expression, fit preprocessing,
train a model, select hyperparameters or perform inference. SHA-256 hashes of
all protected prediction, selection and performance artifacts were identical
before and after execution.

## Governing convention

`ER_positive = 1` is the positive class and `ER_negative = 0`. Persisted
logistic scores are `P(ER+)`; persisted SVM decision scores increase toward
ER+. Therefore:

- ER+ average precision uses `(er_label, persisted_score)`;
- ER− average precision uses `(1-er_label, 1-P(ER+))` for logistic models;
- ER− average precision uses `(1-er_label, -decision_score)` for SVMs;
- confusion matrices use labels `[ER_negative=0, ER_positive=1]` and ravel as
  `(TN, FP, FN, TP)`.

The canonical cohort contains 807 ER+ and 237 ER− patients: ER+ prevalence is
`807/1044 = 0.77299`. Locked test contains 162 ER+ and 47 ER− patients: the ER+
average-precision baseline is `162/209 = 0.77512`; the ER− baseline is
`47/209 = 0.22488`.

## Logistic L1 manual reconciliation

The frozen matrix `(TN, FP, FN, TP) = (46, 1, 25, 137)` means:

- `TP=137`: true ER+ predicted ER+;
- `FN=25`: true ER+ predicted ER−;
- `TN=46`: true ER− predicted ER−;
- `FP=1`: true ER− predicted ER+.

Thus:

- ER+ sensitivity/recall = `137/(137+25) = 0.845679`;
- ER− specificity/recall = `46/(46+1) = 0.978723`;
- balanced accuracy = `(0.845679+0.978723)/2 = 0.912201`.

These reproduce the previously displayed 0.8457, 0.9787 and 0.9122.

## Class-explicit locked-test metrics

| Model | ER+ AP baseline | ER+ AP / PR-AUC | ER− AP baseline | ER− AP / PR-AUC | ER+ sensitivity/recall | ER− specificity/recall | Balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic L1 | 0.7751 | 0.9908 | 0.2249 | 0.8166 | 0.8457 | 0.9787 | 0.9122 |
| Logistic L2 | 0.7751 | 0.9894 | 0.2249 | 0.8266 | 0.8765 | 0.9787 | 0.9276 |
| Linear SVM | 0.7751 | 0.9811 | 0.2249 | 0.8192 | 0.9074 | 0.9574 | 0.9324 |
| RBF-SVM | 0.7751 | 0.9895 | 0.2249 | 0.8908 | 0.9444 | 0.9362 | 0.9403 |

Under the fixed ER+=positive convention, ER− specificity is the same number
as ER− recall. If each class is separately treated as one-vs-rest positive,
ER+ one-vs-rest specificity equals ER− recall, and ER− one-vs-rest specificity
equals ER+ recall. Both are named explicitly in the machine-readable audit.

## Macro descriptive metrics

| Model | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|
| Logistic L1 | 0.8203 | 0.9122 | 0.8465 |
| Logistic L2 | 0.8450 | 0.9276 | 0.8727 |
| Linear SVM | 0.8683 | 0.9324 | 0.8932 |
| RBF-SVM | 0.9055 | 0.9403 | 0.9211 |

Macro recall equals balanced accuracy in this binary task. These are
descriptive additions and did not participate in model selection.

## Reconciliation conclusion

Thirty-two comparisons across four models—ER+ AP, ER+ recall, ER− recall,
balanced accuracy and all four confusion cells—matched frozen values at
absolute tolerance `1e-12`. No label inversion or numerical discrepancy was
found. The only defect was semantic ambiguity in legacy names, a reporting
issue rather than a computational error. Frozen JSON/CSV fields were not
renamed because that would invalidate provenance hashes; the explicit mapping
now lives in configuration, code, tests and this audit layer.
