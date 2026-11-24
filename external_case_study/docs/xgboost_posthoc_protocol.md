# XGBoost post-hoc complementary classical baseline: protocol gate

Status: **approved, executed and frozen on 2026-10-09**.

## Scientific role and non-retroactivity

This experiment adds one gradient-boosted-tree family after the locked-test
results of the four prespecified Phase-1C models were observed. It is therefore
labelled **post-hoc complementary classical baseline**. It cannot replace,
reselect or reinterpret the primary reference:

`P_classical-full(primary) = Logistic L1 locked-test ROC-AUC = 0.9660`.

The XGBoost result will be reported separately and cannot retroactively alter
the Phase-1C protocol, predictions, selected hyperparameters or hashes.

## Software and fixed estimator configuration

Execution is proposed with `xgboost==3.4.1`, `XGBClassifier`, Python 3.12+, CPU
`tree_method="hist"`, `objective="binary:logistic"`, `booster="gbtree"`,
`grow_policy="depthwise"`, `max_bin=256`, `eval_metric="auc"`, four CPU
threads and seed 1729. GPU is excluded from the primary run to avoid
hardware-specific execution differences. Early stopping is disabled: each
candidate has a fixed prespecified number of boosting rounds.

The exact package and its OpenMP runtime were installed only after protocol
approval. Transformed features are passed as `float32`, preserving column order.

## Partitions and preprocessing

The experiment reuses, byte-for-byte:

- the 835-patient development and 209-patient locked-test partition;
- the five Phase-1C inner folds, seed 1729;
- `ER_positive=1`, `ER_negative=0`;
- the Phase-1B preprocessing specification;
- the existing 5,000 paired bootstrap indices.

Each inner fold performs:

`raw counts -> inner-train CPM filter -> inner-train TMM -> log2-CPM prior=2 -> inner-train exact-zero-variance filter -> XGBoost`.

No `StandardScaler` is used. Preprocessing is fitted once per inner-training
partition and reused across its six candidates; validation never contributes
to a fitted preprocessing parameter. There is no supervised feature selection.
The selected configuration is refit from raw counts on all 835 development
patients before exactly one XGBoost inference on locked test.

## Class imbalance

The typical XGBoost heuristic is `scale_pos_weight = n_negative/n_positive`.
Here positive means ER+, the majority class, so the development value will be
approximately 0.29 rather than greater than one. That value has the correct
*relative* class-weight ratio for balancing, but fixes ER− at weight 1 and can
change the absolute Hessian scale seen by `min_child_weight`, `reg_lambda` and
`reg_alpha`.

The primary strategy is therefore:

- fixed `scale_pos_weight=1`;
- pass fold-local sample weights to `fit`;
- for class `c`, `w_c = n_train/(2*n_train,c)`;
- derive weights only from inner-training labels;
- derive final-refit weights only from the 835 development labels;
- never weight validation/test rows for metric calculation.

This gives each class total training weight `n_train/2` and exactly matches the
definition of scikit-learn `class_weight="balanced"`. Weighting strategy is
not a candidate dimension and cannot be selected by performance.

## Candidate configurations

All omitted parameters are the fixed values above.

| ID | Trees | Depth | LR | Min child weight | Row sample | Column sample | L1 | L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C01 shallow strong-reg | 250 | 2 | 0.03 | 10 | 0.8 | 0.10 | 1.0 | 10 |
| C02 depth-3 strong-reg | 300 | 3 | 0.03 | 10 | 0.8 | 0.10 | 1.0 | 10 |
| C03 depth-3 moderate | 250 | 3 | 0.05 | 5 | 0.8 | 0.10 | 0.5 | 5 |
| C04 depth-3 wider columns | 300 | 3 | 0.03 | 10 | 0.8 | 0.20 | 1.0 | 10 |
| C05 depth-4 high-reg | 300 | 4 | 0.03 | 15 | 0.8 | 0.10 | 2.0 | 20 |
| C06 shallow faster | 150 | 2 | 0.05 | 5 | 1.0 | 0.20 | 0.5 | 5 |

Depth is restricted to 2–4 because development folds contain roughly 668
patients while preprocessing retains about 18.4k genes. Column subsampling of
10–20%, shallow trees, nonzero L1/L2 penalties and minimum child weights of
5–15 constrain the opportunity for unstable high-order interactions. The six
configurations are a structured sensitivity set, not a Cartesian grid.

## Selection and deterministic tie-break

The only selection evidence is mean inner-validation ROC-AUC over the five
frozen folds. Selection is the strict argmax. Only a numerical equality within
`abs(delta) <= 1e-12` invokes the deterministic complexity ordering:

1. lower `max_depth`;
2. lower `n_estimators * 2^max_depth`;
3. lower `colsample_bytree`;
4. higher `min_child_weight`;
5. higher `reg_lambda`;
6. higher `reg_alpha`;
7. lower `learning_rate`;
8. lexicographic candidate ID.

The selected candidate and its complete fold results are frozen before locked
test access. No parameter may change afterward.

## Metrics and bootstrap

ROC-AUC remains the primary metric. The fixed classification threshold is
`P(ER+) >= 0.5`; no calibration or threshold tuning is allowed. Reports use
explicit names: ER+ AP, ER− AP, balanced accuracy, ER+ recall, ER−
recall/specificity, macro F1, accuracy, and named ER−/ER+ confusion cells.

Test intervals reuse the existing `bootstrap_indices.npz` and its frozen
patient order exactly. The indices are not regenerated. This preserves paired
comparisons with all four original models.

## Fit count and estimated cost

Six configurations × five folds produce 30 CV model fits. One final refit gives
31 model fits total. Preprocessing is fitted five times for CV and once for the
development refit. The candidates construct 7,750 trees during CV plus 150–300
trees in the final refit.

On the current CPU environment, a conservative planning range is 5–20 minutes
and 1–4 GB peak memory. This is an estimate, not a benchmark; actual per-fold
preprocessing, training and inference times will be persisted.

## Artifacts and guards planned after approval

Planned outputs are complete candidate/fold metrics, selected-configuration
receipt, OOF predictions, one locked-test prediction per patient, paired
bootstrap metrics, model JSON/UBJSON, gain/cover/weight importance, leakage QC,
software/protocol provenance and SHA-256 hashes. Importance is descriptive and
is permanently prohibited as a Phase-2/VQC feature selector.

Tests will cover balanced weight calculation, candidate immutability, tie-break
ordering, fold/test isolation, absence of scaling, feature order, reuse of the
exact bootstrap indices, explicit ER semantics, protected Phase-1C hashes and
the one-use locked-test receipt.

## Methodological risks

- The experiment is post-hoc; test performance is contextual evidence, not a
  new preregistered winner-selection exercise.
- With `p >> n`, many correlated genes offer numerous split candidates and can
  make feature importance unstable even under strong regularization.
- Tree importance is biased toward features that afford useful splits and is
  not causal or stable evidence of biological relevance.
- Balanced sample weights change the target training loss; both class-specific
  AP and recall must accompany aggregate discrimination metrics.
- CPU parallelism and floating-point reductions can limit bitwise portability;
  package version, threads, seed, model file and predictions will be hashed.
- Six candidates provide only a bounded sensitivity analysis; absence of a
  better result does not imply XGBoost was exhaustively optimized.

## Official references

- [XGBoost parameters](https://xgboost.readthedocs.io/en/stable/parameter.html)
- [XGBoost Python interface](https://xgboost.readthedocs.io/en/stable/python/python_intro.html)
- [XGBoost model IO](https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html)
- [XGBoost releases](https://github.com/dmlc/xgboost/releases)

## Frozen execution outcome

All 30 candidate-by-fold fits completed using development data only. Strict
mean-CV ROC-AUC selected `xgb_c06_shallow_faster` (0.96316 ± 0.02522); the
runner recorded that no numerical tie occurred, so the complexity tie-break
was not invoked. The final development refit retained 18,438 genes and the
one-use access receipt records exactly one locked-test inference.

Locked-test ROC-AUC was 0.97413 (frozen paired-bootstrap 95% percentile CI
0.95088–0.99265). The named confusion matrix was TN=46, FP=1, FN=11, TP=151.
The paired ROC-AUC difference versus Logistic L1 had CI −0.00066 to 0.01878;
this is consistent with the experiment's complementary role and does not
replace the preregistered Logistic L1 reference. Complete results and hashes
are in `reports/data/xgboost_posthoc_locked_test_performance.json` and
`reports/data/xgboost_posthoc_provenance.json`.
