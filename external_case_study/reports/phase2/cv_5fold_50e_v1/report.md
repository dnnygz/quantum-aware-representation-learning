# Five-fold development validation of A/B and classical controls

## Scope and principal findings

**Status: completed development cross-validation, not locked-test evaluation.**
The pending folds were executed on 2026-10-28 (America/Lima).
The fixed protocol covers five existing folds, three original seeds (1729,
1730, 1731), and five variants: 75 model/fold/seed results. The 15 historical
fold-0 pilot results were reused; 60 pending results were executed. No model
selection, tuning, early stopping, C0, or C1 was performed. Phase 1 and all
historical experimental artifacts were preserved.

B improves ROC-AUC over A in 14 paired results and ties in one, with mean
B−A = +0.0917. The improvement is much larger outside the original pilot:
fold means range from +0.0148 to +0.1878. This supports a recurring descriptive
improvement of this complete B pipeline over this A pipeline, without isolating
a quantum contribution. PCA4 plus logistic regression (A_classical) has higher
mean ROC-AUC (0.9490) than B (0.9215), and the frozen-encoder probe exceeds B's
ROC-AUC in 12 of 15 pairs. Probability losses and threshold decisions give a
more nuanced comparison.

B's numerical latent rank falls below four in 9/15 final embeddings, in every
fold, including one rank-two solution. Nevertheless, B_classical has lower
variance entropy rank than B despite retaining numerical rank four. Rank
contraction alone is therefore not a sufficient explanation or a justified
regularization target. Strong training/validation separation is recurrent for
both trainable encoders, especially B_classical in log loss. These are development
observations and explanatory hypotheses, not confirmatory causal findings.

## Pre-execution audit and frozen methodology

The pilot configuration is `configs/phase2_pilot_fold0.yaml`, with its fully
resolved configuration in `runs/phase2/pilot_fold0_50e_paired_v1/config.json`.
The existing complete configuration, `configs/phase2_ab.yaml`, differs only in
fold selection. `configs/phase2_cv_remaining.yaml` selects folds 1–4 to avoid
repeating the completed pilot. No training code was changed. The execution
source hashes for data loading, models, training, runner, evaluation, and
diagnostics match the pilot; dependency versions and recorded input hashes
also match. Three older smoke directories exist, but their shorter budgets or
initialization histories are not pooled with this experiment.

The canonical split contains 835 development patients (645 ER+, 190 ER−) and
209 locked-test patients. Only split metadata concerning the latter was used
to enforce isolation. Each fold contains 668 training patients (516 ER+, 152
ER−) and 167 validation patients (129 ER+, 38 ER−). Validation partitions cover
development exactly once. Training sets overlap across folds. The runner uses
Arrow row filtering to materialize only development expression and labels;
its existing provenance routine hashes whole input files as opaque bytes,
without evaluating locked features or labels. No locked rows were loaded for
analysis, plotting, fitting, or selection.

All learned preprocessing is fold-local: CPM≥1 in at least 10% of training
patients; frozen-reference TMM (logratio trim 0.3, sum trim 0.05, weighted);
log2-CPM with prior count 2; exact-zero training variance filtering; gene-wise
standardization; randomized PCA with random_state=1729; and PCA-score
standardization. TMM selects and calibrates its reference using training only;
new samples are transformed individually against that reference. PCA uses four
components for A and 32 for B. Validation transformation never refits these
operations. Tests prohibit fitting nested preprocessing components during
validation transformation. Saved representations were checked for canonical
training IDs, training-selected TMM reference, PCA sample count, and score-scaler
sample count.

| Variant | Representation and head | Training rule |
|---|---|---|
| A | Fixed PCA4 → VQC | 50 epochs |
| A_classical | Same fixed PCA4 → logistic regression | C=1, max_iter=2000 |
| B | Fixed PCA32 → 32→8→4 encoder → VQC | Joint training, 50 epochs |
| B_classical | Same PCA32 → identically initialized encoder → sigmoid linear head | Joint training, 50 epochs |
| B_frozen_probe | Final B encoder frozen → new logistic regression head on z | C=1, max_iter=2000, training only |

The encoder has a Tanh hidden activation and 300 parameters. B and B_classical
have 316 and 305 nominal parameters respectively. The VQC has four qubits,
one RY encoding of π·tanh(z/2), two RY/RZ rotation layers, the CNOT chain
0→1→2→3 after each layer, and Pauli-Z measurement on qubit 3. It outputs
clip((1+expectation)/2, 1e−7, 1−1e−7). Simulation uses CPU float64,
PennyLane default.qubit, analytic expectations (shots=None), and backpropagation.
Four final-layer RZ rotations are structurally redundant and were retained.
Nominal parameter count does not establish matched functional capacity.
Classical heads operate on pre-angular z, without the angular transformation.

Trainable networks use unweighted BCE, AdamW, learning rate 0.01, weight decay
0.001, batch size 32, and exactly 50 epochs (1050 updates). Evaluation uses the
final checkpoint, ER+=1, ER−=0, and a fixed threshold ≥0.5. A/B receive identical
VQC weights and shuffle RNG states within each seed; B/B_classical start with
identical encoders. All 15 pairs of saved initial checkpoints were checked.
Identical seed numbers do not imply identical gradient trajectories or matched
capacity across architectures. Logistic regression uses its original different
optimizer and budget.

ROC-AUC and AP ER+ use ER+ probabilities; AP ER− uses 1−p and ER− as the target.
Balanced accuracy averages sensitivity for the two classes. Log loss is in nats
and lower is better. Brier score and fixed-bin reliability summaries were added
as read-only descriptions of saved probabilities; they did not change the
training or evaluation protocol. Neither loss isolates calibration from
resolution/discrimination. No calibrator was fitted.

## Predictive results and variation

Values below are mean ± sample SD of the **five fold means over seeds**.
They are descriptive, not confidence intervals. `summary.csv` separately reports
the mean within-fold seed SD, and `metrics.csv` retains all 75 results. Folds
have equal size, so equal-fold means also equal the mean across 15 combinations;
this numerical equality does not make those combinations independent. A_classical
is deterministic here: its seed repetitions add no independent evidence.

| Model | ROC-AUC | AP ER+ | AP ER− | Balanced accuracy | Log loss (nats) |
|---|---|---|---|---|---|
| A | 0.8297 ± 0.0918 | 0.9339 ± 0.0433 | 0.6317 ± 0.1666 | 0.7634 ± 0.0638 | 0.4769 ± 0.0737 |
| A_classical | 0.9490 ± 0.0261 | 0.9791 ± 0.0132 | 0.8629 ± 0.0523 | 0.8608 ± 0.0516 | 0.2286 ± 0.0619 |
| B | 0.9215 ± 0.0295 | 0.9637 ± 0.0149 | 0.8148 ± 0.0634 | 0.8815 ± 0.0306 | 0.2867 ± 0.0629 |
| B_frozen_probe | 0.9371 ± 0.0186 | 0.9746 ± 0.0090 | 0.8337 ± 0.0326 | 0.8749 ± 0.0389 | 0.2819 ± 0.0516 |
| B_classical | 0.9345 ± 0.0209 | 0.9741 ± 0.0089 | 0.8249 ± 0.0527 | 0.8738 ± 0.0299 | 0.7585 ± 0.2801 |

For ROC-AUC, mean within-fold seed SD is 0.0024 (A), 0 (A_classical),
0.0159 (B), 0.0134 (probe), and 0.0111 (B_classical). B is more seed-sensitive
than A, but A is much more fold-sensitive. Fold 3 is difficult for all variants:
mean ROC-AUC is 0.6831 for A, 0.8709 for B, 0.9098 for A_classical, 0.9071 for
the probe, and 0.9004 for B_classical. These values do not identify why that
partition is difficult. No fold was excluded or given a different budget.

### Paired B−A comparisons

| Fold | Δ ROC-AUC | Δ AP ER+ | Δ AP ER− | Δ balanced accuracy | Δ log loss |
|---|---|---|---|---|---|
| 0 | +0.0148 | −0.0045 | +0.0471 | +0.0622 | −0.1749 |
| 1 | +0.0668 | +0.0180 | +0.1975 | +0.1087 | −0.1918 |
| 2 | +0.0786 | +0.0263 | +0.0906 | +0.1164 | −0.1814 |
| 3 | +0.1878 | +0.0758 | +0.3162 | +0.1655 | −0.2040 |
| 4 | +0.1107 | +0.0337 | +0.2644 | +0.1379 | −0.1991 |

These differences are first calculated within each fold/seed and then averaged
within fold. Across all pairs, B has higher AP ER− and balanced accuracy and
lower log loss in 15/15 cases; AP ER+ improves in 13/15. The B−A ROC-AUC fold-mean
SD is 0.0639. Direction is recurrent, while magnitude depends strongly on fold
and seed. A pipeline change from A to B includes PCA dimensionality and a learned
encoder; it cannot be attributed solely to end-to-end quantum training.

No t-test, naive bootstrap of the 15 rows, or standard error divided by √15 is
used. Seeds repeat the same patients; fold training sets overlap; fold 0 has
already informed the research discussion. Results are development evidence,
without a fresh confirmatory cohort or independent replication.

### What the controls establish

The frozen probe has higher ROC-AUC in 12/15 pairs, with mean probe−B=+0.0157;
its fold-mean improvement ranges from +0.0038 to +0.0362. AP ER+ favors the probe
in 11/15 pairs, AP ER− in 10/15. B has higher balanced accuracy in 8 pairs, ties
in 2, and is lower in 5: a fixed threshold can produce a different conclusion
from ranking. Mean balanced accuracy favors B by 0.0067.

Probability loss is variable: B has lower log loss than the probe in 7/15 pairs
and higher in 8/15. The probe's mean log loss is lower by 0.0047, but fold means
favor B in folds 0, 1, and 4 and favor the probe in folds 2 and 3. Mean Brier
scores are 0.0709 (B), 0.0682 (probe), 0.0820 (B_classical), 0.0619 (A_classical),
and 0.1532 (A). The probe has lower Brier score in 11/15 pairs. Fixed-bin reliability
plots describe probability behavior, with sparse bins and patient repetitions;
no independent-sample calibration intervals or calibration superiority claim is
made. B/probe score Spearman correlations span 0.4553–0.7393: their scores are
not interchangeable.

B_classical has higher ROC-AUC than B in 10/15 pairs (mean difference +0.0130),
but B has lower log loss in all 15 (mean difference −0.4718). A_classical exceeds
A in ROC-AUC, both AP metrics, balanced accuracy, and log loss in all pairs.
Consequently the available controls do not show a predictive need for the VQC
head. The B probe is trained on an encoder already shaped by VQC gradients;
its success does **not** establish that the VQC was irrelevant to encoder
learning. B_classical changes joint training dynamics, output link, loss
landscape, and head capacity. These controls do not causally isolate a
uniquely quantum contribution, nor establish equal optimization difficulty.

## Generalization under a shared final checkpoint

Final training predictions were reconstructed from saved latent arrays and saved
heads. Training labels were recovered consistently from the existing development
validation predictions across folds; no raw cohort or locked data were loaded,
and no parameters were fitted. Final validation head outputs were also reproduced
to absolute tolerance 1e−12. These diagnostics are retrospective and do not
select checkpoints.

| Model | Training ROC-AUC | Validation ROC-AUC | Training log loss | Validation log loss | Mean validation−training loss |
|---|---|---|---|---|---|
| A | 0.8328 | 0.8297 | 0.4713 | 0.4769 | +0.0057 |
| A_classical | 0.9498 | 0.9490 | 0.2201 | 0.2286 | +0.0085 |
| B | 0.9993 | 0.9215 | 0.0488 | 0.2867 | +0.2379 |
| B_frozen_probe | 0.9758 | 0.9371 | 0.1104 | 0.2819 | +0.1715 |
| B_classical | 1.0000 | 0.9345 | 0.0022 | 0.7585 | +0.7563 |

B and B_classical have positive loss gaps in every pair: respectively
0.1405–0.3692 and 0.4027–1.2886 nats. Their training discrimination is nearly
perfect, while validation discrimination is lower in every pair. This supports
recurrent training specialization and poor probability generalization, especially
for B_classical; it is not proof of a particular causal mechanism. In contrast,
A's modest average gap coexists with poorer absolute fit and strong partition
sensitivity. Small average gaps do not prove good representations or absence of
overfitting in every fold.

The historical learning curves retain online minibatch training BCE, whereas
validation BCE uses the end-of-epoch checkpoint. Their gaps must not be treated
as same-state estimates. The new `generalization.csv` removes that particular
measurement mismatch for the final checkpoint. Earlier validation minima remain
diagnostic only and were not used for reporting or checkpoint selection.

## Latent geometry and angular embedding

All initial representations have numerical rank four. Numerical rank is the
count of centered singular values exceeding 1e−6 times the largest singular
value. This threshold is descriptive and was inherited from the pilot.

| Fold | B rank, seed 1729 | B rank, seed 1730 | B rank, seed 1731 |
|---|---|---|---|
| 0 | 4 | 3 | 3 |
| 1 | 3 | 3 | 4 |
| 2 | 4 | 3 | 3 |
| 3 | 4 | 3 | 4 |
| 4 | 3 | 4 | 2 |

Training and validation rank patterns agree in all 15 B runs. Eight solutions
have rank three and one has rank two. Nearly constant output coordinates recur,
with the affected coordinate depending on fold and seed; the fold-4/1731 solution
has two almost constant coordinates. `latent.csv` retains all four singular
values, per-coordinate SD, and initial/final, training/validation diagnostics.
The normalized spectrum and coordinate heatmap in the notebook distinguish
relative contraction from absolute scale. A nearly constant coordinate can
still provide a constant angular input; this observation alone does not identify
an implementation error or establish that four varying coordinates are needed.

Continuous effective dimension uses variance weights wᵢ=sᵢ²/Σsⱼ²: entropy rank
is exp(−Σwᵢlog wᵢ), and participation ratio is 1/Σwᵢ². These definitions are
separate from numerical rank. Mean final training entropy rank is 4.0000 (A),
1.4712 (B; range 1.0324–1.9288), and 1.1642 (B_classical; range 1.1010–1.2173).
B_classical retains numerical rank four in all pairs but is more dominated by
one variance direction. Thus anisotropy also occurs with a classical head and
a full numerical rank is not evidence of distributed useful information.
Preserving rank indiscriminately is not supported by these results. Class
separability, out-of-sample stability, and geometry must be examined together.

Angular saturation is |tanh(z/2)|>0.95, pooled over patients × coordinates.
B's final validation saturation averages 2.8842%, spanning 0–11.3772%; >0.99
saturation averages 0.7585%. A has zero validation saturation at both thresholds.
B_classical has a hypothetical >0.95 rate averaging 15.5988%, spanning
7.6347–25.1497%; its actual head does not apply this transformation. The fraction
with derivative of tanh(z/2) below 0.01 is also retained, excluding the π factor.
B's low saturation in some runs alongside generalization gaps and contraction
means saturation is not a universal explanation. No causal relation between
saturation, rank loss, and validation performance has been established.

## Gradients, stability, and computational cost

All 45 network histories contain 50 finite epochs with finite logged gradients,
losses, and angles; training raised no nonfinite-parameter or missing-gradient
errors. At epoch 50, mean head gradient norms are 0.1446 (A), 0.0479 (B), and
0.0030 (B_classical); mean encoder norms are 0.2333 (B) and 0.0241 (B_classical).
These are mean minibatch L2 norms, not full-batch gradients or unbiased estimates
of optimizer quality. The maximum logged RMS loss gradient among final-layer RZ
parameters is about 3.1e−16, consistent with structural redundancy. Those
parameters remain in place and may move through AdamW weight decay.

Finite nonzero relevant gradients and decreasing training losses exclude a
complete disconnection in these executions. They do not prove useful quantum
contribution, good conditioning, convergence to a global optimum, or mitigation
of barren plateaus. Seed variation and recurrent coordinate contraction can
reflect different solutions or implicit regularization; this experiment does
not distinguish those mechanisms. No budget extension or parameter removal was
performed.

| Model | Mean recorded seconds per seed/fold | Minimum–maximum seconds |
|---|---|---|
| A | 3.0483 | 2.8998–3.2756 |
| A_classical | 0.0012 | 0.0009–0.0020 |
| B | 3.5344 | 3.3243–3.7991 |
| B_classical | 0.2441 | 0.2280–0.2888 |
| B_frozen_probe | 0.0014 | 0.0010–0.0023 |

Mean preprocessing time is 8.2940 seconds for A and 8.1950 for B, performed
once per representation/fold and shared across seeds/controls. Repeated columns
must not be summed per model row. The historical process wall time is 38.2084 s;
the new process wall time is 152.9098 s (sum 191.1181 s across two executions,
not one measured end-to-end job). Recorded peak RSS is 4.25 and 4.12 GiB,
respectively, with byte units on macOS. Peaks are process-local and not additive.
Both record CPU and one Torch thread; host load, warm-up, caches, import time,
and process conditions are not controlled benchmark factors.

Network timings include training, per-epoch validation and angular diagnostics,
but exclude preprocessing and final prediction. Logistic regression timings
include fit and validation prediction. Probe timing excludes B encoder training,
which is required for a standalone pipeline comparison. Each VQC result records
1101 QNode calls and 41917 patient-state evaluations, including final validation;
these are analytic simulator counters, not hardware shots, hardware latency,
parameter-shift circuits, or energy measurements. Offline reconstruction adds
no calls to these saved counters. The measured B training time is about 1.16× A
and 14.5× B_classical; these ratios cannot establish hardware efficiency or
quantum advantage. Analysis/notebook verification costs are excluded.

## Interpretation and decision before C0

**Recurring observations:** B's improvement over A, its higher seed sensitivity,
training/validation separation for both encoders, numerical coordinate
contraction in B, anisotropy in both encoders, and the usefulness of classical
heads on the available representations recur beyond the pilot. **Variable
results:** gain magnitude, which coordinates contract, B/probe threshold
performance and probability-loss ordering, saturation, and seed-specific
ranking all vary. The five-fold extension strengthens descriptive recurrence;
it does not erase the exploratory origin or supply independent confirmation.

A concrete mechanism worth studying next is whether joint head/encoder training
concentrates useful class information into a small, unstable subset of latent
directions, while the fixed angular map and VQC response leave some directions
weakly useful. The combination of coordinate contraction, good frozen linear
readout, and nearly perfect training fit motivates this hypothesis. The stronger
variance concentration in B_classical is a necessary counterexample to any claim
that low effective rank is uniquely quantum or automatically harmful. The data
do not yet show that preserving rank would improve generalization.

For C0, first identify and document an existing geometric or quantum-aware
objective and the precise limitation it addresses. A subsequent separately
fixed protocol could test whether it changes coordinate stability and
out-of-sample loss, with matched classical regularization, a frozen probe,
encoder-only controls, and ablations of the angular map/head contribution.
This is a proposed testable direction, not a selected new objective, evidence of
novelty, or an implementation. C0 and C1 remain untouched pending review.

Before proposing C1, resolve: whether contraction removes label information or
merely redundant nuisance directions; whether sensitivity in contracted
directions matters on unseen development patients; whether classical objectives
produce the same effect under matched budgets; whether probability behavior
rather than representational geometry explains the main loss differences;
whether a proposed geometric intervention improves stability without forcing
uninformative variance; and whether the mechanism and objective are already
covered by prior work. The four redundant RZ parameters and the different LR
optimization budget remain explicit comparability limitations, not silent
protocol repairs. A new method requires a concrete unmet limitation, relevant
prior-art assessment, and a separately frozen validation plan. Development CV
alone cannot establish novelty, clinical utility, or quantum superiority.

## Artifacts, provenance, and reproduction

Detailed new artifacts are in `runs/phase2/cv_folds1to4_50e_paired_v1/`: resolved
configuration, environment, source snapshot, Git commit/patch, fitted
representations, initial/final checkpoints, validation predictions, curves,
metrics, latent arrays, and process resources. The original pilot directory is
unchanged. Patient-level predictions remain Git-ignored under `runs/`.

This directory contains small aggregate tables: `metrics.csv`, `summary.csv`,
`paired.csv`, `paired_fold_means.csv`, `generalization.csv`, `latent.csv`,
`angles.csv`, `curves.csv`, and `calibration.csv`. `provenance.json` records input
artifact hashes, package versions, source hashes, verification counts, and run
locations. `verification.json` records execution and preservation checks.
These summaries contain no patient IDs. The reliability counts are descriptive
aggregates with repeated patients across seeds.

The [exploratory notebook](../../../notebooks/02_development_cross_validation.ipynb)
loads these summaries and generates figures dynamically, without fitting or
accessing cohort data. It organizes objective, artifact loading, diagnostics,
interpretation, and limitations. Generated PNGs and an executed notebook copy
remain only in `runs/phase2/cv_notebook_verification/`; the versionable notebook
has cleared outputs. No report depends on ignored figure links.

Actual training command (already completed; do not rerun to regenerate figures):

```bash
.venv/bin/python -m src.phase2.run --config configs/phase2_cv_remaining.yaml --output runs/phase2/cv_folds1to4_50e_paired_v1
```

Analysis-only reproduction, using a **new** destination to prevent overwriting
human edits:

```bash
MPLCONFIGDIR=/tmp/qarl-mpl .venv/bin/python -m src.phase2.analyze_cv --output reports/phase2/cv_5fold_50e_recomputed
```

The analysis requires the trusted private run archives and split metadata; it
never fits models or loads cohort features. Pickled estimators must come only
from trusted local experiment archives. Everyday notebook execution needs only
the aggregate CSVs and existing analysis/notebook dependencies:

```bash
MPLCONFIGDIR=/tmp/qarl-mpl .venv/bin/python -m jupyter nbconvert --execute --to notebook --output-dir runs/phase2/cv_notebook_verification_new notebooks/02_development_cross_validation.ipynb
```

Python is 3.12.11, Torch 2.9.0, PennyLane 0.43.0, NumPy 2.5.3,
pandas 2.3.3, scikit-learn 1.9.1, PyArrow 21.0.0, and Matplotlib 3.10.7.
Full environment and Git provenance are saved with each run. New analysis code
is uncommitted and identified by content hashes; no commit or push was made.
The execution Git base is `32f0cdbe741bc90a308f3fa3f373b551ae9833ff`.

The initial relevant audit tests passed (36). The complete-suite attempt
`python -m pytest -q` failed at collection because `xgboost` is not installed;
no environment packages were changed. The remaining suite was run with
`--ignore=tests/data/test_xgboost_baseline.py`: **106 tests passed**. The new
notebook executed top-to-bottom and its ten figures were visually inspected.
The 92 historical files in the preservation manifest retained identical hashes.
Details are recorded in `verification.json`. This test exclusion is a historical
baseline dependency gap, not an unexecuted A/B experiment.
