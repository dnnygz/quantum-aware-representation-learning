# E0 approved v1: Head–Representation Compatibility Controls

**Status: approved closed protocol, registered before execution.**
Prepared on 2026-10-28. D0 is closed. This document authorizes no E0.3, C0, C1,
encoder retraining, calibration, architecture search, or locked-test evaluation.
The user approved this design with the three additions below. The draft and methodological review remain unchanged.


## Approved additions registered before execution

- **E0.1 primary contrasts:** final-checkpoint training log loss and validation log loss of each refitted VQC minus the corresponding historical B head. Their separation describes fit and generalization. Report every initialization and the prespecified hierarchical mean. All other planned metrics remain secondary.
- **E0.2 primary contrasts:** paired validation ROC-AUC and log loss of LR-angle minus historical LR-z, within each fold and encoder seed. Retain training metrics, AP ER−, balanced accuracy, Brier, and cost.
- **Prespecified diagonal comparison:** separately present all 15 cases with h=s against the corresponding historical B head. These are a subset of the 45 fits, not additional fits. The historical encoder coevolved with its head, so this is not a pure initialization contrast.
- **Closed phase:** no added epochs, optimizers, angular maps, architectures, initializations, or hyperparameter searches after observing results. Any extension needs an independent protocol; E0.3/C0/C1 remain excluded.

## Question and estimand

Can alternative fitted heads use the *same final B representation* differently,
and do three prespecified VQC initializations produce materially different
solutions under a fixed budget? Does linear classification change when the
identical embeddings are presented before versus after the fixed angular map?

The estimand is a controlled, development-only comparison of head-fitting
procedures conditional on the already learned B encoders. It is not the effect
of using a quantum head to learn an encoder, a comparison of optimal head
capacities, or an independent confirmation of generalization. The original
encoders were learned jointly with their VQCs and cannot be treated as neutral
or randomly assigned representations.

## Fixed inputs and isolation

Use all five canonical development folds and the original encoder seeds
s in {1729,1730,1731}, giving 15 frozen representations. Each fold contains
668 training patients (516 ER+, 152 ER−) and 167 validation patients
(129 ER+, 38 ER−). Across development there are 835 patients; the 209 locked-test
patients remain untouched. Split metadata may be read solely to enforce isolation.

Read the final B latent arrays and checkpoint provenance from
`runs/phase2/pilot_fold0_50e_paired_v1/` for fold 0 and
`runs/phase2/cv_folds1to4_50e_paired_v1/` for folds 1–4. The saved arrays are
sufficient; neither raw RNA-seq nor locked features/labels need to be loaded.
Validate hashes, canonical train/validation row membership and ordering, finite
values, and agreement of reconstructed original B/probe predictions with their
saved validation probabilities. Reuse the development-only label reconstruction
validated in D0. Do not refit preprocessing, PCA, score scaling, or encoders.

For every representation, define exactly two feature arrays:

- z: the saved final four-dimensional B embedding, without additional scaling.
- a=phi(z)=pi*tanh(z/2): the fixed elementwise angular map, applied exactly once.

No centering, whitening, variance floor, coordinate deletion, angular rescaling,
or validation-derived transformation is added. The map has no fitted parameters.
Original B and B_frozen_probe remain immutable historical references, including
weights, predictions, configurations, metrics, and timing records.

## Minimal design and number of fits

| Component | Input and fitted head | Runs per frozen encoder | New fits |
|---|---|---:|---:|
| Historical B | Original jointly learned VQC on phi(z), now evaluated frozen | One retained reference | 0 |
| Historical B_frozen_probe / LR-z | Existing fitted logistic regression on exactly z | One retained reference | 0 |
| E0.1 VQC refit | Fresh copy of the original VQC on phi(z) | Three fixed head initializations | 45 |
| E0.2 LR-angle | New logistic regression on exactly phi(z) | One deterministic fit | 15 |

**Total: 60 new fits, plus the 30 historical reference results.** The LR-z head
has already been fitted on exactly these training embeddings; duplicating the
fit is unnecessary. A numerical reconstruction check is not a new training run.
If its source artifacts fail verification, stop that comparison and report the
missing or inconsistent input rather than silently replacing the reference.

The three encoder seeds and three head initialization seeds are distinct design
axes, even though their numerical values are the same. Use the full Cartesian
product (fold f, encoder seed s, head seed h) with h in {1729,1730,1731}; do not
restrict E0.1 to h=s. The diagonal initializations were present in the original
joint-learning histories, so report diagonal and off-diagonal results without
selecting either group. Three head fits on one encoder do not create three
independent validation cohorts.

## E0.1: new VQC heads on frozen B encoders

### Architecture and initialization

Preserve the original four-qubit circuit exactly: one RY input encoding, two
layers of RY/RZ rotations followed by CNOT 0→1→2→3, measurement Pauli-Z on qubit 3,
and p(ER+)=clip((1+expectation)/2,1e−7,1−1e−7). Use default.qubit, analytic
expectations, CPU float64, and backpropagation. Retain all 16 nominal parameters,
including the four structurally redundant final-layer RZ parameters. Removing
these parameters would introduce another architectural intervention.

For each h, initialize all 16 angles with exactly 0.1 times independent standard
normal draws from a fresh CPU Torch generator seeded by h, in the original
(2,4,2) tensor order and float64 dtype. Verify the corresponding initial head
weights against saved original initial checkpoints before fitting. All folds
and encoder seeds receive the same three head initialization tensors. Do not
start from original final head weights, perturb a favorable solution, or add
unregistered restarts.

Use a separate shuffle generator seeded by encoder seed s. Reset it identically
for all three h fits on the same frozen encoder, so each epoch has identical
minibatch membership and order across those fits. Initialization and shuffle
streams are independent. Consequently the head-seed contrast within an encoder
changes initialization, rather than simultaneously changing the minibatch stream.
This E0-specific RNG policy is explicit; it does not claim to replay the original
joint-training trajectory.

### Fixed optimization budget

Use unweighted BCE, AdamW, lr=0.01, weight_decay=0.001, batch size 32, and exactly
50 epochs: 21 batches per epoch and 1050 optimizer steps per head. Only head
parameters enter the optimizer. Encode no backward path to B, and assert that
encoder and input-array hashes remain unchanged. Reuse the established
probability link; add no trainable output calibration or threshold.

Evaluate the fixed final checkpoint, never the validation-best epoch or seed.
Retain epoch-wise training loss, validation loss, head/parameter gradient
summaries, finite-value checks, and final training/validation metrics. Validation
logging is observational and cannot change the run. Final training metrics must
use the same final checkpoint as validation; distinguish them from online
minibatch training losses.

No extended budget is included in this minimal version. A 100/200-epoch arm,
learning-rate change, warm start, or optimizer change would require a separate
amendment approved before observing its validation results.

## E0.2: classical heads before and after the angular map

Use the existing B_frozen_probe as LR-z and fit LR-angle on the paired training
angular arrays. Preserve its logistic-regression specification: C=1,
max_iter=2000, solver=lbfgs, tol=1e−4, fit_intercept=True, no class weighting,
no warm start, and the same effective L2 regularization/default l1_ratio=0.
Copy and record the complete saved estimator settings with the pinned
scikit-learn version, rather than depending on changing library defaults.
Use random_state=s for provenance. L-BFGS is deterministic here; do not create
three identical head-seed repetitions and count them as additional evidence.

No hyperparameter grid, C selection, feature standardization, class balancing,
calibration, or new decision threshold is introduced. Record actual iterations,
convergence warnings, fitting time, and prediction time. A convergence warning
is a reported limitation; it does not authorize increasing max_iter after seeing
validation results. Numerical failures remain explicit outcomes.

The same C before/after a nonlinear feature map does **not** mean identical
effective regularization or the same hypothesis class in z. E0.2 therefore tests
the specified map-plus-linear-head fitting procedure, not information destruction
by the map. Classical invertibility of the finite-valued tanh map also does not
imply invariance of linear separability, regularization, numerical conditioning,
or the quantum encoding's periodic response. The LR and VQC optimizers/budgets
are different, so their contrast cannot isolate a uniquely quantum physical
contribution or establish equal optimization effort.

## Metrics, contrasts, and summaries

For each completed head, retain ROC-AUC, AP ER−, balanced accuracy, log loss
(nats), Brier score, and computational cost. ER+=1, ER−=0. Use p(ER+) for ROC-AUC,
1−p with ER− as the positive target for AP ER−, and p>=0.5 for decisions. Brier is
mean((p−y)^2). Lower log loss and Brier are better; the discrimination and
classification scores are higher-is-better. Neither proper scoring rule isolates
calibration from discrimination. No post-hoc calibrator is fitted.

Prespecified contrasts, always inside the same fold/encoder seed:

1. Each new VQC head versus historical B.
2. Each new VQC head versus LR-z and LR-angle.
3. LR-angle minus LR-z.
4. Dispersion across the three new VQC initializations with the encoder fixed.

For new VQCs, first show all three head-seed outcomes per encoder. Then average
head seeds within each encoder seed, encoder seeds within fold, and fold means
across the five folds. Also retain within-encoder head-seed spread and
within-fold encoder-seed variation; neither is an independent-cohort uncertainty
estimate. Classical outcomes appear once per encoder, without head-seed
replication. The primary summary is the prespecified mean, not the best head.

Show effect sizes, fold/seed matrices, minima/maxima, all exceptions, and final
training/validation gaps. Do not treat the 45 fits as independent observations,
compute naive sqrt(45) standard errors, or use recurrence thresholds to declare
a discovery. No confirmatory p-values or locked-test claims are planned.

## Falsifiable questions and interpretation criteria

| Question | Descriptively compatible result | Result that weakens the account | Limit |
|---|---|---|---|
| Does the original final B head leave fit attainable by this procedure on the same z? | New heads have lower final training BCE, with their validation effects fully reported. | All fixed restarts have comparable/worse training fit. | Improvement demonstrates an attainable alternative, not a global optimum; a negative result cannot prove optimality. |
| Is head initialization consequential on the fixed representation? | The three fixed starts yield appreciable differences in losses, scores, or failures under identical batches. | Outcomes are close for the three tested starts. | No claim about every possible initialization or a barren plateau follows. |
| Does the angular map change classical readability under the fixed LR procedure? | LR-angle and LR-z differ in paired discrimination, decisions, or probability losses. | Differences are small across the retained distributions. | The intervention includes the induced change in feature geometry and L2 penalty; small differences do not prove equivalence. |
| Do head effects reflect fit versus generalization? | Training improvements coexist with heterogeneous/worse validation loss. | Paired training and validation changes agree across the measured cases. | Neither outcome identifies the encoder's learning mechanism. |

No retrospective numerical cutoff defines 'appreciable' or 'small': report the
actual scales and distributions for review. The questions are exploratory and
must not become claims that stronger sensitivity, higher rank, or a particular
head is inherently useful.

### What is causally controlled, and what is not

Once z, labels, architecture, budget, and minibatch order are fixed, changing
h is a reproducible intervention on head initialization. Its consequences for
that deterministic fitted pipeline can be directly compared. Replacing the
head-fitting procedure while holding z fixed is likewise a controlled algorithmic
intervention. After fitting, frozen predictions compare the resulting heads on
the same patients and embeddings.

The new-head versus historical-B contrast also changes the training trajectory:
the historical head co-evolved with an encoder, while a refitted head sees final
z from step one. It therefore cannot isolate an effect of initialization alone,
'freezing the encoder', quantum expressivity, or additional training time.
LR-angle versus LR-z controls the input source and LR specification, but changes
functional geometry and effective regularization together. These are causal
contrasts of the **specified procedures conditional on the realized embedding**,
not causal effects of quantum mechanics or population-level clinical effects.

None of E0.1/E0.2 establishes how B learned z, whether a classical training
objective would learn the same representation, whether VQC gradients were
necessary, or whether low latent rank caused generalization problems. A successful
new VQC would not prove quantum advantage; a successful linear head would not
prove that the original VQC was irrelevant to representation learning. Missing
counterfactual encoder histories require a separate, explicitly approved stage.
No E0.3 intervention is defined or executed by this protocol.

## Computational cost and failure discipline

Record source-loading time, angular-map time, head-fitting time, per-epoch
validation/diagnostic time, final train/validation prediction time, total process
wall time, peak RSS with platform units, CPU/device, thread counts, and dependency
versions. Report optimizer steps for VQCs and iteration counts for LR. For VQCs,
count QNode invocations and sample-state evaluations separately from classical
operations; these are analytic simulator counters, not shots or hardware cost.

Record the execution order before fitting and retain all runs; do not rerun
favorable timings. Compare new head costs within the same current environment.
Historical B cost includes encoder training under different execution conditions,
whereas probe cost excludes learning the encoder: present historical costs
separately. All conditional head timings exclude the sunk cost of obtaining B
unless an explicitly labeled whole-pipeline total is reported. Fifty epochs and
2000 L-BFGS iterations are not matched wall-clock or optimization budgets.

A nonfinite loss/gradient, corrupted source, or failed isolation check stops the
affected computation and is reported. Do not silently alter a learning rate,
initialization, batch order, C, budget, or failed-run denominator. Technical
re-execution must preserve settings and retain the failure record; numerical
optimization failures cannot be replaced by favorable restarts.

## Planned implementation and review gate

If approved, create a new versioned E0 configuration and new run directory;
leave original source experiments, B/probe references, and D0 reports untouched.
Detailed predictions/checkpoints/curves and provenance belong in Git-ignored
`runs/`; small non-patient-level summaries and a self-contained English report
belong in `reports/`; a read-only notebook should load completed outputs only.
Capture input/source hashes, resolved settings, actual commands, environment,
and all run statuses. Refuse overwriting existing run/report directories.

Tests should cover split/ID isolation, unchanged encoder/embedding hashes,
head-only optimizer membership, exactly-once angular mapping, reconstruction of
historical probabilities, Cartesian initialization coverage, fixed paired batch
orders, fixed epoch count, metric recomputation, and no selection or head-seed
pseudoreplication. Preserve fixed-final-epoch evaluation without early stopping. Implementation and execution may begin only after this approved version and its configuration have been saved and hashed.

Related methodological review: [Quantum Metric Learning and Neural Quantum
Embedding](e0_methodology_review_v1.md). The papers motivate distinctions between
embedding learning and readout fitting; they are not implementations or validation
of this E0 protocol.
