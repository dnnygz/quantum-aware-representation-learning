# E0: Head–Representation Compatibility Controls

## Status and principal findings

**Completed, closed E0.1/E0.2 development experiment.** Forty-five fresh VQC heads
and 15 LR-angle heads were fitted on the 15 existing final B representations.
All planned fits completed, without warnings, failed fits, additional restarts,
extra epochs, optimizer changes, feature searches, or architecture changes.
Original B and B_frozen_probe remain unchanged. No locked-test features or labels
were accessed. E0.3, C0, and C1 were not implemented.

The primary E0.1 contrast shows small mean changes from historical B: training
log loss −0.000220 nats and validation log loss −0.000382 nats. Validation changes
have both signs and span −0.01144 to +0.01651 across the 45 heads. Fresh heads
largely recover the original head's fit on fixed embeddings under this budget;
this is not an equivalence test or proof of global optimality.

For E0.2, LR-angle minus LR-z has mean validation ROC-AUC −0.000068 and log loss
+0.002029 nats. The angular features improve logistic-regression training loss
in every encoder case but do not deliver a consistent validation improvement.
These results do not support a large, general remedy based solely on head
restarting or replacing z by angular features in the specified linear readout.
They do not identify how the encoder was learned or establish quantum advantage.

## Protocol registration, questions, and data isolation

The [approved E0-v1 protocol](../e0_protocol_v1.md) was saved before fitting,
including all three user amendments. E0.1 primary endpoints are final training
and validation log loss of the refit minus corresponding historical B. E0.2
primary endpoints are paired validation ROC-AUC and log loss of LR-angle minus
historical LR-z. All other planned discrimination, classification, probability,
and cost metrics were retained. The h=s diagonal subset was prespecified and
is explicitly reported below. The phase remained closed after observing results.

Five existing development folds and encoder seeds s in {1729,1730,1731} give
15 fixed representations. Each fold has 668 training patients (516 ER+, 152 ER−)
and 167 validation patients (129 ER+, 38 ER−). Development totals 835 patients;
209 locked-test patients are isolated and only their split metadata is consulted
for isolation checks. Development validation sets are disjoint, while their
training sets overlap. All three encoder seeds reuse the same patients per fold.

The original RNA-seq preprocessing, PCA, score scaling, and encoder states were
not fitted or executed again. E0 reads saved final B latent arrays and canonical
split metadata, validates row membership/order and provenance hashes, and
reconstructs development labels from the existing validation prediction files.
It never opens expression or label parquet files. Historical probabilities are
reproduced within absolute tolerance 1e−12 before fitting. Embedding arrays are
read-only; their content hashes and source-checkpoint hashes are unchanged.

## Models, seeds, and fixed fitting budgets

The original B pipeline used training-only CPM/TMM/log2-CPM filtering/scaling,
PCA32 with representation seed 1729 and score scaling, followed by a 32→8→4
encoder with a Tanh hidden activation. Its encoder and VQC coevolved under BCE.
The original B_frozen_probe is logistic regression fitted on the final z from
that B encoder. The historical final z matrices, not intermediate checkpoints
or a newly selected representation, are the E0 inputs.

| Condition | Fitting in E0 | Input | Number of final heads |
|---|---|---|---:|
| Historical B | None | Saved z → angular map → historical VQC | 15 |
| Historical LR-z | None; the unchanged B_frozen_probe | Saved z | 15 |
| VQC refit, E0.1 | Three fresh starts per fixed encoder | Same z → angular map → new VQC | 45 |
| LR-angle, E0.2 | One new deterministic LR per fixed encoder | pi*tanh(z/2) | 15 |

All new VQCs retain four qubits, one RY encoding of a=pi*tanh(z/2), two RY/RZ
layers with the CNOT chain 0→1→2→3 after each layer, and Pauli-Z measurement on
qubit 3. The probability is clip((1+expectation)/2,1e−7,1−1e−7), with unweighted
BCE, ideal analytic default.qubit simulation, CPU float64, and backpropagation.
All 16 nominal head parameters are retained, including four structurally
redundant final RZ parameters. There is no additional trainable output link.

Head seeds h in {1729,1730,1731} are fully crossed with every encoder seed and
fold. Each initial tensor is exactly 0.1*Normal(0,1) in shape (2,4,2), generated
by a fresh CPU float64 Torch generator seeded by h. Those tensors were checked
against saved initial historical VQC weights. A separate generator seeded by s
controls shuffling and is reset for each h. Thus all three heads on a given
encoder use exactly the same batches; changing h changes initialization alone
within that conditional comparison. Head and encoder seed numbers are separate
axes, not 45 independent cohorts. The diagonal h=s was not selected by outcome.

Every new VQC uses AdamW lr=0.01, weight_decay=0.001, batch size 32, exactly
50 epochs and 1050 optimizer steps. Only the VQC weight tensor enters the
optimizer; the computation contains no trainable encoder. Validation logging
cannot alter optimization. The final epoch is the only evaluation checkpoint.
Final training and validation metrics use the same final state; online minibatch
training losses in the curves are separately identified.

LR-angle uses exactly the same fitted-estimator specification as its associated
historical LR-z: C=1, max_iter=2000, L-BFGS, tol=1e−4, intercept enabled,
no class weighting, no warm start, effective L2 regularization/default l1_ratio=0,
and random_state=s. Complete resolved settings are recorded per encoder in the
run configuration, with scikit-learn pinned to the historical version. No extra
feature standardization, C grid, calibrator, threshold adjustment, or repeated
identical LR fits was added. LR-angle applies the angular map exactly once;
VQC inputs are z and their forward pass applies it exactly once as well.

## Metrics and dependent-data summaries

ER+=1 and ER−=0. ROC-AUC uses p(ER+); AP ER− uses 1−p and ER− as the target.
Balanced accuracy uses the fixed p>=0.5 decision rule. Log loss is in nats;
Brier is mean((p−y)^2). Lower log loss and Brier are better, whereas the other
three requested scores are higher-is-better. AP ER+ and accuracy are retained
as supplementary shared-evaluator outputs. Neither proper probability score
isolates calibration from discrimination.

For VQC refits, summaries first average the three h outcomes within each
(fold,s), then average encoder seeds within each fold and the five fold means.
Historical/classical heads appear once per encoder. Tables report the sample
SD of the five fold means, alongside within-fold encoder variation and
within-encoder head variation in the CSVs. No head selection, independent-45-fit
test, naive standard error, recurrence discovery rule, confirmatory p-value,
or locked-test claim is used. Counts of improvements below are descriptive,
not independent Bernoulli trials or significance evidence.

## E0.1 primary results: adjustment and generalization of the refitted head

| Prespecified subset | Mean Δ training log loss | Mean Δ validation log loss | Between-fold SD, training / validation |
|---|---:|---:|---|
| All 45 initializations | −0.000220 | −0.000382 | 0.000170 / 0.005023 |
| Diagonal h=s, 15 fits | −0.000223 | −0.000437 | 0.000169 / 0.004846 |
| Off-diagonal, 30 fits | −0.000218 | −0.000355 | 0.000170 / 0.005111 |

Deltas are refit minus historical B on exactly the same final z. Training loss
is lower in 42/45 refits and higher in three; the entire training-effect range
is −0.000814 to +0.000300 nats. Validation loss is lower in 21/45 and higher in
24/45, ranging from −0.011440 to +0.016511. Average validation improvements are
therefore small compared with their partition- and encoder-dependent effects.

| Fold | Mean Δ training loss, all h | Mean Δ validation loss, all h |
|---|---:|---:|
| 0 | −0.000230 | −0.000549 |
| 1 | −0.000460 | +0.006131 |
| 2 | −0.000244 | +0.001950 |
| 3 | −0.000179 | −0.001950 |
| 4 | +0.000015 | −0.007493 |

The mean within-encoder SD across the three head initializations is approximately
0.000006 nats for training loss and 0.000171 for validation loss; the largest
within-encoder validation SD is 0.000888. By comparison, mean within-fold SD
between encoder seeds, after averaging heads, is 0.033873 for validation loss.
This descriptive separation shows that these tested head starts contribute
little of the observed variability conditional on the saved encoders. It does
not prove that encoder initialization alone caused the original variation:
original encoders and heads were jointly trained.

### Prespecified diagonal comparison: all 15 h=s cases

These rows were part of the original 45-fit plan and add no training. H denotes
historical B; R denotes the new VQC refit. All values are final-checkpoint log
loss in nats.

| Fold | s=h | H train | R train | Δ train | H validation | R validation | Δ validation |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1729 | 0.044497 | 0.043976 | −0.000522 | 0.256243 | 0.250169 | −0.006074 |
| 0 | 1730 | 0.068196 | 0.068117 | −0.000078 | 0.225815 | 0.227388 | +0.001573 |
| 0 | 1731 | 0.027203 | 0.027109 | −0.000094 | 0.195264 | 0.198071 | +0.002808 |
| 1 | 1729 | 0.025102 | 0.024681 | −0.000421 | 0.266714 | 0.281531 | +0.014817 |
| 1 | 1730 | 0.067725 | 0.066912 | −0.000814 | 0.271262 | 0.273532 | +0.002270 |
| 1 | 1731 | 0.059098 | 0.058936 | −0.000161 | 0.199632 | 0.199938 | +0.000306 |
| 2 | 1729 | 0.036133 | 0.035704 | −0.000429 | 0.272998 | 0.273139 | +0.000141 |
| 2 | 1730 | 0.066758 | 0.066693 | −0.000065 | 0.252852 | 0.254824 | +0.001972 |
| 2 | 1731 | 0.033668 | 0.033422 | −0.000246 | 0.264479 | 0.268005 | +0.003526 |
| 3 | 1729 | 0.041028 | 0.041025 | −0.000003 | 0.410187 | 0.407671 | −0.002516 |
| 3 | 1730 | 0.051203 | 0.051132 | −0.000071 | 0.386489 | 0.383293 | −0.003197 |
| 3 | 1731 | 0.068997 | 0.068540 | −0.000458 | 0.348023 | 0.347784 | −0.000238 |
| 4 | 1729 | 0.022720 | 0.022953 | +0.000233 | 0.384704 | 0.374448 | −0.010257 |
| 4 | 1730 | 0.061895 | 0.061755 | −0.000141 | 0.269306 | 0.260589 | −0.008717 |
| 4 | 1731 | 0.057816 | 0.057741 | −0.000075 | 0.296109 | 0.293139 | −0.002970 |

The diagonal has 14 lower training losses and one higher; validation improves
in seven cases and worsens in eight. The historical encoder coevolved with its
head, whereas the refit sees the final encoder output from the first update.
The E0 shuffle stream is also explicitly separate from initialization. Matching
h=s is therefore **not a pure initialization contrast against historical B**
and does not replay its historical optimization trajectory.

## E0.2 primary results: LR-angle versus historical LR-z

| Primary endpoint | Mean paired effect | Between-fold SD | Minimum–maximum across 15 encoders |
|---|---:|---:|---|
| Validation ROC-AUC | −0.000068 | 0.000838 | −0.003264 to +0.003468 |
| Validation log loss, nats | +0.002029 | 0.006747 | −0.017244 to +0.014072 |

LR-angle has higher validation ROC-AUC in five cases, the same in three, and
lower in seven. Validation loss is lower in five cases and higher in ten.
These are mixed effects with small mean AUC change, not evidence of statistical
equivalence. The full paired matrix is in `e02_paired.csv` and the notebook.

| Fold | Mean Δ validation ROC-AUC | Mean Δ validation log loss |
|---|---:|---:|
| 0 | −0.000476 | −0.007245 |
| 1 | +0.000748 | +0.006687 |
| 2 | −0.001020 | +0.008150 |
| 3 | +0.000884 | +0.005508 |
| 4 | −0.000476 | −0.002955 |

Training log loss improves for LR-angle in all 15 cases: mean change −0.018191,
with range −0.031234 to −0.007690 nats. Its mean training loss is 0.092192 versus
0.110383 for LR-z, while mean validation loss is 0.283954 versus 0.281925.
The angular feature map permits a different linear fit in transformed space;
same C does not imply equal effective regularization in the original variables.
Consequently improved training fit plus mixed validation effects does not prove
information gain/loss, a uniquely angular bottleneck, or a calibration remedy.

## All requested validation metrics and generalization gaps

Means ± sample SD of five fold means; VQC head seeds are averaged within encoder
first. These are descriptive SDs, not confidence intervals.

| Model | ROC-AUC | AP ER− | Balanced accuracy | Log loss, nats | Brier |
|---|---|---|---|---|---|
| Historical B | 0.921474 ± 0.029469 | 0.814820 ± 0.063357 | 0.881538 ± 0.030603 | 0.286672 ± 0.062889 | 0.070913 ± 0.015086 |
| VQC refit | 0.921837 ± 0.029166 | 0.817110 ± 0.062676 | 0.882415 ± 0.028928 | 0.286290 ± 0.060367 | 0.070866 ± 0.014960 |
| Historical LR-z | 0.937141 ± 0.018622 | 0.833676 ± 0.032640 | 0.874888 ± 0.038905 | 0.281925 ± 0.051563 | 0.068230 ± 0.015624 |
| LR-angle | 0.937073 ± 0.017970 | 0.834573 ± 0.028332 | 0.877982 ± 0.035429 | 0.283954 ± 0.051222 | 0.068917 ± 0.015623 |

Both classical readouts have higher mean discrimination and lower mean Brier
than the VQC refits, while VQC mean balanced accuracy is higher. VQC refit minus
LR-z has mean validation AUC −0.015304 and log loss +0.004365; versus LR-angle,
these are −0.015236 and +0.002336. Neither contrast isolates quantum physics or
matched optimization effort. Ranking, thresholded classification, and probability
assignment answer different questions and should not be collapsed into one verdict.

| Model | Mean final training loss | Mean final validation loss | Mean validation−training gap |
|---|---:|---:|---:|
| Historical B | 0.048803 | 0.286672 | 0.237869 |
| VQC refit | 0.048583 | 0.286290 | 0.237706 |
| Historical LR-z | 0.110383 | 0.281925 | 0.171541 |
| LR-angle | 0.092192 | 0.283954 | 0.191762 |

The VQC gap remains nearly unchanged by refitting its head. LR-angle has a larger
average gap than LR-z despite fitting training better. These are shared-final-
checkpoint comparisons; the online minibatch curves are not substituted for
final training loss. Refit histories show finite losses and gradients throughout
all 2250 logged epochs; no result was selected at an earlier validation minimum.

## Computational cost and limitations

| Newly fitted head | Mean fitting/update time | Mean epoch-validation time | Mean final train+validation prediction time |
|---|---:|---:|---:|
| VQC refit | 2.888859 s | 0.092836 s | 0.004822 s |
| LR-angle | 0.001480 s | Not iterative epoch monitoring | 0.000141 s |

VQC update times span 2.694497–3.328802 s; LR-angle fitting spans
0.001014–0.004319 s. Each VQC performs 1050 optimizer steps, 1100 QNode calls
through training and per-epoch validation, and 1102 after final prediction.
Total per-head sample-state evaluations are 42585. These are analytic simulator
counters, not quantum hardware shots, hardware latency, or parameter-shift
circuits. LR-angle required 8–20 L-BFGS iterations, mean 14.93, with no convergence
or other warnings. The cost table's iteration value −1 for VQCs means not
applicable; optimizer steps provide their actual budget.

The measured execution wall time is 135.64 s, including source loading (0.1282 s),
fit/evaluation, saving, and provenance. Process peak RSS is 483852288 bytes
(about 0.451 GiB on macOS); Torch uses one CPU thread. Analysis, tests, and
notebook rendering are excluded. Per-fit cost records also retain elapsed
fit-and-save time and angular-map reference time. Mapping is shared within an
encoder; its repeated reference-time column must not be summed for every head.
VQC forward calls intrinsically include their fixed map computation.

These are conditional head costs on already learned embeddings. They exclude
the sunk cost of training B and cannot be compared as complete pipelines with
historical B/probe timings recorded under different host conditions. The VQC
and LR optimizers/budgets are not matched. Host load, warm-up, logging, simulator
implementation, and file I/O limit benchmarking claims. No claim of hardware
speedup, quantum efficiency, or advantage is supported.

## Controlled conclusions and implications for C0

Within one frozen encoder and fixed batch schedule, the head-seed intervention
changes initialization while holding the other E0.1 ingredients fixed. Its
conditional effects can be directly measured. The tested starts converge to
similar losses; this weakens an explanation based on a large readily recoverable
head-initialization problem under the approved 50-epoch procedure. It does not
prove global optimality, rule out untested starts/budgets, or justify extending
this closed phase.

Comparing a refit with historical B additionally changes the trajectory from
coevolving features to final fixed features. Comparing LR-angle with LR-z changes
the feature geometry and the effective penalty/hypothesis class together.
These are controlled algorithmic contrasts conditional on realized embeddings,
not causal estimates of quantum contribution to representation learning or
population-level clinical effects. Classical readability does not show that
VQC gradients were irrelevant to learning B, and a small restart effect does
not prove the encoder caused every performance limitation.

For a future C0, these controls provide a fixed reference: changing the readout
alone under this budget did not substantially remove the generalization gap,
and linear readability remained broadly similar before/after the angular map.
A separately specified embedding-learning intervention should therefore test
its claimed mechanism during representation learning, retaining classical
objective/readout controls and E0 baselines rather than assuming a head restart
or angular remapping resolves it. This is motivation for a test, not evidence
that a geometric intervention will help or a choice of definitive regularizer.

The [approved methodological review](../e0_methodology_review_v1.md) distinguishes
Lloyd/Hur embedding objectives from our single-example BCE. Its architecture-
specific deduction also matters: a common data-independent unitary head cannot
change state overlaps after a fixed encoding, and the existing product RY
fidelity kernel has a direct classical expression. A C0 adaptation must specify
which embedding component learns and include corresponding classical controls;
changing only this unitary head cannot train class-ensemble fidelity separation.
No new metric-learning code, C0, E0.3, or C1 was added in E0.

Unresolved questions remain about how joint encoder/head training produced the
representations, what label-relevant information would change under an embedding
objective, whether improved training geometry would generalize, and what prior
art already covers any proposed mechanism. These require an independent protocol,
not retrospective additions to E0.

## Artifacts, provenance, and reproduction

The only authoritative new experiment is `runs/phase2/e0_v1/`. It contains the
approved protocol snapshot and SHA-256, fully resolved configuration, a fixed
60-fit plan, statuses, source snapshots/hashes, original input hashes, initial
and final heads, LR-angle estimators, train/validation predictions, 50-epoch
curves, timing/counter records, and environment/provenance. Patient-level outputs
remain Git-ignored under `runs/`. All 60 statuses are complete, with no warnings
or failures. Historical training runs, D0, original references, and draft/review
documents are preserved.

This report directory contains non-patient-level `metrics.csv`, `head_results.csv`,
`e01_paired.csv`, `e01_diagonal.csv`, `e02_paired.csv`, `summary.csv`,
`primary_summary.csv`, `fold_primary.csv`, `initialization_spread.csv`,
`curves.csv`, and `costs.csv`. The diagonal CSV includes every requested h=s
comparison. `analysis_provenance.json` records independent recomputation of all
180 split-level metric sets, reconstruction of all 120 newly fitted split
predictions, initial-weight checks, fixed epoch budgets, and all 15 paired
shuffle groups. `validation_record.json` records final tests, preservation,
notebook checks, and delivery-source hashes.

The [exploratory notebook](../../../notebooks/04_e0_head_compatibility.ipynb)
reads only these aggregate CSVs, without fitting or accessing checkpoints/cohort
data. It presents objective, artifact loading, both primary contrasts, the
prespecified diagonal, initialization spread, all requested metrics,
shared-checkpoint gaps, fixed-budget curves, cost, interpretation, and limitations.
Plots are generated dynamically by shared functions in `src/phase2/e0_views.py`.
Executed notebook/images remain local under `runs/phase2/e0_notebook_verification_v1/`;
the versionable notebook has cleared outputs. This report does not depend on
ignored PNG/SVG links.

Actual execution command, already completed; do not rerun to regenerate figures:

```bash
MPLCONFIGDIR=/tmp/qarl-mpl .venv/bin/python -m src.phase2.run_e0 --config configs/e0_v1.yaml --output runs/phase2/e0_v1
```

Analysis-only recomputation uses the saved experiment and a new destination:

```bash
MPLCONFIGDIR=/tmp/qarl-mpl .venv/bin/python -m src.phase2.analyze_e0 --run runs/phase2/e0_v1 --output reports/phase2/e0_analysis_new
```

The runner rejects existing destinations and any configuration change to this
closed phase. Analysis never fits models. Trusted private archives are needed
for checkpoint/prediction reconciliation; everyday notebook use requires only
aggregate summaries and existing analysis/notebook dependencies:

```bash
MPLCONFIGDIR=/tmp/qarl-mpl .venv/bin/python -m jupyter nbconvert --execute --to notebook --output-dir runs/phase2/e0_notebook_new notebooks/04_e0_head_compatibility.ipynb
```

The source environment matches the original experiments: Python 3.12.11,
Torch 2.9.0, PennyLane 0.43.0, NumPy 2.5.3, pandas 2.3.3, scikit-learn 1.9.1,
and PyArrow 21.0.0. The execution Git base is
`e32a3a791ac2c4f5327305a804648e9c9be464cd`; new implementation files remain
uncommitted and are identified by content hashes in the source snapshot and
verification record. No commit or push was performed.

## Final verification

All 180 split-level metric sets were independently recomputed from saved
probabilities or reconstructed historical references. The 120 newly fitted
train/validation probability vectors were also reproduced from their saved
checkpoints/estimators. All 45 initial tensors, 50-epoch histories, 1050-step
budgets, and 15 paired shuffle groups passed verification. The 15 diagonal
comparisons are a prespecified subset, not additional training.

The available suite passed **127 tests**, including eleven E0 cases covering
closed-configuration rejection, initialization and exactly-once mapping,
validation-label isolation, frozen inputs, shared batch streams, the locked-source
gate, paired/hierarchical summaries, analysis without fitting, and notebook use
with checkpoint/cohort access and fitting forbidden. The historical XGBoost test
module was excluded because `xgboost` is absent; no package versions were changed.
The notebook executed top-to-bottom and all eight figures were visually
inspected. All 909 files in the pre-E0 historical preservation manifest retained
their hashes. See [the validation record](validation_record.json) for commands,
source/summary hashes, and execution checks.
