# D0: frozen representation–VQC compatibility diagnostics

## Status and main findings

**Completed exploratory D0-A/D0-B, using existing development checkpoints only.**
The five development folds and seeds 1729, 1730, and 1731 were retained. No network,
probe, calibrator, or preprocessing pipeline was retrained. No locked-test
features or labels were loaded. D0 produced 210 model/checkpoint/split diagnostic
groups and 8160 final-head intervention cells. C0 and C1 remain unimplemented.

The strongest distinction is between three quantities: **representation
variation**, **functional sensitivity**, and **predictive effects**. A large
response does not identify useful information; a small response does not
identify wasted information. A frozen intervention's metric change is not a
causal effect on the learning process.

The angular chain rule was reproduced to maximum absolute error 1.09e−15. In
ten nearly constant B coordinates across nine fold/seed solutions, the map's
mean derivative remains 99.42–100.00% of its maximum while the circuit's local
angular response is very small. Thus angular saturation alone is a weak
explanation for those contracted coordinates. Mean replacement is essentially
inert there; zero replacement changes probabilities but reduces validation log
loss in all ten coordinate cases. This supports dependence on constant input
levels without demonstrating that their learned values are beneficial.

B and B_frozen_probe respond differently to the same embeddings. Under
coordinate scrambling, their mean validation ROC-AUC changes are −0.0648 and
−0.0608, respectively; their mean log-loss changes are +0.1183 and +0.1437 nats.
The paired differences vary across folds and seeds. Neither these effects nor
gradient magnitudes establish a useful uniquely quantum contribution.

## Research question and approved scope

D0 asks where variation is present, where a frozen head responds, and how its
predictions change when inputs are perturbed. It does not ask whether forcing
rank, altering the angular map, or replacing the head during training would
improve generalization. Those questions require new experiments.

The [D0-v1 protocol](../d0_protocol_v1.md) incorporates the four approved changes:
separate variation/response/prediction; verify the chain rule and separate its
factors; remove the proposed recurrence discovery rule; and compare B/probe in
their shared coordinates while treating B_classical as an independently learned
representation. There is no significance threshold or recurrence criterion for
hypothesis discovery. All effect distributions and exceptions remain visible.

## Data isolation, source artifacts, and frozen model definitions

The canonical cohort contains 835 development patients (645 ER+, 190 ER−) and
209 locked-test patients. Only partition metadata for the latter was inspected
to enforce isolation. Each development fold contains 668 training patients
(516 ER+, 152 ER−) and 167 validation patients (129 ER+, 38 ER−). The validation
folds partition development; the training folds overlap.

Sources are `runs/phase2/pilot_fold0_50e_paired_v1/` for fold 0 and
`runs/phase2/cv_folds1to4_50e_paired_v1/` for folds 1–4, reconciled against the
existing CV provenance manifest. D0 reads saved latent arrays, initial/final
state dictionaries, fitted probe objects, validation predictions, and partition
metadata. Training labels are recovered consistently from the saved development
validation predictions covering all 835 development patients. D0 does not open
expression or label parquet files and does not run an encoder or RNA-seq
preprocessor. Saved latent row order is checked against each fold's saved
representation metadata; prediction IDs and labels are reconciled before use.

The original training protocol remains: training-only CPM/TMM/log2-CPM,
zero-variance filtering, gene-wise scaling, randomized PCA with representation
seed 1729, and PCA-score scaling. A uses four PCA scores; B and B_classical use
32 scores and independently trained 32→8→4 encoders with a Tanh hidden layer.
B and B_classical started with paired encoder initializations but their final
coordinates are not assumed to correspond.

A/B use four qubits, one RY encoding of phi(z)=pi*tanh(z/2), two RY/RZ rotation
layers with the CNOT chain 0→1→2→3, and Pauli-Z measurement on qubit 3. The ER+
probability is clip((1+expectation)/2, 1e−7, 1−1e−7). The original simulator is
CPU float64 default.qubit, analytic expectations, and backpropagation. Original
networks used 50 fixed epochs, unweighted BCE, AdamW lr=0.01, weight_decay=0.001,
and batch size 32. No checkpoint selection was performed. B_frozen_probe is the
already fitted C=1, max_iter=2000 logistic regression on final B embeddings.
B_classical has a sigmoid linear head on its own embedding; it does not apply
the angular map. D0 does not refit A_classical; its original predictive comparison
is already covered by the CV report.

## D0-A methods: separate the three estimands

### Representation variation

Centered singular values, numerical rank at relative threshold 1e−6, coordinate
SD (ddof=0), and variance entropy rank are calculated separately in z and phi(z).
Entropy rank is exp(−sum(w log w)), with w proportional to squared centered
singular values. These are distributional descriptors, not information or
utility measures. The angular transform of B_classical is explicitly hypothetical.

Training means, SDs, coordinate bounds, class-mean differences, and principal
axes are computed only from each training embedding. Principal axes and their
variances use direct SVD of the centered training matrix, with variance s²/n.
This avoids covariance cancellation inflating almost-null directions. They are
then applied without refitting to validation. The sign convention makes axes
reproducible within a run; neither coordinates nor signed principal directions
are aligned across independently learned encoders. A's nearly degenerate
standardized PCA spectrum also makes individual PC orientations non-unique.

### Functional sensitivity

For each A/B initial/final checkpoint, D0 separately computes:

- The map derivative dphi/dz=(pi/2)*(1−tanh(z/2)²), in radians per latent unit.
- Circuit expectation sensitivity df/dphi, per radian.
- Direct df/dz, per latent unit.
- Probability sensitivity dp/dz, including the historical probability clipping.

Direct df/dz is compared against (df/dphi)*(dphi/dz) for every saved patient and
coordinate. The pi factor is included here; the older saturation diagnostic's
tanh-only derivative excluded it. A slope divided by pi/2 measures reduction
relative to the map's maximum slope. It does not imply absolute contraction:
the unnormalized slope can exceed one, and df/dphi and df/dz have different units.
Averages of factors are not substituted for their pointwise product.

For each derivative type, G=mean(gradient*gradient^T) describes aggregate local
response. Its eigenvalues and entropy rank are sensitivity descriptors, not
class-information rank. Probability responses are additionally scaled by
training coordinate SD. Angle derivatives are not scaled by pre-angular SD,
which would mix spaces. Directional probability responses use the training PCs
scaled by their training directional SD, and the training ER+ minus ER− mean
difference. That class contrast does not exhaust nonlinear class information.

B_classical derivatives are calculated in its own coordinates. The probe is
analyzed only on final B embeddings: no initial-encoder probe was fitted.
B/probe gradient cosines are reported both raw and after multiplication by the
same training coordinate SD. Cosines with gradient-norm products <=1e−15 would
be undefined rather than treated as zero; all final validation cosines were
defined in this execution.

### Predictive effects and numerical checks

Original validation predictions are reproduced before interventions. Maximum
absolute discrepancies are required to be <=1e−12. Central probability finite
differences use step 1e−5 on the first 19 canonical-order patients in each split,
with an absolute tolerance of 1e−7. The chain tolerance is 1e−10. A structural
negative control adds 0.1 to all four final-layer RZ angles; the expectation must
remain unchanged to 1e−10. These tolerances check numerical calculations, not
scientific hypotheses. No patient subset was chosen for favorable outcomes.

| Numerical check | Maximum absolute discrepancy |
|---|---:|
| Chain rule, all saved A/B patients and coordinates | 1.0894e−15 |
| Probability derivative versus central finite difference | 1.2631e−10 |
| Redundant final RZ perturbation | 1.1102e−15 |

All 60 final validation head outputs were reconciled with the original saved
predictions. There are 210 finite-difference check groups and 120 A/B
checkpoint/split chain-rule groups. Intermediate checkpoints are unavailable;
initial/final comparisons change both the representation and head and cannot
identify which caused an endpoint property.

## D0-A results

### Variation is already concentrated before the angular map

Final validation means across fold/seed results:

| Representation | Variance entropy rank in z | After angular transform | Numerical rank pattern |
|---|---:|---:|---|
| A | 3.9437 | 3.9503 | Four throughout |
| B | 1.4402 | 1.5235 | Same before/after: six rank-4, eight rank-3, one rank-2 |
| B_classical | 1.1456 | 1.1364, hypothetical | Four throughout |

The mean increase in B's entropy rank does not demonstrate information gain,
and the concentrated pre-angular spectrum does not identify a training cause.
It does weaken a blanket account in which the angular map creates the observed
rank contraction. B_classical's stronger variance concentration remains an
important counterexample to a uniquely quantum explanation of low effective
variation dimension.

For a numerical description of the previously observed coordinate contraction,
consider coordinates whose training SD divided by the largest coordinate SD of
their own run is <1e−6. This identifies ten coordinates in nine B runs, not a new
discovery threshold or independent sample of ten systems. Their mean map slopes
are 0.99424–0.99999 times pi/2, while their RMS df/dphi ranges from 1.95e−12 to
1.11e−6. These coordinate cases have little variation and low local circuit
response, while their map slopes are near maximal. Sensitivity and variation
are separately low; neither proves that valuable information was discarded.
Across all 60 B run-coordinate means, relative map slopes range from 0.2910 to
0.99999, showing that local map attenuation exists elsewhere and is heterogeneous.

### B and the probe have different responses on the shared representation

Mean final validation RMS probability-gradient norm is 0.2634 for B and 0.2186
for the probe. After multiplying gradients by the shared training SD, the means
are 0.2703 and 0.2219. These are not rankings of useful information. Individual
fold/seed means of the raw gradient cosine span −0.0315 to 0.1963, with average
0.0724; after SD scaling they span −0.0853 to 0.1793, averaging 0.0518. The
patient-wise distributions also include both positive and negative alignment.
The two heads are therefore not locally interchangeable, even though they can
have similar discrimination.

Mean sensitivity entropy rank for dp/dz is 1.5485 for B and 1.0000 for the probe.
The latter follows algebraically from a sigmoid linear head: probability
gradients are scalar multiples of a fixed coefficient vector. It is not evidence
that the probe uses less valuable information. B_classical has the same linear-
head property in its own representation, with mean RMS probability-gradient
norm 0.1271, or 0.2612 after its own training-SD scaling. No coordinate-level
B/B_classical correspondence or head-only causal contrast is asserted.

## D0-B design and interpretation

Only final heads are perturbed, one input direction at a time. The 68 fixed
interventions per model/split consist of four training-mean replacements, four
zero replacements, twelve training-donor replacements (four coordinates × three
donor seeds), 24 coordinate-local shifts, and 24 training-PC-local shifts.
Local shifts are +/-alpha times the relevant training directional SD, with
alpha in {0.01, 0.05, 0.10}. All are evaluated; no magnitude is selected using
validation effects. Nearly constant directions receive correspondingly tiny
shifts, without an artificial minimum perturbation.

Diagnostic donor seeds are 271828, 271829, 271830, combined deterministically
with fold, original seed, and split. Training donors are a permutation of training
patients. Validation donors are training patients sampled without replacement,
not a permutation of validation patients. Donor choices use no labels or
outcomes. B and its probe receive exactly the same altered embeddings;
B_classical uses its own training distribution and directions.

Metrics use ER+=1, ER−=0 and the original >=0.5 decision rule. ROC-AUC and AP ER+
use p(ER+); AP ER− uses 1−p and ER− as target. Log loss is in nats; lower is
better. Brier score, decision-flip fractions, absolute/signed probability changes,
and first-order approximation errors are also retained. None of these artificial
outputs replaces the original CV results.

### Predictive changes and exceptions

Each table entry first averages the fixed design cells within a fold/seed, then
averages seeds within fold and the five fold means. Mean/zero rows average four
separate coordinate interventions, not simultaneous replacement of all four
coordinates. Scrambling averages twelve cells. Each local row at alpha=0.10
averages eight signed shifts. Signs, coordinates, donor repetitions, and smaller
magnitudes remain available in detailed outputs.

For B on validation, altered minus original metrics:

| Intervention | Δ ROC-AUC | Δ AP ER+ | Δ AP ER− | Δ balanced accuracy | Δ log loss |
|---|---:|---:|---:|---:|---:|
| Training mean | −0.030661 | −0.013128 | −0.039270 | −0.090172 | +0.043476 |
| Zero | −0.037597 | −0.015465 | −0.048344 | −0.060897 | +0.078233 |
| Training donor | −0.064788 | −0.022031 | −0.121615 | −0.104370 | +0.118291 |
| Coordinate local, alpha=0.10 | −0.000219 | −0.000015 | +0.001397 | −0.000077 | −0.001589 |
| PC local, alpha=0.10 | −0.000333 | −0.000268 | +0.000849 | −0.000226 | +0.000394 |

The sign and magnitude vary. For mean replacement, B's fold/seed average Δ
ROC-AUC spans −0.1041 to −0.00046 and Δ log loss spans −0.0101 to +0.0989.
For zero replacement, Δ ROC-AUC spans −0.1057 to +0.00066, while average log loss
increases in all 15 fold/seed groups, by +0.0359 to +0.1213. Scrambling has Δ
ROC-AUC from −0.1072 to −0.0315 and Δ log loss from +0.0214 to +0.2758.
Those aggregate effects coexist with individually inert or favorable ablations.
Coordinate-local shifts have fold/seed Δ ROC-AUC from −0.00280 to +0.00342 and
Δ log loss from −0.00835 to +0.00126. Small or favorable changes are not grounds
for selecting a validation-optimized intervention.

For B's mean-replacement Δ ROC-AUC, between-fold SD of seed means is 0.0110 and
mean within-fold seed SD is 0.0303. For scrambling they are 0.0098 and 0.0214.
The corresponding log-loss SDs are 0.0105/0.0326 and 0.0306/0.0746. These describe
dependent observations; they are not standard errors or confidence intervals.
`effect_distribution.csv` retains those two variability components and all
fold/seed extrema for every metric, family, and local magnitude.

### Direct shared-encoder comparison and independent classical control

| Intervention | B Δ AUC / Δ loss | Probe Δ AUC / Δ loss | B_classical Δ AUC / Δ loss |
|---|---|---|---|
| Training mean | −0.03066 / +0.04348 | −0.03578 / +0.03815 | −0.00061 / −0.13578 |
| Zero | −0.03760 / +0.07823 | −0.03578 / +0.04553 | −0.00061 / −0.17559 |
| Training donor | −0.06479 / +0.11829 | −0.06083 / +0.14370 | −0.01136 / −0.05555 |
| Coordinate local, alpha=0.10 | −0.00022 / −0.00159 | 0.00000 / +0.00046 | 0.00000 / +0.00056 |
| PC local, alpha=0.10 | −0.00033 / +0.00039 | 0.00000 / +0.00071 | 0.00000 / +0.00217 |

B/probe differences are paired by exactly the same fold, seed, altered input,
and validation patient. The mean difference in scrambling-induced Δ AUC is
−0.00396 (B minus probe), but its fold/seed averages span −0.03953 to +0.07730.
For Δ loss, the paired mean is −0.02541, spanning −0.28880 to +0.13101.
There is no uniform ordering of intervention effects.

B_classical comparisons are aggregate properties of a different learned
representation and head. Its reductions in log loss under mean/zero replacement
are consistent with the already observed probability-generalization problem,
but do not isolate a remedy or demonstrate a common coordinate mechanism.

Two algebraic controls matter. A common translation of all patients' inputs
adds a constant logit offset for a linear logistic head, preserving ranking
apart from numerical ties; this accounts for the zero local Δ AUC of the
probe/classical head. For a given coordinate, replacing it by its training mean
versus zero also differs only by a constant logit offset, so identical AUC
changes are expected for those linear heads. Loss and decisions can still
change. These invariances are not evidence of superior general robustness.

### Nearly constant coordinates: variation versus offset

In the ten numerically near-constant B coordinate cases, mean replacement gives
mean absolute probability changes between 7.58e−17 and 3.43e−13; all have zero
Δ ROC-AUC. Zero replacement gives mean absolute probability changes from
2.17e−5 to 5.12e−3, with angular displacement 0.0105–0.2384 radians. Its Δ
ROC-AUC ranges from −0.000204 to +0.000408 and Δ loss from −0.02542 to −0.0000973.
Thus learned constant levels influence frozen predictions, while their tiny
patient-to-patient variation contributes negligibly under this particular
replacement. The learned levels are not shown to be beneficial: validation
loss decreases under zero replacement in all ten cases.

For the probe on the same coordinate cases, zero replacement changes mean
absolute probability by only 1.82e−8–2.57e−5 and leaves ROC-AUC unchanged.
This is a direct shared-representation contrast. It still cannot establish
whether VQC gradients caused the encoder to reach these solutions.

Zero replacements place every altered patient outside that coordinate's training
range in these ten cases. Their large SD-normalized displacement reflects a
near-zero denominator, not a realistic biological perturbation. A weak derivative
at the original point also does not establish finite-shift invariance: zero
replacement can expose curvature away from a locally stationary solution.

## Effective displacements and distributional limitations

For B's validation embeddings, mean/zero/donor replacements have average z L2
shifts of 0.6965/0.8877/0.9233 and angular L2 shifts of 0.7867/1.0611/1.0216
radians. The average fractions outside at least one training coordinate range
after intervention are 0.00689/0.25529/0.00689. These are averages over separate
interventions, not unique-patient rates. The smaller fractions for mean/donor
replacement do not establish joint-distribution plausibility: recombined
coordinates can lie inside every marginal bound while violating learned
relationships.

At alpha=0.10, coordinate-local and PC-local B shifts average 0.08458 and 0.06934
in z L2, and 0.08465 and 0.06976 radians in angular L2. Their mean outside-bound
fractions are 0.00878 and 0.00863. Every intervention retains the pre-intervention
bound fraction, effective median/95th percentile/maximum shifts, and nearest-
training distance as well. Training nearest-neighbor distances exclude the
recipient's original row. These descriptors are limited support checks, not
certificates of in-distribution intervention.

The descriptive SD-scaled displacement uses a numerical denominator floor of
1e−12*max(1,max training SD); that floor never enlarges the actual perturbation.
No intervention magnitude, donor seed, coordinate, or direction was selected
based on validation. For B coordinate-local shifts, mean absolute first-order
probability error is approximately 1.0e−5 at alpha=0.01 and 1.03e−3 at 0.10.
This scale dependence is consistent with a local approximation, not proof of
useful predictive response. No finite perturbation was interpreted solely
through its derivative.

## Hypothesis assessment and remaining causal questions

| Hypothesis | Descriptive assessment | What remains unresolved |
|---|---|---|
| H1: angular-map attenuation explains contraction | Weakened as a general explanation: contraction exists in z; near-constant coordinates have nearly maximal map slope. Attenuation exists in other coordinates. | Endpoint spectra and derivatives cannot establish the map's effect during training. |
| H2: response is uneven across representation directions | Supported as a local functional description; variation, circuit response, and scaled response differ across directions and runs. | No evidence that weakly responsive directions contain valuable unused information, or that stronger sensitivity would help. |
| H3: near-constant inputs can act as offsets | Supported for frozen dependence on input levels. The stronger claim that the learned constants are beneficial is weakened by reduced loss under zero replacement. | Zero replacements are outside the observed coordinate range; necessity during learning and effects of retraining are unknown. |
| H4: B and probe use the same embedding differently | Supported by gradient distributions and matched intervention effects, with substantial exceptions in metric ordering. | Useful quantum contribution to encoder learning is not identified. |

No binary discovery rule, naive test of 15 independent repetitions, or
validation-based tuning is used. Ten coordinate cases, multiple perturbations,
and repeated patients do not create additional independent cohorts. The five
fold training sets overlap; seeds repeat the same partitions. D0 remains an
exploratory analysis after the original pilot and CV.

New experiments would be needed to test causality: refit prespecified quantum
and classical heads on the same frozen embeddings with documented budgets;
compare probes on z, angular features, and saved initial-encoder embeddings;
alter the angular map during joint training while controlling initialization
and budget; isolate encoder gradient sources with appropriate classical and
frozen/random-encoder controls; and test offset/link/calibration explanations
using training-only fitting and a separately fixed validation plan. These are
new training or calibration experiments, not analyses completed here. A future
geometric intervention would need its own mechanism, controls, and prior-art
assessment. D0 does not select a definitive regularization function or implement
C0/C1, and provides no claim of novelty or quantum advantage.

## Artifacts, reproducibility, and verification

The authoritative execution is `runs/phase2/d0_v1_verified/`; the small review
summaries are in this report directory. The superseded first execution at
`runs/phase2/d0_v1/` and its summaries at `reports/phase2/d0_v1/` were removed
during subsequent cleanup. Its covariance-based
PC calculation was replaced by direct SVD to avoid roundoff inflation of nearly
null directional variances, and only D0 diagnostics were repeated. This is a
numerical implementation correction, not a training-protocol change or a
validation-selected perturbation. The removed execution is not needed to
reproduce the verified results. The verified run and summaries remain intact.

Detailed private outputs include per-patient derivative arrays, per-intervention
predictions, training reference arrays, exhaustive intervention tables, resolved
configuration, source snapshots, input hashes, versions, and runtime. No patient
IDs or patient-level matrices are included in the versionable summary CSVs.
The verified run checks that all 224 accessed historical files retain their
hashes; a broader pre-D0 manifest covers 462 historical files. Final preservation
and test/notebook results are recorded separately in the versionable
[verification record](verification.json). The numerical, test, and notebook
checks in that record describe the completed D0 verification; recorded source
hashes refer to the files at that time, before subsequent presentation edits.

The [read-only notebook](../../../notebooks/03_d0_compatibility.ipynb) loads only
aggregate CSVs and generates figures dynamically. It presents objective,
artifact loading, numerical checks, variation, sensitivity, predictive effects,
support limitations, interpretation, and open questions. It reuses functions in
`src/phase2/d0_views.py`. No PNG/SVG files are required by this report or added
to Git. An executed notebook copy and inspected images stay under `runs/`.

Summary files include `geometry.csv`, `coordinates.csv`, `functional.csv`,
`directions.csv`, `checks.csv`, `baseline_metrics.csv`, `shared_encoder.csv`,
`shared_scaled.csv`, `coordinate_ablation.csv`, `intervention_summary.csv`,
`paired_interventions.csv`, and `effect_distribution.csv`. The latter separates
between-fold variability from mean within-fold seed variability. The exhaustive
individual intervention cells remain in the run archive. `provenance.json`
records the exact source inputs and numerical execution; `verification.json`
also records export/plot/notebook code hashes.

Already executed diagnostic command; use a new output path for reproduction:

```bash
MPLCONFIGDIR=/tmp/qarl-mpl .venv/bin/python -m src.phase2.run_d0 --output runs/phase2/d0_v1_verified --summaries reports/phase2/d0_v1_verified
```

Aggregate exports after a new diagnostic execution:

```bash
MPLCONFIGDIR=/tmp/qarl-mpl .venv/bin/python -m src.phase2.summarize_d0 --run runs/phase2/d0_v1_verified --summaries reports/phase2/d0_v1_verified
MPLCONFIGDIR=/tmp/qarl-mpl .venv/bin/python -m src.phase2.summarize_d0 --run runs/phase2/d0_v1_verified --summaries reports/phase2/d0_v1_verified --shared-scaled-only
MPLCONFIGDIR=/tmp/qarl-mpl .venv/bin/python -m src.phase2.summarize_d0 --run runs/phase2/d0_v1_verified --summaries reports/phase2/d0_v1_verified --distributions-only
```

Each command refuses to overwrite its existing target. Everyday plotting needs
only the existing summaries, not a diagnostic rerun or private archives:

```bash
MPLCONFIGDIR=/tmp/qarl-mpl .venv/bin/python -m jupyter nbconvert --execute --to notebook --output-dir runs/phase2/d0_notebook_new notebooks/03_d0_compatibility.ipynb
```

The verified numerical run took 86.18 seconds in this process, excluding report,
exports, tests, and notebook rendering. This is CPU diagnostic runtime, not model
training or quantum-hardware cost. Its duration is affected by concurrent host
activity and is not a controlled benchmark. Environment versions are Python
3.12.11, Torch 2.9.0, PennyLane 0.43.0, NumPy 2.5.3, pandas 2.3.3,
scikit-learn 1.9.1, SciPy 1.18.1, and Matplotlib 3.10.7. The execution Git base
is `9e6e2582dc643adc78fab77adae8e3f97b3e5f20`; new source content is identified
by hashes and remains uncommitted. No commit or push was performed.

Final verification passed 116 tests, including ten D0 tests covering numerical
identities, training-only perturbations, nearly-null SVD scales, shared-input
pairing, saved-summary reconstruction, the locked-source gate, and read-only
notebook execution. The historical XGBoost test module was excluded because
`xgboost` is not installed; no packages were changed. The notebook executed
top-to-bottom and all seven figures were visually inspected. All 462 historical
file hashes in the pre-D0 manifest remained unchanged. `verification.json`
records the exact command and content hashes.
