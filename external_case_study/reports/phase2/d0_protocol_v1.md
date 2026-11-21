# D0-v1: exploratory compatibility diagnostics for frozen representations and heads

## Status and scope

This protocol implements the approved D0-A and D0-B plan with four revisions:
representation variation, functional sensitivity, and predictive effects are
separate estimands; the angular chain rule is checked numerically; no recurrence
threshold is used for discovery; and coordinate-level B/probe comparisons are
restricted to their shared encoder. B_classical has its own learned coordinate
system and is compared using aggregate properties only. This is exploratory
post-CV analysis, without new training, calibration, checkpoint selection, C0,
C1, or locked-test evaluation. The experiment uses the original five development
folds and seeds 1729, 1730, 1731.

Each fold has 668 training and 167 validation patients. Saved initial and final
latent arrays, initial/final head checkpoints, trusted fitted probe objects,
validation predictions, and split metadata are sufficient. No raw expression or
label file is needed. Development training labels can be recovered from the
existing validation predictions covering the development cohort. Intermediate
checkpoints are absent; endpoint comparisons do not reconstruct training paths.

## Three distinct questions

1. **Variation:** how do centered latent values and their angular transforms
   vary across patients? Spectra, numerical rank (relative threshold 1e-6),
   variance entropy rank, and coordinate standard deviations describe variation.
   None establishes label information or predictive usefulness.
2. **Functional sensitivity:** how does a frozen head respond locally to an
   input displacement? Derivatives describe response, not the usefulness of the
   input direction. A highly responsive direction may contain nuisance variation;
   a weakly responsive direction need not contain valuable ignored information.
3. **Predictive effects:** how do saved-head predictions and development metrics
   change under an artificial intervention? These measure the frozen predictor's
   dependence on inputs under that intervention, not the necessity of those inputs
   for learning or their clinical relevance.

## D0-A: endpoint geometry and derivatives

For A/B initial and final checkpoints, compute the angular map
phi(z)=pi*tanh(z/2), its derivative dphi/dz=(pi/2)*(1-tanh(z/2)^2), VQC expectation
sensitivity df/dphi, direct df/dz, and probability sensitivity dp/dz including
clipping. Compare direct df/dz against (df/dphi)*(dphi/dz) on all saved patients;
require maximum absolute discrepancy <=1e-10. The derivative includes pi, unlike
the historical tanh-only saturation derivative. Attenuation of the map and low
response of the circuit are separately reported; spectra alone cannot identify
causality. Classifier probabilities remain subject to their historical links.

Use training principal axes from a direct SVD of the centered embedding,
with directional variances equal to squared singular values divided by the
training sample count, standard deviations (ddof=0), bounds,
and ER+ minus ER− class-mean difference estimated on training only. Apply them
unchanged to validation. Evaluate directional probability derivatives along
training PCs (scaled by the square root of their training variance) and along
the training class-mean difference. The latter is a descriptive class contrast,
not a fitted classifier or a test of all class information.

Aggregate G=mean(gradient*gradient^T) separately for df/dphi, df/dz, and dp/dz.
Report eigenvalues and sensitivity entropy rank without interpreting them as
information rank. Coordinate-scale-adjusted probability responses use training
SD. Do not scale angle derivatives by pre-angular z SD, which would mix spaces.
Probe derivatives are evaluated on final B embeddings only; its fitted head is
not a matched initial-encoder probe. B_classical initial/final probability
sensitivities are computed in its own z coordinates. Its angular geometry is
hypothetical because its head does not use phi.

Numerical controls: reconcile all 60 final validation model/head outputs with
saved predictions (absolute tolerance 1e-12); use central probability finite
differences with step 1e-5 on the first 19 patients in canonical split order,
requiring max absolute error <=1e-7; and perturb all four structurally redundant
final RZ angles by +0.1, requiring expectation change <=1e-10. These are numerical
validation tolerances, not discovery thresholds.

## D0-B: fixed final-head interventions

Evaluate A, B, B_frozen_probe, and B_classical on train and validation. No
intervention is selected, excluded, or scaled using validation effects. For each
coordinate, replace values by (a) its training mean, (b) zero, or (c) training
donors. For training, donors are a permutation of training rows. For validation,
donors are training rows drawn without replacement, rather than a permutation of
validation rows. This preserves the approved coordinate-scrambling question
while ensuring the donor distribution comes exclusively from training. The
validation procedure is explicitly not a within-validation permutation.

Use three diagnostic donor seeds: 271828, 271829, 271830, combined with fold,
original model seed, and split. Donors are independent of labels and outcomes.
B and its probe receive identical altered B embeddings. B_classical receives
interventions defined by its own training embedding; coordinate j is not assumed
to correspond to coordinate j in B.

For all four coordinates and all four training principal axes, apply symmetric
local shifts of +/-alpha times the relevant training SD, with alpha fixed to
0.01, 0.05, 0.10. No minimum perturbation is imposed on a nearly constant
direction. There are 68 interventions per model/split: 4 mean, 4 zero, 12 donor,
24 coordinate-local, and 24 principal-axis-local. All 120 final model/split
combinations are evaluated, for 8160 intervention cells.

Record changes in ROC-AUC, AP ER+, AP ER−, balanced accuracy, accuracy, log loss,
Brier score, probabilities, decisions, and local linear approximation error.
For effective displacement, retain mean/median/95th percentile/maximum L2 changes
in z and angles; training-SD-scaled displacement; fraction outside training
coordinate bounds before/after; and nearest-training Euclidean distance
before/after. For training, exclude the recipient's original row when measuring
nearest training distance. A floor of 1e-12*max(1,max training coordinate SD) is
used only in the descriptive SD-scaled displacement, never to enlarge a shift.
Bounds include a 1e-12*max(1,training coordinate range) numerical tolerance.

Zero replacements, donor recombinations, and even small local shifts can create
out-of-distribution inputs. Marginal bounds and nearest-neighbor distance are
limited descriptors, not proof of manifold membership. A constant angular offset
can matter while its patient-to-patient variation does not. Compare mean and zero
replacement to distinguish these questions.

## Hypotheses and interpretation

- **H1:** the angular map attenuates variation still present in z. Quantify map
  derivative and functional response separately. Contraction already present in
  z weakens an explanation based solely on angular-map compression. No spectrum
  comparison identifies a training cause.
- **H2:** the circuit responds weakly to some varying directions. Report response
  and variation separately. A small response is evidence of weak local response,
  not evidence of discarded useful information. Predictive interventions provide
  an additional, still model-conditional descriptor.
- **H3:** nearly constant B coordinates may act as fixed circuit inputs. Mean
  replacement that preserves outputs together with consequential zero replacement
  supports this dependence at the frozen solution. It does not establish that
  keeping the coordinate during training was necessary.
- **H4:** B and the linear probe respond differently to the same representation.
  Compare probability gradients, their cosine distribution, and matched
  perturbation effects. Differences do not establish that VQC gradients were
  irrelevant to encoder training, nor establish quantum advantage.

Report every fold/seed, effect sizes, distributions, exceptions, and within-fold
seed variability. Summaries first average the fixed intervention cells within a
fold/seed; no individual patient, intervention, seed, or overlapping fold is
promoted to an independent replicate. No recurrence rule, naive 15-row test,
validation-optimized perturbation, or confirmatory p-value is used. Hypotheses
may be descriptively supported, weakened, or unresolved; none is accepted as a
causal mechanism by D0 alone.

## Existing coverage and future causal experiments

The CV notebook already presents predictive metrics, B/probe differences,
checkpoint-matched generalization gaps, learning curves, latent spectrum,
coordinate variability, saturation, aggregate gradient norms, and reliability.
D0 adds numerical chain decomposition, real-embedding sensitivities, shared-head
response comparisons, and frozen interventions. It must not label the existing
CV plots as new D0 evidence.

Different head fits on a fixed encoder, probes on initial encoders or angular
features, altered angular maps during training, encoder-gradient controls,
calibration fits, and geometric objectives are new experiments. They require a
separate protocol, training-only fitting, appropriate controls, and explicit
budgets. D0 does not implement them or select a definitive regularizer.

## Delivery and preservation

Detailed arrays, intervention predictions, and exhaustive tables belong under a
new Git-ignored runs/ directory. Small non-patient-level summaries, an independent
English Markdown report, and a read-only notebook provide reproducible review.
Source and input hashes, numerical checks, runtime, and artifact preservation
are recorded. Existing run/report paths must never be overwritten; no commit or
push is authorized.
