# A/B pilot: fold 0, 50 epochs, and three seeds

## Protocol and audit

Seeds 1729, 1730, and 1731 were evaluated using the existing fold-0 partition:
668 training patients and 167 validation patients (129 ER+, 38 ER−). All five
variants are complete. The locked test was not used, Phase 1 was not modified,
and C0, NQE, QPMeL, and C1 were not implemented. The primary evaluation always
uses the final checkpoint, epoch 50, without validation-based selection.
This is a single-partition pilot, not five-fold validation.

Git was clean at the start of the pilot task. One methodological issue was
identified and corrected: B consumed random numbers when constructing its
encoder before the VQC. Encoder construction now occurs within
`torch.random.fork_rng(devices=[])`, so A/B receive identical quantum weights
and begin minibatch shuffling from the same RNG state for each seed. The encoder
retains its seed-specific initialization and matches the encoder in B_classical.
Tests and saved initial checkpoints verify this correspondence. Neither the
architecture nor the hyperparameters were changed in response to the smoke test.
Additional instrumentation only observes angles, gradients, and states; it does
not make optimization decisions.

**Frozen circuit:** four qubits, one RY encoding of `pi*tanh(z/2)`, two layers of
RY/RZ rotations per qubit followed by CNOT gates 0→1, 1→2, 2→3, and a Pauli-Z
measurement on qubit 3. Simulation uses ideal CPU float64, default.qubit,
shots=None, and PyTorch backpropagation. The ER+ probability is
clip((1+⟨Z₃⟩)/2, 1e-7, 1-1e-7), with unweighted BCE and a fixed threshold of 0.5.
AdamW uses lr=0.01, weight_decay=0.001, batch=32, and 50 epochs: 1050 optimizer
steps per network/seed. A_classical and the probe use LR with C=1 and
max_iter=2000, with a different optimizer and budget.

**Representations:** Phase 1 CPM/TMM/log2-CPM/zero-variance filtering and gene-wise
StandardScaler are fitted exclusively on training patients. Randomized PCA uses
random_state=1729, k=4 for A and k=32 for B, followed by a score StandardScaler
also fitted only on training. The B encoder is 32→8→4 with Tanh between its linear
layers, 300 parameters, and AdamW regularization. The VQC has 16 nominal
parameters; the classical head has 5. B/B_classical therefore have 316/305 nominal
parameters in total. These counts do not establish equal capacity. The angular
function is fixed; the classical controls operate on z before this function.
The probe fits a new LR head exclusively on training z from the frozen final
encoder, without selecting C, the encoder, or a threshold using validation.

## Diagnostics before training

The four RZ rotations in the final layer are structurally redundant: propagating
Z₃ backward through the final CNOT chain yields Z₀Z₁Z₂Z₃, which commutes with all
four rotations. They are retained to preserve the frozen architecture. The other
12 parameters showed nonzero sensitivity at the generic points examined; this
does not establish a global functional rank of 12 or parameter independence.

The maximum discrepancy between automatic differentiation and finite differences
was $sensitivity_error. The fixed synthetic task reduced BCE from
$synthetic_initial_loss to $synthetic_final_loss and achieved accuracy
$synthetic_accuracy on its six examples. This checks learnability for that task,
without demonstrating generalization or comparative advantage. Tests cover finite
gradients, differentiability, paired initialization, and persisted metrics.
The preflight records initial angles without using them to modify the protocol.

## Observed results

Mean ± sample standard deviation (ddof=1), descriptive across three seeds.
These are not confidence intervals or three independent cohorts. A_classical is
approximately deterministic; its three repetitions do not add independent evidence.

$summary_table

### Each model and seed

$metrics_table

### Paired comparisons

$paired_table

Complete differences, including AP and losses, are available in
[paired_comparisons.csv](paired_comparisons.csv). Scores were joined by patient ID
with one-to-one cardinality, 167 patients, and identical labels. Spearman correlation
is calculated on ER+ probabilities; it does not imply identical calibration or
clinical equivalence.

Notebook diagnostic: **BCE learning curves** (see the analysis notebook and execution guide).

Training losses are online minibatch averages during parameter updates;
validation loss corresponds to the state at the end of each epoch. These are
not two losses evaluated using the same checkpoint. The validation minimum is
not used to select the reported metrics.

Notebook diagnostic: **Gradient evolution by epoch** (see the analysis notebook and execution guide).

Notebook diagnostic: **Gradients by parameter** (see the analysis notebook and execution guide).

The evolution panels use independent y-axis scales to show within-group dynamics;
axis values retain their units. The aggregate norm includes nominal parameters.
The redundant RZ rotations have loss gradients near numerical precision;
AdamW may move them through weight decay without demonstrating predictive influence.
Nonzero gradients in relevant parameters rule out complete disconnection but do
not establish usefulness or favorable optimization conditioning.

Notebook diagnostic: **B and frozen-probe scores** (see the analysis notebook and execution guide).

## Angles and saturation

Diagnostic saturation is defined as |tanh(z/2)|>0.95. Rates above 0.99 and rates
with derivative of tanh(z/2) below 0.01 are also reported. These thresholds are
descriptors, not criteria for modifying the method. Rates pool patient × coordinate
values; the CSV retains individual coordinates and quantiles. The recorded
derivative excludes the π factor in the angle. For B_classical, these angles are
hypothetical: its head does not apply the angular tanh transformation.

Notebook diagnostic: **Angular distributions** (see the analysis notebook and execution guide).

Notebook diagnostic: **Angular saturation over training** (see the analysis notebook and execution guide).

These panels have independent y-axis scales; compare magnitudes using their axes
and the table below. A retains a training rate of approximately 0.0374% and zero
in validation, identical across epochs because the representation is fixed.

$angles_table

### Final centered latent spectrum (training)

$spectrum_table

## Costs

$cost_table

Total wall time: $wall_seconds s; peak RSS: $rss_gib GiB.
The monitored training cost of B was $cost_B_over_A× A and
$cost_B_over_classical× B_classical in this process.
The pilot wall time excludes preflight diagnostics and report generation.
Per-network times include evaluation each epoch and angular instrumentation;
they exclude preprocessing and final prediction. Preprocessing was performed once
per representation and shared across its three seeds and controls: do not sum the
repeated column six times. LR timing includes fitting and prediction; probe timing
excludes prior B training, which must be added for a standalone pipeline comparison.
B/B_classical have 1050 steps, whereas LR uses a different budget. Each VQC records
batch calls and patient-state evaluations, not hardware executions or
parameter-shift evaluations. Timings from one process are not a robust benchmark:
warm-up, memory, and monitoring may affect them.

## Decisions, hypotheses, and limitations

### A. Was A undertrained in the smoke test?

Yes, in the limited sense of the two-epoch budget. For seed 1729, the first two
epochs reproduce the smoke-test losses ($first_two_match), and ROC-AUC increases
from $smoke_auc to $final_a_auc; validation BCE decreases from
$smoke_validation_loss to $final_a_validation_loss. This does not imply that
50 epochs achieve the optimum or justify extending the budget here.

### B. Does B improve consistently over A?

The mean B−A ROC-AUC difference is $delta_mean, with sample SD $delta_std;
seed-specific differences are $delta_values. There are two improvements and one
tie in ROC-AUC, with seed-dependent magnitude. This supports a descriptive average
improvement in this partition without establishing stability across partitions
or uniform improvement across metrics. B has ROC-AUC SD $b_auc_std, compared with
$a_auc_std for A. A_classical (PCA4+LR) achieves $a_classical_auc, above B's mean.

### C. Does z support a comparable linear head?

Yes, in the descriptive sense of discrimination within this partition: the probe
exceeds B's ROC-AUC for all three seeds (differences: $probe_delta_values).
Seed-specific Spearman correlations are $rho_values. Neither statistical
equivalence nor score identity is demonstrated. The probe has lower mean AP ER−
and higher log-loss for all three seeds: ranking, class-specific precision,
thresholding, and probability assignments characterize different aspects.
The pipeline's performance cannot be attributed to the VQC alone, nor can one
conclude that the VQC did not influence encoder training: z was learned jointly
with it.

### D. Capacity or optimization anomalies

The initialization mismatch was resolved before the pilot. The four redundant
RZ rotations are retained, and nominal parameter counts are distinguished from
observed sensitivity. Relevant gradients remain finite and are not completely
disconnected; near-zero gradients in the final RZ rotations have a structural
explanation. There is no evidence here of a compatibility failure or a complete
learning failure.

There are signs of overfitting: B ends with online training BCE
$b_train_losses and validation BCE $b_validation_losses. For B_classical,
training loss approaches zero and validation loss deteriorates substantially
relative to epoch 2; high ROC-AUC does not imply well-behaved probabilities.
No earlier epoch is selected, and the head is not calibrated in response to this
result. Angular saturation alone does not explain this pattern: B's final rate
above 0.95 varies by seed, and the B_classical head does not use the angular function.

Another capacity observation is contraction of one latent coordinate in two B
seeds. In training, seed 1730 has std(z3)≈4.0e-9 and seed 1731 has std(z1)≈4.0e-10
(zero-based indices); their centered matrices have numerical rank 3 using a
relative singular-value threshold of 1e-6. Seed 1729 retains rank 4. These are
nearly constant learned outputs, not an explicit reduction in the number of VQC
inputs. The checkpoint also confirms near-zero output-layer row norms for those
coordinates (`solution_state_diagnostics.json`). This limits interpretation of a
nominally 4D latent space and should be monitored across five folds before C0.
It does not by itself demonstrate an implementation defect or justify redesigning
the architecture using validation. The complete spectrum is in `latent_spectrum.csv`.

**Explanatory hypotheses:** end-to-end optimization and the 300-parameter encoder
may amplify variability and specialize to training; A's fixed representation and
restricted head may constrain its fit. These are hypotheses, not identified causal
mechanisms. Neither regularization nor capacity was changed to improve validation.

**Recommendation: evaluate five folds with the frozen configuration of 50 epochs
and three seeds before preparing C0.** No known blocking technical failure remains,
but variability and overfitting require evaluating the complete protocol and
reporting discrimination and loss together. Removing redundant RZ rotations,
changing the budget, or adding early stopping would constitute a new protocol
version requiring independent justification; none was applied in this pilot.
Claims of novelty, general superiority, or quantum advantage are not justified.


## Interpreting the notebook diagnostics

The notebook reconstructs aggregate plots from the six saved summary CSVs.
Validation BCE minus online training BCE is a descriptive difference between
losses measured at different parameter states, not a same-checkpoint estimate
of generalization error. Its positive separation is more pronounced for
B_classical than B; A retains negative values. A negative difference does not
establish the absence of overfitting, nor identify a causal contribution by layer.

Relative singular-value plots hide absolute scale; the saved spectrum table
retains it. Rank uses a relative threshold of 1e-6 and does not measure class
separability. A and B_classical retain rank four; B has rank three for two seeds.
A nearly constant learned coordinate may act as a bias, rather than reducing the
number of circuit inputs. These observations do not justify imposing rank four.

Gradient panels and saturation panels use independent vertical scales. Nonzero
gradients do not establish predictive usefulness, and saturation is not shown to
cause overfitting or latent contraction. B_classical angular diagnostics are
hypothetical because its head operates on z directly. Exact angular histograms
and score scatterplots require the private development archive and are generated
by the separate analysis CLI; they are not required for notebook execution.

For future C0, these remain hypotheses: assess whether prespecified geometric
regularization changes stability, contraction, separability, and gradients while
retaining classical controls and the fixed checkpoint rule. Do not tune against
these pilot diagnostics or infer quantum advantage.

## Reproduction and evidence

```bash
uv sync --extra dev --extra quantum --extra analysis --locked
uv run --extra quantum --extra dev python -m pytest -q
uv run --extra quantum python -m src.phase2.diagnostics --development-angles --output runs/phase2/preflight_nuevo
uv run --extra quantum python -m src.phase2.run --config configs/phase2_pilot_fold0.yaml --output runs/phase2/pilot_nuevo
uv run --extra quantum --extra analysis python -m src.phase2.analyze_pilot --run runs/phase2/pilot_nuevo --output reports/phase2/analisis_nuevo --preflight runs/phase2/preflight_nuevo/diagnostics.json --smoke runs/phase2/smoke_verified_1729
```

These commands document the full historical protocol. For analysis-only
regeneration, invoke only `src.phase2.analyze_pilot` on the existing run, saved
preflight, and smoke artifacts, with a new output directory. Do not retrain to
translate or present results. Existing path names are retained for compatibility.

Never overwrite existing directories. The run directory retains the complete
configuration, commit/patch, environment, hashes, source code, fitted
representations, initial/final checkpoints, curves, and predictions.
Prediction CSV files are read with `float_precision='round_trip'`: the default
pandas parser changed log-loss by 2.7e-8 for extreme B_classical probabilities.
Original precision was retained without changing models or saved metrics.
Report tables do not include patient IDs. Predictions with IDs remain under
`runs/phase2/`, ignored by Git. The report CSV files permit calculation inspection;
`analysis_provenance.json` records their sources and the analysis code.

## Interactive figures

Use [the analysis notebook](../../../notebooks/01_representation_analysis.ipynb)
and [execution guide](../../../notebooks/README_representation_analysis.md) to
render figures from saved development outputs. Generated images remain local.
