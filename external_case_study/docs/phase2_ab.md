# Experiments A and B: development protocol

## Audit and scope

Audit on 2026-10-28: Git was initially clean. Phase 1 uses `src/data/`, pytest,
scikit-learn, and a uv/Python 3.12 environment. No PCA, VQC, or Phase 2 implementation
was present. The canonical counts (1044 × 60660), ER+=1/ER−=0 labels, and
`data/processed/splits/phase1c_inner_cv.json` were available: 835 development
patients, 209 locked-test patients, five folds, and seed 1729. No essential local
artifact was missing. These data are not versioned and must be obtained through
the Phase 1 procedure when reproducing the work on another machine.
Phase 1 pipelines and results are unchanged.

The new loader applies an Arrow filter to development IDs before materializing
counts and labels. Locked-test IDs are consulted only to verify isolation;
locked observations are not transformed and their labels are not evaluated.
Whole-file hashes provide provenance only, without calculating statistics.

## Frozen pilot architectures

All transformations are fitted on training counts within each fold: CPM filtering,
frozen-reference TMM, log2-CPM, exact zero-variance removal, and gene-wise
StandardScaler, reusing Phase 1 `_build_preprocessor`. No matrix normalized over
the entire cohort is reused.

- A: randomized PCA k=4 → score StandardScaler → angles → VQC.
- B: randomized PCA k=32 → score StandardScaler → Linear(32,8), Tanh,
  Linear(8,4) → angles → the same VQC. The encoder has 300 parameters;
  AdamW uses weight decay 0.001. An MLP is not trained over all genes.
- A_classical: the same four standardized scores → LogisticRegression,
  C=1, with an intercept, no class weighting, and at most 2000 iterations.
- B_classical: exactly B's intermediate representation and encoder architecture,
  trained from scratch → Linear(4,1), Sigmoid.
- B_frozen_probe: B's final-epoch encoder in evaluation mode, without gradients;
  a new LogisticRegression C=1 is fitted only on training z and evaluated on
  validation z. Neither encoder, C, nor threshold is selected using validation.
  Its checkpoint reference identifies B's encoder and representation.

PCA uses fixed seed 1729, independently of network initialization. Explained
variance, the complete transformation state, and fitting IDs are saved.
Angular scaling is `pi*tanh(z/2)` in [−π,π]. For A, the mean and scale of z come
only from training; for B, z is learned jointly and the angular function is fixed,
differentiable, and does not require validation statistics. Classical controls
use scores/z before the angular function; this head difference is part of the
comparison and does not establish capacity equivalence.

The single circuit uses four qubits, one RY encoding of the four angles, two
RY/RZ layers per qubit, and CNOT gates 0→1, 1→2, 2→3 in each layer. It measures
⟨Z₃⟩. `p(ER+) = clip((1+⟨Z₃⟩)/2, 1e-7, 1-1e-7)`, with unweighted BCE and a
fixed threshold of 0.5. There is no additional trainable calibration.
Simulation uses CPU float64, `default.qubit`, shots=None, no noise, and
`interface=torch, diff_method=backprop`, following the
[official documentation](https://docs.pennylane.ai/en/stable/introduction/interfaces/torch.html).

The VQC has 16 parameters; the linear head has 5. Nominal counts do not imply
equal capacity: some rotations may be redundant for the observable. The LR
diagnostic uses a different optimizer and budget. Its timing excludes the prior
encoder cost, which must be added when comparing total costs. B and B_classical
share their encoder architecture (300 parameters), epochs, minibatch size,
learning rate, and regularization, but not necessarily the same optimization
trajectory. In the original smoke test, A/B shared a configuration but had
different initialization draws. This was corrected for the 50-epoch pilot:
encoder construction forks the CPU RNG, and A/B share exactly the initial VQC
weights and RNG state at the start of shuffling for each seed. The encoder
matches B_classical. See the pilot report for the audit and limitations.

## Execution

From the repository root using Python 3.12 (the quantum extra requires Python >=3.11):

```bash
uv sync --extra dev --extra quantum --locked
uv run --extra dev --extra quantum python -m pytest -q
uv run --extra quantum python -m src.phase2.run --smoke --output runs/phase2/smoke_nuevo
uv run --extra quantum python -m src.phase2.run --config configs/phase2_ab.yaml --output runs/phase2/cv_ab
```

Independent configurations are `configs/phase2_a.yaml` and `configs/phase2_b.yaml`;
both use the same infrastructure and execute their controls. Every output
directory must be new: existing executions are never overwritten. There is no
locked-test command. Existing path names are retained for compatibility.

`--smoke` forces fold 0, seed 1729, and two epochs while retaining ALL patients
in the fold. For multiple seeds in one partition, copy the YAML and use
`folds: [0]`, `seeds: [1729,1730,1731]`. The full protocol uses five folds,
three seeds, and 50 epochs. Networks receive the same epoch and optimizer-step
budget (21 minibatches/epoch in fold 0). The final epoch is evaluated; there is
no early stopping, hyperparameter search, or exploratory selection.
Validation is recorded each epoch for diagnostics only.

## Artifacts and metrics

Each execution directory contains the expanded configuration, versions, full
environment, Phase 2 source snapshot, commit and tracked-change patch, data/split
hashes, and `comparison.csv`. Each fold/experiment retains its fitted
preprocessor/PCA. Each model/seed retains a checkpoint, JSON metrics, validation
prediction CSV, and learning curves for networks. Curves record training/validation
BCE and the mean minibatch L2 gradient norm for encoder and head; A's identity
encoder has zero norm by definition.

Metrics include ROC-AUC, AP ER+, AP ER−, balanced accuracy, accuracy, log-loss,
and a confusion matrix ordered [ER−,ER+]. Per-model seconds cover training and
per-epoch evaluation; preprocessing is recorded separately. QNode calls count
batches, and `circuit_sample_evaluations` counts patient states including training
and validation evaluations. Backpropagation reuses differentiable simulation:
these counts do not represent hardware executions or parameter-shift evaluations.
`resources.json` contains total time and process peak RSS with units.

Checkpoints are saved under `runs/phase2/`, ignored by Git. Pickle and state_dict
artifacts permit restoration using `Representation`, `Model(inputs, encoder=...)`,
and the recorded versions; load only trusted artifacts. No copy of the expression
matrix is saved. PCA state may occupy several MB and is shared across seeds.
The `source/` snapshot and its hashes also preserve Phase 2 code not yet added
to Git; Phase 1 preprocessing is referenced through the commit.

## Levels of evidence

1. Synthetic tests: software integrity, not biomedical evidence.
2. Real two-epoch smoke test on one partition: functionality, not a final estimate.
3. Single-partition pilot with multiple seeds: preliminary stability.
4. Five folds and multiple seeds: development evaluation. Compare paired results
   by fold/seed; folds and seeds are not independent replicates.
5. Future confirmatory experiments: a preregistered protocol and eventual locked
   test only when scientifically authorized. None of these steps demonstrates
   quantum advantage.

See `docs/phase2_ab_results.md` for the original executed smoke test and next
steps. C0 is not implemented; NQE/QPMeL remain outside the current scope.

## Completed 50-epoch pilot

Configuration: `configs/phase2_pilot_fold0.yaml`, fold 0, seeds 1729/1730/1731.
The [technical report](../reports/phase2/pilot_fold0_50e_verified/report.md) includes
results by seed, descriptive SDs, paired comparisons, diagnostic interpretations, and costs.
Evaluation still uses the final checkpoint; epochs and hyperparameters were not
selected using the new results.

Additional diagnostics and analysis can be reproduced using the report commands.
`latent_diagnostics.npz` stores initial/final training and validation z; the curves
record angular saturation and per-parameter quantum gradient RMS.
