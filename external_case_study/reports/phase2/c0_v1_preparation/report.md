# C0-v1 implementation preparation and technical readiness

**Status: availability audit passed; implementation and synthetic technical checks completed; full development CV not launched.** Prepared 2026-10-28 (America/Lima). The user approved the recommended four-arm scientific design before this work. This report is an implementation-readiness artifact, not a scientific results report or an authorization to run all 60 pipelines.

## Approved scope and immutable interpretation

The [scientific protocol](../c0_protocol_v1.md) remains byte-for-byte unchanged, SHA-256:

`f166d168e1ceadb7ded89dccaa905cd97a379ab0e3f4237d404f05f78ffde4ae`.

Its original design-task status text is preserved as historical provenance; the subsequent user approval is recorded here rather than by rewriting that artifact. The approval covers C0-NQE-Q, C0-NQE-E, C0-Hybrid-Q and C0-Hybrid-E; the supervised objectives; 64 balanced pairs (16 ++, 16 --, 32 +-), without diagonals; lambda=1; fixed Gaussian denominator 4; historical initializations; five folds; three seeds; stage freezing and fixed budgets. The [configuration](../../../configs/c0_v1.yaml) mirrors the closed in-code specification and is validated for equality before execution.

The user's three interpretation requirements are explicit:

1. **Q minus E within each training schedule is primary.** Comparisons with historical B are contextual because procedures, data exposure and total computational budgets are not fully equivalent. The implementation's paired output selects only the prespecified VQC endpoints, not the best diagnostic readout.
2. **No computational quantum advantage claim.** Both the RY product fidelity and the Gaussian comparator are evaluated by classical analytic arithmetic. Auxiliary pair losses make no quantum-device calls.
3. **Distinct quantities remain distinct.** Class-state trace distance, raw observable class-mean contrast, clipped prediction contrast and predictive metrics are separate outputs. The accessibility ratio is explicitly marked as not being a complete discriminative-capacity measure. Unclipped contrast is used in the operator identity; clipped contrast is recorded separately.

No C1 work, architecture changes, scientific coefficient changes, retrospective selection, commit or push occurred.

## 1. Availability and provenance audit

**Result: passed for all five folds and all 15 fold/seed groups.** The audit was completed before any C0 implementation or technical fitting. Source availability is not inferred from a synthetic test.

| Requirement | Observed evidence | Result |
|---|---|---|
| Historical preprocessing | Five B `representation.pkl` objects, one per fold, with SHA-256 matching the completed CV provenance | Passed |
| Fold-local fitted state | PCA has 32 components and 668 fitted samples; component scaler records 668 samples; saved training IDs match canonical order; TMM reference belongs to training | Passed |
| Initial checkpoints | All 15 original B `initial_model.pt` tensors, seeds 1729/1730/1731, match both recorded source hashes and regeneration from the original Model initialization | Passed |
| Saved initial embeddings | All 15 historical `latent_diagnostics.npz` inputs have training (668,4) and validation (167,4) initial embeddings | Passed |
| Partition identifiers | 835 development IDs, five disjoint 167-patient validation partitions and corresponding 668-patient training sets; 209 locked IDs used only to check isolation | Passed |
| PCA32 inputs | No standalone original PCA32 matrices were found. Applied the verified historical fitted preprocessors, without refitting, to development-only rows as expressly allowed by protocol Section 2 | Passed via the pre-authorized reconstruction route |
| Initial embedding reconstruction | Every reconstructed PCA32 split passed through its corresponding original initial encoder | Maximum absolute error **0.0** across all 30 fold/seed/split comparisons; tolerance 10⁻¹² |
| Audited-cache loader | Dry-loaded all 15 groups from the new caches with hashes, row ordering, class counts, initial tensors and embedding reconstruction rechecked | Passed; no model fitting |

The original matrices were not silently assumed to exist. The reconstruction route uses `Representation.transform` with the original trusted objects; every fitted component's `fit` method was replaced in memory with a function that raises, so accidental fitting would have failed the audit. No historical object was serialized back to its source location. The processed expression file's opaque fingerprint matched historical provenance, while Arrow row filtering materialized only the 835 development rows. No expression/label locked rows were materialized, and no label parquet was opened. Development labels for the later loader are authenticated from the existing development validation prediction files and checked against canonical row membership and class counts.

The audit created five new private caches under `runs/phase2/c0_v1_availability_audit/`, with training shapes (668,32), validation shapes (167,32), and explicit ordered IDs. These contain patient-level derived features and remain Git-ignored. Their file hashes and all 44 accessed historical source hashes are retained in the private `audit.json`. Original runs remain at `runs/phase2/pilot_fold0_50e_paired_v1/` and `runs/phase2/cv_folds1to4_50e_paired_v1/`.

The audit took 14.489 s on this host; peak memory was not recorded for that operation. PyArrow emitted non-fatal sandbox CPU-cache-query warnings, completed transformation, and reproduced embeddings exactly. No permission change, dependency installation, or preprocessing change was made.

Actual audit command used a temporary read-only scientific audit script:

```bash
MPLCONFIGDIR=/tmp/qarl-c0-mpl XDG_CACHE_HOME=/tmp/qarl-c0-cache PYTHONPATH=. .venv/bin/python /tmp/audit_c0_availability.py
```

The same audit computation is now retained in [c0_audit.py](../../../src/phase2/c0_audit.py) with a CLI wrapper. A later independent audit can use a **new** output destination:

```bash
MPLCONFIGDIR=/tmp/qarl-c0-mpl XDG_CACHE_HOME=/tmp/qarl-c0-cache .venv/bin/python -m src.phase2.c0_audit --output runs/phase2/c0_v1_availability_audit_new
```

This latter command is documented, not an additional executed audit. The [aggregate readiness record](readiness.json) provides nonsensitive audit counts and errors without publishing patient identifiers or features.

## 2. Implemented components

| Artifact | Responsibility |
|---|---|
| [c0.py](../../../src/phase2/c0.py) | Closed configuration, deterministic initialization and random streams, stratified non-diagonal sampling, exact Q/E objectives, stage-specific optimizers, freezing, fixed-final-checkpoint training, gradient-component diagnostics and counters |
| [c0_data.py](../../../src/phase2/c0_data.py) | Authenticated audited caches; canonical IDs/class counts; source/checkpoint hashes; initial embedding reconstruction; no cohort or locked-feature reads |
| [c0_metrics.py](../../../src/phase2/c0_metrics.py) | Fixed class-reference scores with leave-one-out training rules; training-only LR-z/LR-angle probes; balanced pair risks, spectra, state distances, observable contrasts, calibration and response derivatives |
| [run_c0.py](../../../src/phase2/run_c0.py) | Synthetic smoke by default; explicit separate full-CV flag; private non-overwriting outputs; checkpoints, stage logs, metrics, paired Q−E contrasts and hierarchical fold summaries |
| [c0_v1.yaml](../../../configs/c0_v1.yaml) | Reviewable closed scientific settings; no parameter grid |
| [test_c0.py](../../../tests/phase2/test_c0.py) | Synthetic objective, gradient, optimizer, integration, isolation, provenance and reproducibility checks |

The original B encoder, four-qubit RY map, two-layer RY/RZ+CNOT ansatz and Z(3) output are imported without editing `models.py`. The angular map is applied exactly once. Frozen blocks are excluded from optimizers, including AdamW decay. The two-stage encoder is cached after geometric training and remains bitwise fixed during head fitting. The pristine initial head receives no stage-1 update or decay. Joint head loss gradients come from BCE only, while the encoder receives BCE plus the fixed auxiliary objective.

All four new arms share initial tensors, optimization pair streams and the BCE patient-order stream within fold/seed. Diagnostic sampling uses its own stream and does not mutate parameters, accumulated gradients or optimization RNG state. Validation is observational; its labels do not enter pair sampling, optimizer steps, prototypes, probe fitting or checkpoint selection.

The runner writes detailed patient-level data only beneath the ignored `runs/` directory, rejects existing destinations, and does not generate or overwrite a scientific narrative report. Full CV is **not** the default CLI action and was not invoked. Failure statuses are retained rather than replaced by favorable seeds or altered settings. Probe convergence warnings are recorded instead of triggering a budget increase.

## 3. Tests actually executed

### Final dedicated suite: 34 passed in 4.01 s

```bash
MPLCONFIGDIR=/tmp/qarl-c0-mpl XDG_CACHE_HOME=/tmp/qarl-c0-cache .venv/bin/python -m pytest -q tests/phase2/test_c0.py tests/phase2/test_ry_fidelity_identity.py
```

Coverage includes:

- Rejection of changed lambda, epochs, pair counts, bandwidth, seeds and checkpoint selection; YAML/code agreement.
- Exact 16/16/32 stratum counts, self-pair exclusion, deterministic streams and minimum class membership.
- Numerical equality of Q/E pair residuals to independent matrix calculations, signed versus 0/1 labels and gradients through both endpoints.
- Q/E angular gradcheck; absence of auxiliary head gradients; equality of joint-head and BCE-head gradients.
- Gaussian local curvature matching and independent RY statevector overlap identity.
- Two-stage frozen-head/frozen-encoder hashes, optimizer membership, synthetic step/pair/state counts and fixed scientific-versus-technical budget labels.
- Repeated synthetic fitting giving identical tensors, shared Q/E pair and patient streams, and unchanged tensors when only validation labels change.
- Training-only probe fitting, leave-one-out references, trace-distance versus observable-contrast separation and response chain rule.
- A lambda-zero **synthetic update compatibility test** against the original optimizer/update routine, without fitting historical B data or changing the registered lambda.
- Synthetic canonical-dimension availability fixtures, successful loader operation without any expression parquet, and rejection of altered cache fingerprints, reordered IDs, locked overlap, corrupted initial checkpoints and a changed protocol hash.
- Technical smoke routing that fails if the patient-input loader is called, rejection of shortened real-data budgets and unsafe output locations, a frozen-architecture source-hash gate before fitting, and correct Q−E contrast direction without diagnostic-readout selection.
- The two existing synthetic fidelity/terminal-unitary identity tests, including numerical derivatives and nontrivial local observable response.

### Original architecture/initialization regressions: 6 passed in 1.53 s

```bash
MPLCONFIGDIR=/tmp/qarl-c0-mpl XDG_CACHE_HOME=/tmp/qarl-c0-cache .venv/bin/python -m pytest -q tests/phase2/test_experiments.py::test_paired_quantum_initialization tests/phase2/test_experiments.py::test_quantum_shapes_and_gradients tests/phase2/test_experiments.py::test_angles
```

These are synthetic regressions, not reruns of the historical experiments. The installed environment is Python 3.12.11, Torch 2.9.0, PennyLane 0.43.0, NumPy 2.5.3, pandas 2.3.3, scikit-learn 1.9.1 and PyArrow 21.0.0; no dependencies were changed. Together, **40 targeted cases passed** in the final dedicated and regression runs. The complete repository suite was not executed, and these tests do not establish scientific performance or absence of every possible defect. Python compilation and `git diff --check` also passed.

An earlier 23-case suite passed. After extending provenance/isolation coverage, one repeat encountered a test-only destination-name collision: **29 passed, one failed** because the non-overwrite guard correctly rejected the existing output folder. The test destination was made unique per pytest session; no old output was removed, no scientific setting changed, and the expanded 30-case suite then passed before the final 34-case suite. This failure was in test-output naming, not an optimization failure or a model result used for tuning.

## 4. Technical smoke experiments actually executed

A direct synthetic smoke command completed all four arms:

```bash
MPLCONFIGDIR=/tmp/qarl-c0-mpl XDG_CACHE_HOME=/tmp/qarl-c0-cache .venv/bin/python -m src.phase2.run_c0 --output runs/phase2/c0_v1_synthetic_smoke_technical
```

This used invented Gaussian PCA32 inputs: 64 training rows and 16 validation rows, with synthetic labels derived from their first coordinate. It used seed 1729 and technical fold identifier 0, not TCGA fold-0 patients. Each stage ran **two technical epochs**, retaining the separately recorded scientific budget of 50 epochs. Each arm received 256 pairs and 128 BCE patient occurrences; NQE arms had eight updates (four geometry, four head), and hybrids had four joint updates. No technical metric was used to choose lambda, bandwidth, architecture, seeds, epochs or the scientific protocol.

The end-to-end synthetic tests also execute all four short arms in separate private destinations, and unit tests perform additional tiny synthetic updates for reproducibility and gradient checks. These are technical executions, not additional development-CV seeds, pilots or confirmatory results. Their individual outputs remain local; synthetic predictive scores are not interpreted as research evidence.

The direct smoke's measured process wall time was **0.8991 s**, with one Torch thread, CPU, and process-local peak RSS **482,263,040 bytes** (approximately 460 MiB). This includes imported-runtime memory in the process high-water mark and does not forecast full-run memory. The direct smoke predates small bookkeeping additions validated by the final integration suite; source snapshots/hashes distinguish those executions. There were no failed arms in the direct or successful integration smokes.

**Not executed:** any real-patient C0 fitting, the 60 full pipelines, the 150 planned real-data probes, a development pilot, a locked-test evaluation, or any A/B/D0/E0 rerun. The only real development-data computation was the authorized availability/reconstruction audit and a no-fit audited-loader dry run.

## 5. Resources and prospective full-run budget

Full CV remains pending separate authorization. It would execute serially on CPU, one Torch thread, float64, `default.qubit`, backpropagation and `shots=None`. No GPU or quantum hardware is requested. The model has 300 encoder parameters and 16 nominal VQC parameters; the redundant final RZ parameters remain present. The prepared caches avoid repeating RNA preprocessing during model fitting. Conservatively allow about 8 GiB available host memory for execution headroom, informed by the earlier approximately 4.25 GiB historical process peak; this is a planning allowance, not a measured new CV requirement or a configured memory reservation.

| Future item | Registered amount |
|---|---:|
| Fold/seed input groups | 5 × 3 = 15 |
| Primary pipelines | 4 × 15 = 60 |
| Optimizer stages | 90 |
| Sequential updates | 94,500 |
| Sampled optimization pairs | 4,032,000 |
| Requested geometric endpoint occurrences | 8,064,000 |
| BCE training patient-state evaluations | 2,004,000 |
| VQC calls including per-epoch validation and final train/validation scoring, excluding diagnostics | 66,120 |
| Corresponding analytic patient-state evaluations | 2,555,100 |
| LR probes | 120 final-encoder fits + 30 common-initial-encoder fits = 150 |

Additional initial/final geometry, response derivatives, frozen-v₀ scoring, gradient-component diagnostics, reference kernels, serialization and data-loading costs are recorded separately. The fixed initial probes are shared across arms, and two-stage identical encoder endpoints are reused. Counts are simulator operations, not hardware shots. The two-stage arms have twice as many sequential update events as hybrids/B, while each active parameter block receives the registered 1,050 updates. No equal-total-compute comparison with historical B is claimed.

The protocol's honest wall-time planning expression remains **195.5 s + 30 T_G + 30 ΔT_A + T_D + T_input**, where the first term uses historical B/head-loop timings and the remaining terms are unmeasured geometry-stage, auxiliary, diagnostic and input costs. The 0.8991 s synthetic smoke is not substituted into that expression. Full-run wall time and memory have not been measured, and no speedup forecast is asserted.

## 6. Acceptance status, limitations and execution gate

The availability prerequisite, synthetic implementation checks, source preservation and configuration checks have passed. The four scientific arms and their objectives remain intact. The implementation is prepared for a **separate full-execution review**, and no complete-CV launch occurred in this task.

The evidence is technical: synthetic tests cannot predict generalization, prove optimizer convergence, certify all numerical behavior over 50 real-data epochs or establish that one metric geometry is preferable. New-real-encoder initialization sensitivity beyond h=s remains outside the minimum design. The main Q/E comparison controls schedules and pair weighting; it does not make their entire gradient landscapes identical. No gap in synthetic coverage authorizes post-hoc changes when real results arrive.

Use new destinations for any later approved run. The full-run route requires the explicit `--execute-cv` flag and revalidates the audit, protocol hash, source inputs, configuration and dependency versions before fitting. A failed gate stops rather than falling back to new preprocessing, altered inputs or unregistered settings. The full execution command was deliberately not invoked here.
