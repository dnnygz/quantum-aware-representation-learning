# C0-v1: Supervised geometric representation learning with the fixed B classifier

**Status: proposed experimental protocol for review, not an approved implementation or executed experiment.** Prepared 2026-10-28 (America/Lima). This document fixes a recommended minimum design and explains consequential alternatives. No C0 training, C0 objective implementation, historical experiment rerun, or locked-test evaluation has occurred. The numerical work accompanying this document verifies mathematical identities using synthetic inputs only. C1 is outside scope.

## 1. Research question, estimands, and historical motivation

Does increasing supervised separability of the quantum states prepared by the existing RY embedding yield better prediction and generalization with the restricted B VQC? Distinguish three estimands:

1. **Geometry transfer:** does training-fold geometric supervision improve the corresponding held-out geometric summaries?
2. **Restricted-readout transfer:** does the resulting representation improve the fixed-architecture VQC's validation log loss and ROC-AUC?
3. **Geometry specificity:** under the same learning schedule, does RY fidelity supervision differ from a classical Euclidean Gaussian similarity with the same labels, encoder, angular coordinates, pair distribution and optimization budget?

These are algorithmic, development-only questions. All five development folds have already informed this research. Pre-specification limits further flexibility but does not make these folds a new independent confirmatory dataset. An improvement cannot establish quantum computational advantage: RY fidelity is exactly classically computable.

Historical evidence is read from completed aggregate reports, not re-executed:

| Historical source | Result relevant to C0 | Implication and limit |
|---|---|---|
| [A/B development CV](cv_5fold_50e_v1/report.md) | Validation ROC-AUC: A 0.8297, B 0.9215, PCA4-LR 0.9490, B frozen LR-z probe 0.9371. B training/validation log loss: 0.0488/0.2867 nats; B_classical: 0.0022/0.7585. | B improves over A, but existing classical readouts are competitive and encoder training specializes strongly. This does not isolate a quantum contribution. |
| Same CV | B numerical latent rank <4 in 9/15 solutions; B_classical has lower variance entropy rank while retaining numerical rank four. | Neither maximizing rank nor treating contraction as uniquely quantum is justified. |
| [D0](d0_v1_verified/report.md) | Ten nearly constant B coordinates have map slopes 99.42–100% of their maximum, alongside weak circuit responses. | Angular saturation alone is not a general explanation. Variation, sensitivity and predictive utility must remain separate. |
| [E0](e0_v1/report.md) | VQC refits minus B: mean training/validation log loss −0.000220/−0.000382 nats. LR-angle minus LR-z: validation AUC −0.000068, log loss +0.002029 nats. | Simple head restarting and angular linear readout changes did not yield a large general remedy in that protocol; this is not equivalence or global optimality. |

Historical means summarize folds after averaging seeds, not independent patients or 15 independent experiments. Historical B is the fixed reference, not a newly selected comparator. D0 and E0 are closed. Their outputs will be reused only through read-only references or already saved representations; no A/B/D0/E0 training or analysis runner is to be rerun.

## 2. Fixed scientific substrate and data isolation

The cohort has **835 development patients: 645 ER+, 190 ER−**, with **209 locked-test patients** excluded. The five canonical development folds each have **668 training patients (516 ER+, 152 ER−)** and **167 validation patients (129 ER+, 38 ER−)**. Retain fold identifiers 0–4 and seeds **1729, 1730, 1731**, with no fold, seed, or failed-run exclusions.

Inputs are the same fold-local PCA32 representation used by B. Preserve training-only CPM≥1 in ≥10% of training samples, frozen-reference TMM (logratio trim 0.3, sum trim 0.05, weighted), log2-CPM with prior count 2, exact-zero training-variance filtering, gene-wise scaling, randomized PCA32 with random_state=1729, and PCA-score standardization. Validation is transformed with training-fitted objects without refitting. No additional whitening, feature scaling, coordinate deletion, batch normalization, encoder widening or input feature selection is introduced.

Historical run roots are `runs/phase2/pilot_fold0_50e_paired_v1/` for fold 0 and `runs/phase2/cv_folds1to4_50e_paired_v1/` for folds 1–4. A later implementation should reuse verified fitted B preprocessing objects and verified saved PCA32 inputs if available. Saved four-dimensional final B embeddings cannot serve as inputs to a new encoder-learning experiment. If PCA32 arrays were not saved, transform development-only rows with the trusted, training-fitted historical preprocessing object and verify reconstruction against historical initial encoder embeddings. Do not retrain B or refit preprocessing to create them. Missing or unverifiable preprocessing objects block implementation readiness rather than justify silently changing preprocessing. Trusted pickle loading requires validated local provenance.

Training pairs, diagnostic training batches, class reference sets, probes, optimizer updates and any class statistics used for prediction must come entirely from that fold's training partition. Validation labels are visible only to the evaluator, which returns logged metrics after a checkpoint is fixed. They cannot affect sampling, lambda, bandwidth, learning rates, stopping, checkpoints or method choice. Locked features/labels must never be materialized; split metadata may only enforce disjointness. Store patient-level outputs only under ignored `runs/` and retain existing data-governance restrictions.

### Immutable architecture

Let x denote a standardized PCA32 vector. The encoder g_w is Linear(32,8) → Tanh → Linear(8,4), with 300 parameters including biases. Define z=g_w(x) and a_j=π tanh(z_j/2). State preparation is

\[
|\phi_w(x)\rangle=\bigotimes_{j=0}^{3}R_Y(a_j)|0\rangle.
\]

The existing VQC V_v has two layers of four RY/RZ rotations, each followed by CNOT 0→1→2→3; there are 16 nominal head parameters. Retain the structurally redundant four final-layer RZ parameters. The output is Pauli-Z on qubit 3, with

\[
p_{w,v}(x)=\operatorname{clip}\left[\frac{1+\langle\phi_w(x)|V_v^\dagger Z_3V_v|\phi_w(x)\rangle}{2},10^{-7},1-10^{-7}\right].
\]

CPU float64, one Torch thread, deterministic Torch operations, `default.qubit`, `shots=None` and backpropagation remain fixed. No new measurement, trainable calibration, input re-uploading, data-dependent RZ, ansatz search, qubit count, angular map or prediction threshold is introduced. Auxiliary classical probes below are diagnostic controls, not replacements for this primary prediction pipeline.

## 3. Exact objectives and classical metric control

Data labels remain y=1 for ER+ and y=0 for ER−. Signed labels t=2y−1 are used only to express equivalence to the literature. For a pair i≠j define

\[
s_{ij}=\mathbb{1}(y_i=y_j)=(1+t_i t_j)/2.
\]

BCE is the **unweighted natural-prevalence** mean in nats on a patient minibatch I:

\[
L_{\mathrm{BCE}}(w,v;I)=-|I|^{-1}\sum_{i\in I}[y_i\log p_i+(1-y_i)\log(1-p_i)].
\]

The RY similarity and pair loss are

\[
k_Q(i,j)=\prod_{r=0}^{3}\cos^2[(a_{ir}-a_{jr})/2],\qquad
\ell_Q(i,j)=[k_Q(i,j)-s_{ij}]^2.
\]

This reproduces **Hur et al.'s Eq. (3) objective formula** on a different, explicitly restricted feature map. It does not reproduce their ZZ circuit, neural architectures, pair distribution, optimizer, hardware experiment or QCNN. Neither Lloyd's Hilbert–Schmidt cost nor Sharma's curriculum is substituted for this objective.

### Recommended classical control: Euclidean Gaussian pair regression

Use the same four bounded angles a, with no additional learned transformation, and define

\[
k_E(i,j)=\exp[-\|a_i-a_j\|_2^2/4],\qquad
\ell_E(i,j)=[k_E(i,j)-s_{ij}]^2.
\]

The fixed Gaussian bandwidth, in the convention exp(−d²/(2σ²)), is σ=√2 radians. Its scale is fixed analytically, not estimated from data: log k_Q=−||Δa||²/4+O(Σ_r Δa_r⁴) for small Δa, so k_E matches local second-order curvature while removing the product of periodic overlaps. Both similarities range in [0,1], share identical-pair value one and the same target/loss range, and introduce no learned parameters. This is a classical Euclidean metric-learning control specified by this protocol, not an exact reproduction of another paper.

This control changes the similarity geometry while retaining bounded coordinates, architecture and supervision. It does not perfectly match global gradient scale or all inductive biases. Gaussian similarity approaches zero with distance rather than attaining exact orthogonality at a finite displacement. RY fidelity identifies some widely separated angular coordinates through periodic state equivalences; Gaussian distance on the unwrapped angular chart does not. Those are intended differences, not proof that a resulting effect is uniquely quantum. The two kernels also share local behavior, making a null contrast informative but not proof that all classical metric learners are equivalent.

A raw-z triplet/contrastive-margin control would add margin, latent-scale and potentially normalization choices. A data-tuned RBF bandwidth would introduce selection flexibility. **Recommendation: use the fixed angle-space Gaussian in both schedules below**, avoiding either complication in C0-v1. An ordinary linear probe alone is not a control for the effect of geometric supervision during encoder learning.

## 4. Pair sampling, class balance and diagonals

Define a class-balanced pair population risk, for R∈{Q,E},

\[
L_R=\tfrac14\mathbb E_{++ ,\ i\ne j}\ell_R+
\tfrac14\mathbb E_{-- ,\ i\ne j}\ell_R+
\tfrac12\mathbb E_{+-}\ell_R.
\]

At every metric update sample **64 pairs: 16 ER+/ER+, 16 ER−/ER−, and 32 ER+/ER−**. Within a same-class draw, choose an ordered first patient uniformly and a second uniformly among the other patients in that class; evaluate the symmetric loss once. Cross-class draws choose each endpoint uniformly from its corresponding class. Draw pairs independently with replacement across draws/updates; repeated pairs and repeated patients across different pairs are allowed. Both endpoints receive gradients. Do not treat reverse orientations as independent observations.

The estimator is the simple mean of the 64 losses; with these stratum counts it is unbiased for the displayed balanced risk. It balances same/different targets 1:1 and endpoint class contributions, while intentionally oversampling the minority relative to natural patient prevalence. It is **not** the empirical all-pairs population risk of this imbalanced cohort. This choice is identical for Q and E, so their contrast holds supervision balance fixed. BCE minibatches retain original prevalence; no class-weighted BCE or balanced patient resampling is added. Consequently comparisons to historical B change supervision weighting as well as adding geometry, and cannot isolate geometry from class reweighting alone. A separate reweighted-BCE experiment would require a later protocol, not a post-hoc addition.

Exclude **all diagonal pairs i=j** from optimization and pair-regression diagnostics. Their target and kernel are both one, yielding zero loss/gradient and diluting a finite pair budget. Exclude self references when evaluating class-reference scores on the training split. Validation queries have no identity overlap with training references. Density-matrix purities, however, include diagonal contributions by definition; label them explicitly and do not call the diagonal-excluded estimator an exact finite-sample purity.

Each geometry epoch is **21 fixed updates**, matching ceil(668/32), rather than one complete pass over all possible pairs or a guarantee that every patient appears. Fifty geometry epochs yield **1,050 updates, 67,200 sampled pairs and 134,400 endpoint occurrences**. The patient BCE stream has 20 batches of 32 and one batch of 28 per epoch, with all 668 training patients seen once. Pair generation uses the full fold-training pool, independently of the current patient batch; this prevents a small minibatch from omitting a required class stratum.

## 5. Four recommended arms and fixed parameter updates

There are two scientific schedules and one matched classical metric control instantiated under each schedule. All are new C0 procedures; the historical B reference is reused without rerunning it.

| Arm | First 50 epochs / 1,050 updates | Next 50 epochs / 1,050 updates | Trainable parameters | Literature status |
|---|---|---|---|---|
| **C0-NQE-Q** | Minimize L_Q; encoder w only; VQC held at its initial tensor v₀ | Freeze w; minimize BCE over VQC v only, starting at v₀ | 300 then 16 | Hur pair-loss formula, RY/two-stage frozen-encoder adaptation |
| **C0-NQE-E** | Same, replacing L_Q by L_E | Same frozen-encoder VQC fit | 300 then 16 | Matched classical geometric-supervision control |
| **C0-Hybrid-Q** | Jointly minimize BCE + λL_Q over w,v for 50 epochs | None | 316 | Own hybrid adaptation using a published pair objective; not exact NQE |
| **C0-Hybrid-E** | Same, replacing L_Q by L_E | None | 316 | Matched classical hybrid control |
| **Historical B** | Existing joint BCE, 50 epochs | None | Existing result only | Immutable reference, zero new fits |

In the two-stage arms, the encoder is frozen during stage 2; no fine-tuning is allowed. This directly tests whether geometry learned without VQC feedback is usable by the restricted head. Fine-tuning would answer a different question and blur stage attribution. In hybrids,

\[
\nabla_wL=\nabla_wL_{\rm BCE}+\lambda\nabla_wL_R,\qquad
\nabla_vL=\nabla_vL_{\rm BCE},
\]

because full-state fidelity does not depend on the terminal unitary. Gaussian similarity also depends only on w. AdamW weight decay is a decoupled optimizer operation, not an additional term in these displayed loss functions. Frozen modules must be excluded from the optimizer entirely, not merely assigned a zero loss gradient; otherwise weight decay can move them. Stage-1 VQC weights must remain bitwise unchanged, and stage-2 encoder weights and resulting embeddings must remain bitwise unchanged.

### Lambda and relative loss scale

**Recommend λ=1 as the only hybrid coefficient.** Both losses are means; neither uses an unnormalized sum. Pair loss is dimensionless and bounded [0,1]; BCE is in nats and unbounded before clipping. λ=1 is a transparent unit-weight intervention, not equality of loss magnitudes, gradient magnitudes, or effective influence on encoder learning. Do not rescale BCE by log 2, divide by current losses, normalize gradients, anneal lambda, apply automatic task weighting, or change lambda by fold/seed.

No lambda grid, validation choice or favorable-seed selection is included. Historical B supplies an existing λ=0 reference but is not a contemporaneous timing control. At epochs 0, 25 and 50, record each loss, encoder gradient L2 norms g_B and g_R, λ||g_R||/(||g_B||+10⁻¹²), and the encoder gradient cosine when both norms exceed 10⁻¹². Use one fixed training-only diagnostic patient batch of 32 and fixed 64-pair list, with separate diagnostic RNG; evaluate at the current checkpoint without updates or optimizer-state changes. A zero denominator/cosine case is marked undefined, not replaced by a favorable number. These diagnostics explain the intervention's realized strength but cannot trigger coefficient changes. If one term dominates or is negligible, report that limitation; any alternative scale becomes C0-v2 before further validation use.

## 6. Initialization, random streams, optimization and checkpoints

Reuse the **initial**, not final, historical B encoder and head tensors for each (fold,s). Validate their hashes and reproduce their generation using the existing CPU Model initialization/RNG-fork policy. Do not warm-start any arm from a trained B encoder or head. Across all four arms in a given (fold,s), w₀ and v₀ are identical. The two-stage pair-trained encoders Q/E see identical pair-index sequences; the two hybrids use those same 50-epoch pair sequences. Identical seeds do not imply identical gradients after objectives differ.

Use the original patient shuffle stream: seed Torch with s, instantiate the unchanged B model, capture its post-construction RNG state, then clone that state into a dedicated patient-order generator before any new pair/diagnostic draws. Verify compatibility with saved historical shuffle hashes if available; otherwise verify reproduction from the pinned source/environment and record generated order hashes. Reset this generator to the same state at the start of each BCE stage, so every new arm receives identical 50 patient permutations. Do not continue the patient shuffle stream through geometric pretraining.

Use an independent NumPy PCG64 generator initialized by SeedSequence([s,f,1]) for optimization pair sampling; diagnostic batches/pairs use SeedSequence([s,f,2]). The numerical values 1 and 2 are fixed stream roles, not additional experimental seeds. Preserve patient canonical ordering before indexing. No sampling from validation is allowed. Diagnostic calculations must neither consume the optimization generators nor change global RNG state.

Every trainable stage uses AdamW, lr=0.01, weight_decay=0.001, batch size 32 for BCE or the explicit 64 pairs for geometry, no scheduler, clipping, early stopping or budget extension. Freeze complete resolved optimizer defaults from the pinned environment (betas=(0.9,0.999), eps=10⁻⁸, amsgrad=False, maximize=False; record device/backend defaults). Instantiate fresh optimizer state at the start of each stage. Stage-2 heads start from pristine v₀ with a fresh optimizer; they have not received stage-1 decay or gradient updates. Retain all nominal head weights despite known redundant RZ gradients.

Only the fixed last checkpoint is evaluated as an arm's result: stage 2 epoch 50 for two-stage methods, epoch 50 for hybrids. Save initialization, geometric-stage endpoint, and final checkpoint; in two-stage methods the geometric-stage and final encoder hashes must agree. Epoch curves may be logged but never choose a validation-minimum epoch. Retain every failure, warning and seed. Technical interruption may resume an identical deterministic state with provenance; numerical optimization failures cannot be replaced by alternative restarts, lambda or budgets.

**Minimal head-initialization recommendation:** one head initialization h=s per learned encoder, paired across methods, not a 3×3 encoder/head grid. E0 already tested the crossed starts on historical B embeddings with small effects, supporting this economical first design. This does not prove new C0 representations are initialization-insensitive; seed variation here mixes encoder and head effects. A later full crossed-head study would be separate. No favorable head selection is allowed.

## 7. Controls, frozen readouts and comparability

For every final C0 encoder (60 total), fit exactly two training-only logistic probes, with the encoder fixed: LR-z on z and LR-angle on a. Use C=1, max_iter=2000, L-BFGS, tol=10⁻⁴, intercept=True, no class weighting, no warm start, effective L2/default l1_ratio=0, random_state=s; pin all other estimator settings to historical E0. No extra feature standardization or C search. These are 120 deterministic probe fits, not 360 replicated fits. Historical B LR-z/LR-angle already exist and are reused without refitting. The same C does not imply identical effective regularization across coordinate maps.

At each of the 15 common initial encoders, fit the same two probes once and share these initial-readability references across all four arms: 30 additional fits. This is a **frozen random-initial encoder capacity reference**, not a new encoder architecture or an estimate of maximal encoder capacity. Keep parameter count 300 and bottleneck four everywhere; no width/depth sweeps or rank-preservation penalties. Historical B_classical remains a separate co-trained classical-head reference, not a coordinate-aligned encoder.

Use three further read-only controls without fitting new parameters:

- **Frozen initial VQC:** evaluate the same v₀ on the initial encoder and on each final learned encoder; for two-stage arms also record v₀ at the geometry endpoint (the same encoder as final). This shows how encoder motion changes one fixed observable. Compare v₀ and v_final on the exact same final encoder. Such differences are conditional prediction effects, not a causal decomposition of encoder learning.
- **Fidelity class-reference readout:** for a frozen encoder and training class c, define μ_c^Q(x)=n_c⁻¹Σ_{j:y_j=c}k_Q(x,j). Score r_Q=μ_+^Q−μ_-^Q, threshold r_Q≥0, and if probability scores are desired use the explicitly uncalibrated p_Q=(1+r_Q)/2, clipped by the same fixed ε. Use leave-one-out own-class means for training queries (n_c−1 denominator). For validation, all references come from training. This is Lloyd-style class-overlap scoring, not the trained B observable or a claim of probability calibration.
- **Euclidean class-reference readout:** repeat the identical class-reference rule using k_E on **every** encoder, including Q-trained ones. Conversely evaluate the Q readout on E-trained encoders. This crosses training geometry with readout geometry rather than only giving each arm its favored readout. No learned bandwidth, prototype, kernel coefficient or calibration is introduced.

Compute the class-reference controls at the common initial encoder and the final encoders; obtain historical B's corresponding descriptive scores only from verified saved final development embeddings if needed. This is a newly specified frozen diagnostic on old artifacts, not a rerun of A/B/D0/E0 or a new model fit. The primary pipeline remains its unchanged VQC. Conditional probe timings exclude representation learning; whole-pipeline comparisons must add it back.

### Comparability matrix and allowed interpretation

| Contrast | Held fixed | Changes | Supported interpretation / remaining confounding |
|---|---|---|---|
| NQE-Q vs NQE-E | Architecture, initialization, pair strata, steps, stage freeze, VQC budget/readout | Similarity formula | Effect of the specified geometry/its gradient landscape under two-stage learning; not unique quantum physics |
| Hybrid-Q vs Hybrid-E | Same, including λ=1 and patient orders | Similarity formula | Geometry-specific auxiliary intervention at fixed coefficient, not globally matched gradient strengths |
| Each hybrid vs historical B | Initial tensors, architecture, intended patient stream, BCE, 1,050 joint steps | Auxiliary supervision, pair exposure/weighting, extra computation and execution context | Added-supervision procedure versus historical reference; does not isolate class reweighting or computational cost |
| Each two-stage arm vs B | Initial tensors, total active updates per parameter block, 50 BCE head epochs | Encoder objective, scheduling/freeze, data exposure and total steps | Whole-procedure comparison, not equal total compute or pure loss substitution |
| NQE-Q vs Hybrid-Q | Geometry and architecture | Stage sequence, frozen/joint encoder, BCE exposure to encoder, total steps | Schedule/procedure contrast, not a single-factor causal effect |
| VQC vs LR-z/LR-angle on one encoder | Learned representation, patients | Readout hypothesis class, objective/optimizer, regularization and fit budget | Readability/compatibility; probe success alone cannot prove VQC irrelevance during hybrid training |
| Trained vs v₀ VQC on one final encoder | Final encoder and architecture | Head tensor | Conditional attainable-head effect, not contribution to encoder learning |
| Q vs E class-reference scores on one encoder | Training references, labels, no fitting | Similarity at inference | Whether geometry transfers to its own and an alternative fixed readout |
| Initial vs learned-encoder probes | Architecture and probe specifications | Learned representation | Training-induced readability versus a random-feature reference, not maximal functional capacity |

Do not claim that the geometric term trains the VQC, that entanglement caused separation, or that a geometric gain necessarily improves the restricted observable. In two-stage learning the VQC cannot have caused the stage-1 geometry: it was absent from optimization. In hybrids its BCE gradient may influence w, even though the auxiliary gradient to v is zero. These are distinct statements.

## 8. Evaluation endpoints, generalization and geometry

### Classification and paired summaries

The two co-primary predictive endpoints are **validation log loss (nats; lower is better)** and **ROC-AUC for ER+ (higher is better)**. Report both without choosing the favorable one. For each schedule, the prespecified primary method contrast is Q minus its E control; Q minus historical B is a separate contextual contrast. Retain training endpoints at the **same final checkpoint** as validation, not online minibatch averages. Report Δ validation loss, Δ training loss, and the validation−training loss gap separately: a smaller gap caused by worse training fit is not automatically better generalization.

Secondary endpoints: AP ER− using 1−p and ER− as positive target; AP ER+; balanced accuracy at p≥0.5; sensitivity for ER− and ER+; specificity with its positive-class convention stated; accuracy; confusion matrices; Brier mean((p−y)²); and fixed 10 equal-width probability-bin counts, mean probabilities and event rates. Neither Brier nor log loss alone isolates calibration. Do not fit a calibrator, optimize thresholds, choose kernels/readouts by validation, or select the best seed. AP depends on class prevalence, which remains fixed across methods within each split.

Compute paired effects within (fold,s), then average the three seeds within each fold, then the five fold means. Show all 15 paired cells, five fold effects, their descriptive sample SD, within-fold seed SD, ranges and all failures. Fold training sets overlap and seeds reuse patients: no naive standard errors divided by √15, independent-seed hypothesis tests, pair-count pseudo-replication, confirmatory p-values or discovery-by-recurrence rule. Counts of effect directions are descriptive only. OOF predictions for a given seed cover each development patient once, but seeds must not be stacked into 2,505 independent patients. Any OOF metric is secondary and explicitly distinct from mean fold metrics. No independent inferential confidence interval is claimed for this reused-development design.

### Prespecified geometric measurements

Use held-out geometry as the primary geometry-transfer check, not only training loss. Evaluate both Q and E diagnostics on every learned encoder, regardless of which geometry trained it. At initialization, geometry-stage endpoint (two-stage arms), and final checkpoint record:

1. The exact balanced diagonal-excluded pair risk L_Q and L_E on **all unordered within-split pairs** with the same 1/4,1/4,1/2 stratum weighting, together with each stratum's mean squared residual. Training has 222,778 unordered pairs; validation has 13,861. No stochastic validation pair selection is used.
2. Means, SDs and 5th/50th/95th percentiles of similarities in ++, -- and +- strata; balanced separation G_R=½(mean k_R,++ + mean k_R,--)−mean k_R,+-. G is an interpretable summary, not a substitute objective or a sufficient statistic for useful classification. Also record fractions of pairs k_Q<0.01 or >0.99, retaining class strata, to distinguish useful separation from uniform concentration/collapse.
3. Define the empirical class states ρ_c=n_c⁻¹Σ_{i:y_i=c}|φ_i⟩⟨φ_i| separately for each split. Report class purities tr(ρ_c²), tr(ρ_+ρ_-), **D_HS=tr[(ρ_+−ρ_-)²] without a square root**, and balanced trace distance D_tr=½||ρ_+−ρ_-||₁. These finite-ensemble definitions include diagonal contributions to purity. This uses Lloyd's geometry as a **diagnostic**, not an additional optimization objective. State explicitly whether the natural-prior weighted discrimination bound or balanced ensembles are being discussed; C0-v1 uses balanced ensembles here.
4. Quantify accessibility to the actual trained head: J_v=mean_+(p_v)−mean_-(p_v). For the unclipped link J_v=½tr[V_v†Z_3V_v(ρ_+−ρ_-)], so |J_v|≤D_tr. Use **unclipped** probabilities to compute J_v for this operator identity and the accessibility ratio; retain the clipped class-mean contrast separately for consistency with reported predictions. Pointwise clipping is nonlinear, so do not assert that its ensemble contrast obeys exactly the same operator identity. Its contrast differs from the raw contrast by at most 2×10⁻⁷. Record the signed raw J_v and |J_v|/D_tr when D_tr>10⁻¹², otherwise mark the ratio undefined. D_tr bounds an unrestricted discrimination problem, whereas the given ansatz may not attain it. J_v is a class-mean response descriptor, not accuracy or a full accessibility characterization.
5. Centered singular values of z and a, numerical rank with s_j>10⁻⁶s_max, variance entropy rank exp(−Σu_j log u_j), and participation ratio 1/Σu_j² for u_j=s_j²/Σs². If all singular values vanish, mark the effective-rank measures undefined and report the zero spectrum. Report coordinate SDs and class-mean contrasts; do not assume learned coordinates align across independent runs.
6. Angular saturation fractions |tanh(z/2)|>0.95 and >0.99, and map derivatives da/dz=(π/2)sech²(z/2). Distinguish these from the circuit expectation derivatives dm/da and probability derivatives dp/dz. Summarize each factor independently and verify the pointwise chain rule; do not multiply averaged factors. Initial/final derivatives on all 668/167 points require only read-only differentiation of the fixed circuit, no fitting. Label these as responses, not useful-information measures.

All training/validation geometry computed with validation labels is descriptive evaluation only. Predictions always use training-fitted quantities. Full geometry is computed only at the prespecified snapshots; two-stage endpoint/final encoder diagnostics can be cached because the encoder is identical. Fixed diagnostic thresholds above are measurement definitions, not empirical discovery cutoffs or tuning rules. No additional D0 interventions, coordinate scrambling, exhaustive plots or causal rank claims are introduced.

### Stability

Record online training objectives separately from end-of-epoch/snapshot values; finite losses, gradients and parameters; encoder/head gradient norms; per-parameter head-gradient RMS including redundant RZ gates; lambda-component gradient norms/cosines; optimizer updates; convergence warnings for probes; and fold/seed variability. Nonfinite outcomes are retained as failures with their denominators; incomplete paired contrasts remain incomplete rather than silently dropping the cell. A successful numerical gradient check does not establish good optimization or barren-plateau mitigation.

## 9. Mathematical fidelity verification and attribution limits

For RY(a)|0⟩=cos(a/2)|0⟩+sin(a/2)|1⟩,

\[
\langle R_Y(a)0|R_Y(b)0\rangle=\cos[(a-b)/2].
\]

Tensor products multiply overlaps; absolute square gives exactly k_Q in Section 3. For a shared data-independent unitary V_v,

\[
\left|\langle V_v\phi_i|V_v\phi_j\rangle\right|^2
=\left|\langle\phi_i|V_v^\dagger V_v|\phi_j\rangle\right|^2
=k_Q(i,j),\qquad \nabla_v k_Q=0.
\]

With Δ_r=a_ir−a_jr and f_r=cos²(Δ_r/2),

\[
\frac{\partial k_Q}{\partial a_{ir}}
=-\tfrac12\sin(\Delta_r)\prod_{q\ne r}f_q,\qquad
\frac{\partial a_{ir}}{\partial z_{ir}}=\tfrac\pi2\operatorname{sech}^2(z_{ir}/2).
\]

Gradients to the other endpoint have the opposite angular sign; encoder gradients sum both endpoint paths. Do not divide by f_r to compute this derivative near orthogonal factors. A zero factor can annihilate other-coordinate derivatives, independently of angular saturation. The classical comparator obeys ∂k_E/∂a_ir=−½Δ_r k_E. Its local match does not match all derivatives globally.

The density matrices after the shared head satisfy ρ'_c=V_vρ_cV_v†. Both trace distance and Hilbert–Schmidt distance, as well as pure-state fidelities, are unitarily invariant. Therefore any change in these quantities between stage 1 and stage 2 of a frozen-encoder arm would indicate an implementation/measurement problem, not successful head training. **Local expectation readout and BCE are not invariant in this way**: varying V_v changes which observable interrogates a fixed representation. Two orthogonal states are perfectly distinguishable by some measurement; the restricted 16-parameter ansatz is not guaranteed to implement that measurement. This mismatch is central to the experiment.

### Numerical check actually executed for this design

The dedicated [synthetic identity test](../../tests/phase2/test_ry_fidelity_identity.py) imports the unchanged `VQC` and `angles` definitions and extracts full matrices from its existing QNode. It constructs no learned objective or optimizer and reads no dataset. Across 48 synthetic pairs (16 each for seeds 1729/1730/1731), including near-boundary latent inputs ±20, it compares the product formula with complex128 state overlaps, perturbs all 16 shared head parameters, verifies unitarity, reconstructs the actual Z(3) observable, and compares autograd and central finite differences (step 10⁻⁵ radians) with the analytic input derivatives. Boundary tests separately check identical states, orthogonal states and 2π periodicity; exact boundary angles are mathematical cases, not claimed finite encoder outputs.

| Numerical identity | Maximum absolute discrepancy |
|---|---:|
| RY closed form vs full-circuit fidelity | 7.7716×10⁻¹⁶ |
| Fidelity under finite shared-head perturbation | 1.3323×10⁻¹⁵ |
| Full-circuit unitarity | 1.3323×10⁻¹⁵ |
| Analytic angular derivative vs autograd | 8.3267×10⁻¹⁶ |
| Angular derivative vs central finite difference | 3.6701×10⁻¹¹ |
| Gradient of fidelity with respect to terminal head | 3.8858×10⁻¹⁶ |
| Z(3) reconstruction vs original QNode | 8.8818×10⁻¹⁶ |

The maximum fidelity change when applying **different** heads to the two endpoints is 0.4466, confirming that shared-unitary cancellation is the relevant premise. The observed maximum absolute local-observable derivative with respect to a head parameter is 0.9890, confirming that the test has not made the VQC universally inactive. Neither check trains C0 or evaluates a patient. Tolerances are 10⁻¹⁰ for matrix/autograd identities and 10⁻⁸ for finite differences. The [machine-readable check](c0_v1_design/ry_identity_check.json) records exact errors, versions, seeds and status.

Reproduce only this verification, without any historical or C0 training:

```bash
MPLCONFIGDIR=/tmp/qarl-c0-mpl XDG_CACHE_HOME=/tmp/qarl-c0-cache .venv/bin/python -m pytest -q -s tests/phase2/test_ry_fidelity_identity.py
```

The two dedicated tests passed. Environment: Python 3.12.11, Torch 2.9.0, PennyLane 0.43.0, NumPy 2.5.3. This is a numerical identity check, not evidence for clinical performance, statistical generalization or the success of an experimental arm.

## 10. Planned optimization and computational budget

Each new arm covers 5 folds × 3 seeds = 15 final encoders. **Recommended minimum: 60 primary pipelines, comprising 90 optimizer stages; 150 deterministic LR probe fits.** No hyperparameter variants or extra head restarts are hidden in those counts.

| Per fold/seed pipeline | Encoder-active updates | VQC-active updates | Sequential optimizer updates | Pair draws | BCE training patient-state evaluations |
|---|---:|---:|---:|---:|---:|
| Historical B, existing only | 1,050 | 1,050 | 1,050 joint | 0 | 33,400, historical |
| C0-NQE-Q | 1,050 geometry | 1,050 BCE | 2,100 | 67,200 | 33,400 |
| C0-NQE-E | 1,050 geometry | 1,050 BCE | 2,100 | 67,200 | 33,400 |
| C0-Hybrid-Q | 1,050 joint | 1,050 joint | 1,050 joint | 67,200 | 33,400 |
| C0-Hybrid-E | 1,050 joint | 1,050 joint | 1,050 joint | 67,200 | 33,400 |

Two-stage methods have **twice the sequential update events** and 50 geometry epochs plus 50 head epochs. Each parameter block receives 1,050 active updates, as in B, but this does **not** match total computation, endpoint exposure, optimization path, objective or BCE gradients to the encoder. An endpoint-unshared implementation evaluates the encoder 134,400 times during stage 1, versus B's 33,400 BCE training examples. Hybrid arms add those geometric endpoint occurrences to their 33,400 BCE patients; repeated endpoints may be encoded once per update without changing the loss, so record both requested occurrences and actual encoded unique samples. No claim of equal example or wall-clock budget follows from equal parameter-active updates.

Across all four new arms: 94,500 sequential updates; 4,032,000 sampled pairs; 8,064,000 requested pair endpoints; 2,004,000 BCE training patient-state evaluations. The two-stage encoder is frozen during VQC training; cache its final training/validation embeddings rather than repeatedly differentiating it. Only 30 hybrid pipelines backpropagate BCE through the encoder. Both metric similarities are evaluated by analytic classical formulas: **zero additional QNode calls, hardware circuits or shots are needed for pair loss**. VQC statevector calls remain necessary for BCE.

For each pipeline, the standard 50-epoch fitting loop entails 1,050 BCE training QNode invocations plus 50 validation invocations, or 41,750 patient-state evaluations. Final same-checkpoint training/validation predictions add two calls and 835 states: **1,102 calls and 42,585 states**, excluding initialization, frozen-v₀ and derivative diagnostics. Across 60 pipelines this is 66,120 calls and 2,555,100 states for those specified operations. Historical B recorded 1,101 calls/41,917 states because its saved final scoring included only final validation; that logging difference must not be described as training effort. State counts are analytic simulator evaluations, not parameter-shift circuits, shots, hardware latency, energy or an end-to-end complexity advantage.

Track separately: preprocessing/input retrieval, stage-1 encoder work, pair sampling, pair similarities/backward, stage-2 fitting, per-epoch validation, probes, reference-kernel readouts, geometry, response derivatives, final scoring, serialization, process wall time and peak RSS (bytes on macOS). Fixed diagnostic gradient computations and snapshot matrices have their own counters and are excluded from optimizer step counts. A full validation-to-training reference matrix has 111,556 pair entries per kernel/encoder; full within-training and within-validation geometry has 222,778 and 13,861 unordered pairs. Cache symmetry and two-stage identical encoder snapshots, not patient-derived results across different encoders. No exhaustive pair matrices or figures are to be versioned.

### Wall-time planning, not a fabricated measurement

Historical B's mean recorded fit/logging time is 3.5344 s per fold/seed, excluding preprocessing and final scoring. Existing E0 `costs.csv` gives mean head-only update time 2.8889 s and epoch-validation time 0.09284 s, summed 2.9817 s. These are hardware/environment-specific historical anchors, not current matched benchmarks. Head-only fitting on new geometry can differ.

Let T_G denote one new 50-epoch pure geometry stage, ΔT_A the extra pair work in a hybrid relative to the historical B loop, and T_D all new snapshot/probe/reference/derivative costs. A transparent planning expression for all 60 pipelines is

\[
T_{\rm total}\approx30T_G+30(2.9817\,\mathrm{s})+30(3.5344\,\mathrm{s}+\Delta T_A)+T_D+T_{\rm input},
\]

or **195.5 s + 30T_G + 30ΔT_A + T_D + T_input**, using only historical loop anchors. T_G, ΔT_A, T_D and input preparation have not been measured, so no reliable complete wall-clock forecast or speedup ratio is asserted. Synthetic identity-test duration is not a training-cost estimate. New arms should report measured stage totals and the total cost of learning each representation; probes alone must not be presented as cheap complete pipelines.

A 25+25-epoch two-stage alternative would match the number of sequential update events but halve the VQC optimization budget and alter both encoder/head active budgets. A fresh 100-epoch B would supply an extended-budget reference but would violate the present no-rerun scope. **Recommend 50+50 with explicit cost accounting and matched Q/E schedules**, accepting that comparison with B is not equal-total-compute. Do not add the alternatives after seeing results.

## 11. Falsifiable hypotheses and interpretation discipline

| Prespecified hypothesis | Supporting descriptive pattern | Outcome weakening the explanation |
|---|---|---|
| H1: supervised RY geometry transfers beyond training | Q arm has lower held-out L_Q and larger held-out G_Q than its initial encoder and historical B, with all cells reported | Only training geometry improves; validation geometry is unchanged/worse or gains occur only in isolated favorable cells |
| H2: improved state separation benefits the restricted VQC | Held-out geometry improvement coexists with lower validation VQC log loss and higher AUC; show all paired training/validation effects | D_tr/G_Q improve but VQC loss or discrimination does not; overlap scores/probes improve while the VQC does not |
| H3: the specified state geometry matters beyond generic metric supervision | Within a schedule, Q improves predictive endpoints relative to E, with geometry/readout patterns consistent with the mechanism | E matches/exceeds Q or only training losses differ; supervised compression/regularization remains a sufficient alternative explanation |
| H4: gains reflect generalization rather than easier memorization | Final validation proper scores improve with transfer of geometric structure, not merely a perfect training fit | Larger train/validation gap, worse Brier/log loss, minority deterioration, unstable seeds, or geometric collapse despite improved training fit |
| H5: local observable accessibility matters | Larger balanced class-state distance is accompanied by improved J_v and VQC performance under the fixed readout | Unrestricted distances grow but J_v and prediction do not, suggesting restricted-head mismatch; no claim of a guaranteed Helstrom measurement |

These are directional questions, not promises or retrospective significance criteria. Report effect sizes in natural units and every exception; do not declare a discovery by counting favorable pairs or turn “not worse” into an equivalence claim. If AUC improves while log loss worsens, label the discrimination/probability tradeoff rather than accepting whichever endpoint favors the method. If Q/E are close, acknowledge the limited ability of a single fixed comparator to rule out all classical mechanisms.

No mediator-to-outcome causal claim is identified by a correlation between D_tr, rank, gradient norms or saturation and accuracy. The optimization interventions change representations and their learning histories. Only paired Q/E schedules hold the listed non-geometric factors fixed. In two-stage Q, all stage-1 geometric change is assigned to the classical encoder trained with a state-derived analytic loss. In hybrid Q, do not infer that the VQC was unnecessary merely because the final encoder admits a strong LR probe: BCE through the VQC may have helped produce it. Conversely, do not credit the VQC for an improvement reproduced by a fixed class-reference score or by the classical metric-learning control.

## 12. Bibliographic justification and explicit adaptation boundary

| Source and primary locator | Role in C0-v1 | What is not being reproduced or claimed |
|---|---|---|
| Hur, Araujo & Park (2024), *Neural Quantum Embedding*, Phys. Rev. A 110, 022411; Eq. (3), Sec. II B, Appendix A. [DOI](https://doi.org/10.1103/PhysRevA.110.022411), [arXiv v2](https://arxiv.org/abs/2311.11412v2) | Exact supervised squared-fidelity pair objective on a restricted RY map; separates embedding training from later readout fitting | Their ZZ feature circuit, CNN/MLP, optimizer/pair weighting, QCNN, hardware results; BCE+auxiliary joint learning is our hybrid adaptation |
| Lloyd, Schuld, Ijaz, Izaac & Killoran (2020), *Quantum embeddings for machine learning*, Secs. II–III. [arXiv v2](https://arxiv.org/abs/2001.03622v2) | Class ensemble purities, Hilbert–Schmidt and trace-distance diagnostics, and class-overlap readout distinguish state geometry from measurement capacity | Their QAOA/re-uploading ansatz or an implementable Helstrom optimum under B; their class-weighting/linear-risk identities do not automatically become our BCE claims |
| Sharma et al. (2026), *Metric Learning with Quantum Feature Spaces*, arXiv:2312.01655v6, 2026-10-20; Eq. (2), Secs. 2–3. [Pinned v6](https://arxiv.org/abs/2312.01655v6) | Terminal-unitary cancellation, classical computability of product-state fidelity, concentration and encoder memorization cautions | Historical “Quantum Polar” triplet/ZZ architecture, full two-angle projective manifold, local-to-global curriculum, or an exact reproduction of underspecified few-shot objectives |
| Rodriguez-Grasa, Ban & Sanz (2025), *Neural quantum kernels*, Phys. Rev. Research 7, 023269; Secs. III–V, Appendix B. [DOI](https://doi.org/10.1103/xphb-x2g4) | Distinguishes QNN pretraining and downstream kernel fitting; supports separating training geometry from readout performance | Progressive-width/re-uploading circuit, controlled-SU(2), SVM fitting, local label-state infidelity, or a new projected-kernel readout in the primary VQC |
| Jäger, Elsässer & Torabian (2026), *Quantum feature-map learning with reduced resource overhead*, Phys. Rev. Research 8, 023247; Eqs. (26)–(28), Sec. III. [DOI](https://doi.org/10.1103/v29j-rh32) | Independent statement of terminal gate cancellation and motivation for honest resource accounting | Adaptive gates, feature selection, analytic reconstruction optimization, projector readout or target-alignment training |
| Thanasilp, Wang, Cerezo & Holmes (2024), *Exponential concentration in quantum kernel methods*, Nature Communications 15, 5200; Proposition 3 and Eq. (26). [DOI](https://doi.org/10.1038/s41467-024-49287-w) | Global fidelity can concentrate even with product RY states; motivates reporting kernel distributions and not equating nonzero gradients with useful generalization | A diagnosis of barren plateaus in this four-qubit, correlated-angle, exact simulator; finite-shot asymptotic failure is not assumed here |

C0-v1 deliberately uses **one** published objective family. Lloyd supplies geometry/readout theory and diagnostics; Sharma supplies compatibility and failure-mode cautions, not extra losses. The E similarity and balanced-pair/schedule/lambda choices are transparent project adaptations. No novelty or superiority is presumed, and no C1 function is proposed.

## 13. Implementation acceptance criteria and review gate

The protocol is reviewable now; **implementation and execution remain unauthorized in this task**. Before later implementation, accept or amend the recommended choices as a new saved/hash-pinned version. No config, C0 loss module or training runner is created by this design task. The only new executable file is the synthetic identity test of existing mathematics.

Implementation may be accepted for a later execution review only if all of the following hold:

- The four arms, fixed λ=1, Gaussian scale, balanced 64-pair scheme, diagonal exclusion, frozen stage-2 encoder, 50/50 versus 50 budgets, h=s initialization and 150 probes are explicitly accepted. Alternatives in this document are explanatory, not extra registered arms.
- Input availability/provenance is verified: exact historical PCA32 preprocessing state/row order and initial model tensors, canonical five folds/three seeds, development-only access, initial embedding reconstruction tolerance 10⁻¹², and refusal of altered or missing historical artifacts. This availability audit was not performed on patient artifacts in the present task.
- Numerical tests pass for analytic fidelity, self/orthogonal/periodic limits, derivatives, terminal cancellation, non-invariant local observables and classical-control local scale. Future objective tests must also check the exact squared residual, stratum weights and signed/0–1 label equivalence, both endpoint gradients, fixed bandwidth/lambda and mean reductions. This task executed only the existing-map identity tests, not tests of nonexistent C0 objectives.
- Isolation tests prohibit validation/locked rows or labels from all training, pair sampling, references and probes. Pair sampling tests cover exclusions, stratum counts, replacement policy and deterministic streams; diagnostic RNG does not alter training. Altered validation labels cannot change trained tensors or selection. Every learned transformation remains training-only.
- Optimizer membership, inactive-parameter hashes, fresh stage optimizer state, original initialization tensors, unchanged circuit/readout, all 1,050/2,100 steps, shared BCE/per-pair streams, fixed final checkpoints and no hidden restarts are tested. A λ=0 compatibility test should match the original update behavior on a **synthetic fixture**, without rerunning historical B.
- Metrics are independently recomputable from saved development probabilities; geometric definitions/diagonals, leave-one-out references and train-only class weights are tested. Fold/seed summaries and failed-run denominators cannot be selected by outcome. Probes/readouts remain diagnostic and never choose the reported primary method.
- Stage-specific time, memory, QNode/state counters, encoder endpoint counts, pair counts and diagnostic overhead are logged. No claim of equal total compute or hardware efficiency is allowed. Prospective runtime must be measured later rather than inferred from this identity check.
- Future detailed outputs go to a new, non-overwriting ignored directory such as `runs/phase2/c0_v1/`; independently editable English methodology/results and small non-patient summaries go under `reports/phase2/`; a read-only notebook may inspect completed artifacts. No generated PNG/SVG, raw data or patient-level predictions are committed.