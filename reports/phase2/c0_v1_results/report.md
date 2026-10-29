# C0-v1 development cross-validation results

**Status:** completed development cross-validation on 2026-10-29. All **60/60 primary pipelines** completed once, with **zero failures, replacement runs or parameter changes**. The locked test was not evaluated. A/B/D0/E0 were not retrained. This report describes the approved four-arm experiment, not an independent confirmation, a quantum computational advantage, or an original C1 method.

## Research question and principal observations

Does supervised separation of the existing product-RY states translate into better prediction and generalization with the restricted B variational quantum classifier (VQC)? The primary estimands are **Q minus E within each training schedule**, jointly reporting ROC-AUC (higher is better) and log loss (lower is better).

NQE-Q minus NQE-E has mean validation ROC-AUC **+0.01510** and log loss **−0.01050 nats**. All five seed-averaged folds favor Q on AUC, but only three favor Q on log loss. The largest log-loss improvement occurs in fold 4; two folds worsen. Hybrid-Q minus Hybrid-E has mean AUC **−0.00908** and log loss **+0.02095 nats**. There is no consistent benefit from replacing the Euclidean similarity with RY fidelity across schedules.

All four methods improve held-out pair geometry relative to initialization. That change does not guarantee better VQC prediction: the hybrids have larger class-state trace distance than the two-stage arms but poorer mean predictive scores. On the NQE-Q final encoder, a training-only linear probe on latent coordinates reaches AUC **0.94152**, log loss **0.25597**, versus VQC **0.93003**, **0.29577**. This supports separating representation quality from the ability of the prescribed VQC readout to use it. It does not show that the VQC gradient was unnecessary for learning the hybrid representations.

## Fixed data, objectives and training procedure

The experiment uses **835 development patients: 645 ER+, 190 ER−**, in the canonical five folds, IDs 0–4. Each fold has 668 training patients (516 ER+, 152 ER−) and 167 validation patients (129 ER+, 38 ER−). Seeds are **1729, 1730, 1731**. The 209 locked-test patients remain excluded. The folds have informed previous research decisions; this is development evidence rather than a new independent confirmatory sample.

Inputs are historical fold-local standardized PCA32, reconstructed with the original training-fitted preprocessing objects without refitting: training-only expression filtering, frozen-reference TMM, log2-CPM, zero-variance filtering, gene scaling, randomized PCA32, and score standardization. The availability audit verified five preprocessing objects and fifteen initial checkpoints and reproduced saved initial embeddings with **maximum absolute error 0**, below the prespecified **10⁻¹²** tolerance. Each fold's training partition supplies all pairs, reference sets and probe fits. Validation metrics are logged after fixed updates, never used for selection. Split metadata enforces development/locked disjointness without loading locked features or labels.

The unmodified encoder is Linear(32,8) → Tanh → Linear(8,4), with 300 parameters. Write z=g_w(x), a=π tanh(z/2) and |φ(x)⟩=⊗ⱼ RY(aⱼ)|0⟩. The unmodified four-qubit VQC has two RY/RZ layers and CNOT chain 0→1→2→3, 16 nominal head parameters, terminal Z₃ expectation f, and probability p=clip((1+f)/2,10⁻⁷,1−10⁻⁷). ER+ is y=1; ER− is y=0; threshold p≥0.5 predicts ER+. Log loss uses natural logarithms. AP for ER− uses 1−p and the ER− indicator. Balanced accuracy averages the two class sensitivities.

For a pair, sᵢⱼ=1[yᵢ=yⱼ]. The kernels and losses are exactly

\[
k_Q(i,j)=\prod_{r=0}^{3}\cos^2((a_{ir}-a_{jr})/2),\quad
k_E(i,j)=\exp(-\|a_i-a_j\|^2/4),\quad
\ell_R=(k_R-s_{ij})^2,
\]

\[
L_R=\tfrac14 E_{++,i\ne j}\ell_R+\tfrac14 E_{--,i\ne j}\ell_R+\tfrac12 E_{+-}\ell_R,
\qquad L_{BCE}=-E_i[y_i\log p_i+(1-y_i)\log(1-p_i)].
\]

Each metric update uses **64 pairs: 16 ++, 16 −−, 32 +−**, sampled uniformly within strata, independently with replacement across draws; both endpoints receive gradients. Same-class self pairs are excluded. Repeated patients/pairs across distinct draws are allowed. The simple mean estimates the displayed balanced risk, not the cohort-prevalence all-pairs risk. BCE retains natural prevalence and uses shuffled batches of 32, with a final batch of 28. There are 21 updates per epoch. Diagnostic pair risks exclude diagonals; density-matrix purities include them by definition. Training class-reference scores use leave-one-out own-class references; validation references are training patients only.

| Arm | First stage | Second stage | Updated parameters | Sequential updates |
|---|---|---|---|---:|
| C0-NQE-Q | 50 epochs L_Q | 50 epochs BCE | Encoder only, then VQC only with encoder frozen | 2,100 |
| C0-NQE-E | 50 epochs L_E | 50 epochs BCE | Encoder only, then VQC only with encoder frozen | 2,100 |
| C0-Hybrid-Q | 50 epochs BCE + L_Q | None | Encoder and VQC jointly; auxiliary has no VQC gradient | 1,050 |
| C0-Hybrid-E | 50 epochs BCE + L_E | None | Encoder and VQC jointly; auxiliary has no VQC gradient | 1,050 |

Every trainable stage uses fresh AdamW state, lr=0.01, weight_decay=0.001, betas=(0.9,0.999), eps=10⁻⁸, no scheduler, clipping or early stopping. Frozen blocks are excluded from the optimizer. Lambda is **1**, the Gaussian denominator is **4**, with no data-dependent normalization, bandwidth search or retrospective rescaling. Equal kernel/loss ranges and local second-order curvature motivate this scale; global gradient norms need not match. Decoupled weight decay is an optimizer operation, not another displayed objective term.

All arms start from the same original B **initial**, not trained, encoder/head tensors within each fold/seed. NQE heads remain at their pristine initialization during geometry training and start stage 2 with fresh optimizer state. NumPy PCG64 SeedSequence([seed,fold,1]) controls pair draws; role 2 supplies independent diagnostic randomness. Pair and patient streams are paired across Q/E. The final epoch is the sole result checkpoint. Initialization, geometry endpoints and final checkpoints are retained; no validation-minimum checkpoint was selected.

The scalar kernel is evaluated analytically on CPU; no finite-shot estimates or hardware are used. A shared terminal unitary cancels from full-state overlap: ⟨φᵢ|V†V|φⱼ⟩=⟨φᵢ|φⱼ⟩, so ∂L_R/∂v=0. Synthetic tests verified product-RY fidelity, state-vector agreement, terminal-unitary cancellation and gradients. Independent final verification recomputed class-state trace distances from 16-dimensional states (maximum error **8.88×10⁻¹⁶**). Both Q and E similarities are efficiently classically computable.

## Comparability and bibliographic boundary

| Comparison | Held fixed | Remaining limitation | Role |
|---|---|---|---|
| NQE-Q versus NQE-E | Initial tensors, streams, labels, coordinates, 50+50 stages, readout | Similarity changes global topology and gradient scale | Primary |
| Hybrid-Q versus Hybrid-E | Initial tensors, streams, BCE, lambda, 50 joint epochs, readout | Similarity changes global topology and gradient scale | Primary |
| NQE versus Hybrid | Architecture, initial tensors, data | Encoder/head gradient paths and sequential total budgets differ | Descriptive |
| C0 versus historical B | Splits, seeds, architecture, preprocessing | C0 adds balanced pair supervision; NQE adds a full stage; procedures/total budgets differ | Context only |
| VQC versus probes on same encoder | Final encoder and evaluation patients | Readout family, fitting procedure and calibration differ | Conditional readability control |
| Frozen initial VQC versus trained VQC | Exact same final encoder | One fixed initial observable, not optimal frozen capacity | Conditional readout control |

[Hur et al. (2024), Eq. (3)](https://doi.org/10.1103/PhysRevA.110.022411) supplies the supervised squared-fidelity pair formula. C0-NQE-Q adapts that objective to the existing RY map and frozen-encoder downstream VQC; it does not reproduce their ZZ circuit, QCNN, hardware, optimizer or pair distribution. BCE plus the auxiliary in C0-Hybrid-Q is explicitly a project hybrid adaptation. E is the project classical metric-learning control, matching local curvature in the same bounded angular chart rather than reproducing another paper.

[Lloyd et al. (2020)](https://arxiv.org/abs/2001.03622v2) motivates class-state distance and class-overlap readouts; its measurement theory does not equate these quantities with BCE or restricted-VQC capacity. [Sharma et al., pinned v6](https://arxiv.org/abs/2312.01655v6) motivates the terminal-unitary/computability and memorization cautions, without importing a curriculum or another loss. [Rodriguez-Grasa et al. (2025)](https://doi.org/10.1103/xphb-x2g4) informs the separation of learned embeddings from downstream readouts; [Jäger et al. (2026)](https://doi.org/10.1103/v29j-rh32) informs cancellation and resource accounting; [Thanasilp et al. (2024)](https://doi.org/10.1038/s41467-024-49287-w) motivates kernel-distribution/concentration diagnostics. Their different circuits, kernel learners or asymptotic settings are not reproduced. The complete frozen rationale remains in the [design protocol](../c0_protocol_v1.md), whose original proposal-status wording is preserved as a historical design artifact; this report records actual execution.

## Primary and per-pipeline numerical results

The reporting unit is the **mean of three seeds within a fold**, followed by the mean and sample SD of five fold means. SD is descriptive, not a confidence interval. Folds have overlapping training sets, and the fifteen fold/seed cells are not independent replications. No p-values, best-arm selection rule or confirmatory significance claim is introduced. All tables report validation metrics unless stated otherwise. Positive ΔAUC and negative Δlog loss favor Q.

| Arm | ROC-AUC | Log loss (nats) | AP ER− | Balanced accuracy | Brier |
| --- | --- | --- | --- | --- | --- |
| C0-NQE-Q | 0.93003 ± 0.02946 | 0.29577 ± 0.05424 | 0.81237 ± 0.05156 | 0.81752 ± 0.01603 | 0.07850 ± 0.01134 |
| C0-NQE-E | 0.91493 ± 0.02555 | 0.30627 ± 0.06557 | 0.78497 ± 0.05839 | 0.81064 ± 0.03852 | 0.08340 ± 0.01611 |
| C0-Hybrid-Q | 0.90631 ± 0.02487 | 0.34071 ± 0.07771 | 0.79253 ± 0.06368 | 0.84950 ± 0.03230 | 0.07980 ± 0.01673 |
| C0-Hybrid-E | 0.91540 ± 0.03100 | 0.31976 ± 0.09714 | 0.80058 ± 0.06889 | 0.83985 ± 0.04214 | 0.07959 ± 0.01962 |

Primary Q−E contrasts (mean ± descriptive fold SD):

| Schedule | ΔAUC mean | ΔAUC SD | Δlog loss mean | Δlog loss SD |
| --- | --- | --- | --- | --- |
| Hybrid | -0.00908 | 0.01275 | 0.02095 | 0.03742 |
| NQE | 0.01510 | 0.01396 | -0.01050 | 0.04814 |

All seed-averaged fold contrasts:

| Schedule | Fold | ΔAUC Q−E | Δlog loss Q−E (nats) |
| --- | --- | --- | --- |
| Hybrid | 0 | -0.01890 | 0.02639 |
| Hybrid | 1 | -0.01054 | 0.06903 |
| Hybrid | 2 | -0.02258 | 0.03725 |
| Hybrid | 3 | -0.00231 | 0.00270 |
| Hybrid | 4 | 0.00891 | -0.03064 |
| NQE | 0 | 0.00136 | 0.04046 |
| NQE | 1 | 0.02339 | -0.00579 |
| NQE | 2 | 0.01408 | -0.00226 |
| NQE | 3 | 0.00252 | 0.00521 |
| NQE | 4 | 0.03414 | -0.09009 |

All sixty primary pipelines. Each cell is **ROC-AUC / log loss (nats)**; every cell completed:

| Fold | Seed | C0-NQE-Q | C0-NQE-E | C0-Hybrid-Q | C0-Hybrid-E |
| --- | --- | --- | --- | --- | --- |
| 0 | 1729 | 0.94961 / 0.23913 | 0.94961 / 0.22501 | 0.92921 / 0.24294 | 0.92921 / 0.25976 |
| 0 | 1730 | 0.93248 / 0.33230 | 0.92309 / 0.25020 | 0.89229 / 0.35934 | 0.95634 / 0.24642 |
| 0 | 1731 | 0.94166 / 0.24234 | 0.94696 / 0.21719 | 0.95594 / 0.22160 | 0.94859 / 0.23853 |
| 1 | 1729 | 0.92636 / 0.26958 | 0.90167 / 0.27970 | 0.89004 / 0.44148 | 0.90453 / 0.25224 |
| 1 | 1730 | 0.94880 / 0.21499 | 0.91371 / 0.30115 | 0.91391 / 0.27123 | 0.93962 / 0.24009 |
| 1 | 1731 | 0.91718 / 0.36219 | 0.90677 / 0.28327 | 0.92595 / 0.23324 | 0.91738 / 0.24653 |
| 2 | 1729 | 0.96940 / 0.20746 | 0.93329 / 0.28043 | 0.89657 / 0.31769 | 0.94472 / 0.24459 |
| 2 | 1730 | 0.94757 / 0.31550 | 0.92350 / 0.26122 | 0.92758 / 0.32032 | 0.94553 / 0.27446 |
| 2 | 1731 | 0.93472 / 0.26497 | 0.95267 / 0.25305 | 0.91677 / 0.28834 | 0.91840 / 0.29553 |
| 3 | 1729 | 0.89229 / 0.35057 | 0.91310 / 0.29204 | 0.85067 / 0.48789 | 0.86128 / 0.49467 |
| 3 | 1730 | 0.87699 / 0.40199 | 0.84353 / 0.48002 | 0.85700 / 0.53255 | 0.83701 / 0.58155 |
| 3 | 1731 | 0.86822 / 0.42342 | 0.87332 / 0.38829 | 0.88168 / 0.40428 | 0.89800 / 0.34041 |
| 4 | 1729 | 0.94206 / 0.24981 | 0.91656 / 0.38418 | 0.88760 / 0.42151 | 0.91187 / 0.41872 |
| 4 | 1730 | 0.94655 / 0.30401 | 0.91840 / 0.31531 | 0.92391 / 0.31306 | 0.87271 / 0.39784 |
| 4 | 1731 | 0.95655 / 0.25831 | 0.90779 / 0.38293 | 0.94553 / 0.25519 | 0.94574 / 0.26512 |

Individual paired-cell direction counts (out of 15; not independent trials):

| Schedule | AUC favors Q | Log loss favors Q | Both favor Q |
| --- | --- | --- | --- |
| Hybrid | 5 | 8 | 5 |
| NQE | 10 | 7 | 7 |


## Geometry, measurement response and generalization

For each class, ρ_c is the uniform mixture of its embedded pure states. Report **D_tr=½||ρ₊−ρ₋||₁**, **D_HS²=Tr[(ρ₊−ρ₋)²]**, and the raw observable contrast **J=½Tr[Z₃ V(ρ₊−ρ₋)V†]** separately. The descriptive ratio |J|/D_tr obeys a class-mean contrast bound; it is **not a complete measure of discriminative capacity**, ROC-AUC, calibration or patient-level overlap. Density purities include self terms; supervised pair risk does not. The exact unclipped expectation is used for J; prediction uses the separately clipped probability.

All four arms reduce held-out Q pair risk relative to their common initialization in all fifteen cells. Initial mean validation Q risk is **0.33968**, Q separation **0.00802**, D_tr **0.12014**. Final validation Q risks are **0.22632, 0.22694, 0.21138, 0.21174**, in NQE-Q/E, Hybrid-Q/E order. Historical B Q risk is **0.23827**, Q separation **0.35049**, D_tr **0.71938**. NQE-Q's Q gap **0.33710** is below B despite lower pair risk; Hybrid-E's Q gap **0.40559** exceeds Hybrid-Q's **0.38844** despite E supervision. Pair residual, separation gap and trace distance quantify different features of geometry.

The hybrids' greater state distance and observable contrast do not produce better mean VQC prediction than NQE-Q. NQE-Q's mean accessibility ratio is **0.98633**, yet its latent linear probe predicts better. Class-mean accessibility therefore cannot diagnose all remaining predictive limitations. Frozen-initial-readout and trained-readout results evaluate the same final representations; neither comparison identifies the causal contribution of the VQC to hybrid encoder learning. In NQE, stage-1 geometric changes are entirely caused by the classical encoder optimized with an analytic state-derived loss, not by VQC updates.

Final latent entropy rank is about **1.02–1.04**, versus **3.15** initially. Numerical rank is below four in **4/15, 9/15, 10/15, 10/15** final validation embeddings respectively. Coordinate contraction is not evidence of uniquely quantum failure. The fraction of angles with |a|≥0.95π is approximately **10.06%, 9.92%, 2.09%, 1.72%**. Hybrids have less saturation yet worse mean VQC predictive scores than NQE-Q, weakening a saturation-only explanation. The saved diagnostics also contain angular spectra, within/between-class kernel distributions and coordinate-response norms. Gradient magnitude or nonzero response alone does not establish useful prediction or mitigation of barren plateaus.

Final-checkpoint training AUC exceeds **0.9979** for every arm's mean. Training/validation log-loss gaps are **0.26321, 0.25265, 0.30777, 0.29090 nats**, versus historical B **0.23787**. Training Q risks are **0.01760, 0.03192, 0.02636, 0.03081**, far below held-out risks. Geometric transfer exists, but substantial training specialization remains; these auxiliaries do not show a general reduction of overfitting.

The notebook presents all fifty epochs of each applicable stage. Online losses are averages over changing within-epoch tensors and sampled minibatches; they are not interchangeable with final-checkpoint full-training scores. Validation losses are end-of-epoch measurements. Curve inspection did not select epochs, lambda, seeds or runs. Fixed diagnostic batches at epochs 0, 25 and 50 show auxiliary/BCE encoder-gradient norm ratios increasing from about **0.31** to **1.06 (Q)** and **1.01 (E)**; mean final gradient cosines are **−0.048** and **−0.051**. Lambda=1 therefore does not imply constant relative gradient influence. These are descriptive batch diagnostics, not proof of a causal optimization mechanism.

## Classical and historical controls

Each final encoder is held fixed for LR-z and LR-angle, fitted only on its training partition: C=1, L-BFGS, max_iter=2000, tol=10⁻⁴, intercept, unweighted L2, no scaling or tuning. There are **120 final probe fits**, plus **30 shared initial-encoder probe fits**, not three repeated fits per representation. All 150 saved estimators were independently checked, with no convergence warnings. The encoder stays at 300 parameters and four latent coordinates; the initial encoder is a capacity reference, not maximal achievable capacity.

Class-reference Q/E readouts average similarities to training class members. Their uncalibrated probability is clip((1+μ₊−μ₋)/2); they receive neither validation calibration nor label fitting. Their log loss is link-dependent. LR and VQC optimization/capacity are different, so probe gains are evidence of available information conditional on the encoder, not a matched-compute superiority claim. Historical B_classical is a separately co-trained encoder/head and must not be treated as the same final representation.

Historical B has mean AUC **0.92147**, log loss **0.28667**. NQE-Q has higher AUC **0.93003** but worse log loss **0.29577**, and balanced accuracy **0.81752** versus B **0.88154**. The other C0 arms worsen both mean AUC and log loss versus B. Historical B's latent probe is **0.93714/0.28192**, and PCA4 logistic regression **0.94904/0.22858**. No new C0 VQC shows a joint improvement over these controls. These comparisons provide context and do not claim equal total training budget. Frozen historical checkpoints were read for diagnostics without retraining.

Same-encoder validation readouts: **ROC-AUC / log loss (nats)**.

| Arm | Frozen v₀ | VQC | LR-z | LR-angle | Q reference | E reference |
| --- | --- | --- | --- | --- | --- | --- |
| C0-NQE-Q | 0.28395 / 1.29923 | 0.93003 / 0.29577 | 0.94152 / 0.25597 | 0.94168 / 0.28052 | 0.93169 / 0.26850 | 0.94083 / 0.25191 |
| C0-NQE-E | 0.26074 / 1.27736 | 0.91493 / 0.30627 | 0.93891 / 0.27910 | 0.93865 / 0.28962 | 0.90570 / 0.29337 | 0.93424 / 0.24914 |
| C0-Hybrid-Q | 0.70638 / 0.62895 | 0.90631 / 0.34071 | 0.92829 / 0.30776 | 0.92817 / 0.32382 | 0.90744 / 0.29530 | 0.91288 / 0.29279 |
| C0-Hybrid-E | 0.84125 / 0.38043 | 0.91540 / 0.31976 | 0.93169 / 0.29954 | 0.93105 / 0.31616 | 0.91418 / 0.29298 | 0.91523 / 0.29172 |

Distinct geometry and observable summaries (validation means):

| Arm | Q pair risk | E pair risk | D_tr | D_HS² | J (raw) | Mean abs(J)/D_tr | Latent entropy rank |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C0-NQE-Q | 0.22632 | 0.19849 | 0.58686 | 0.68732 | 0.57904 | 0.98633 | 1.04120 |
| C0-NQE-E | 0.22694 | 0.18887 | 0.58379 | 0.66417 | 0.55286 | 0.95476 | 1.04498 |
| C0-Hybrid-Q | 0.21139 | 0.20290 | 0.66781 | 0.79121 | 0.61807 | 0.92597 | 1.04439 |
| C0-Hybrid-E | 0.21174 | 0.20446 | 0.70020 | 0.82658 | 0.62148 | 0.88560 | 1.02323 |


## Computational cost and actual execution

The approved forecast used historical loop anchors: **195.5 seconds plus geometric-stage time, hybrid auxiliary overhead, diagnostics and input work**, not a complete runtime guarantee. Measured execution took **216.8745 seconds**, on CPU with **one Torch thread**, exact `default.qubit` simulation (`shots=None`), and process peak RSS **553,811,968 bytes (528.16 MiB)**. This excludes the earlier availability audit, subsequent independent verification and notebook rendering. No accelerator or quantum device was used.

Every arm uses 1,050 geometry-bearing updates, **67,200 sampled pairs**, **134,400 pair-endpoint occurrences**, and **33,400 BCE patient occurrences**. NQE adds 1,050 sequential encoder-only updates to its 1,050 head updates; hybrids have 1,050 joint updates. Historical B has no geometric pretraining stage. NQE therefore has twice the sequential update count of B/hybrid and a different active-parameter history, even when fast analytic geometry makes its measured time small. A speed ratio from these small simulator timers would not establish computational advantage.

Across all sixty pipelines: **94,500 optimizer steps**, **4,032,000 sampled pairs**, **8,064,000 pair endpoints**, and **2,004,000 BCE patient occurrences**. The primary-loop counters record **66,210 QNode calls / 2,557,980 sample-state evaluations**, including 90 joint-gradient diagnostic calls / 2,880 states. Additional saved direct response, initial/frozen scoring and diagnostic counters are separate fields in [costs.csv](costs.csv) and per-run resources; the primary counters are not the total cost of every diagnostic. Analytic pair evaluation adds no QNode calls. Class-reference calculations and probes also consume classical compute.

Mean recorded stage times per case are NQE-Q **0.3003 s geometry + 2.9417 s head**, NQE-E **0.2933 + 2.9531 s**, Hybrid-Q **3.5482 s joint**, Hybrid-E **3.5625 s joint**. Final encoder diagnostic timers average about **0.099 s** per case. Partial timers omit some initial/frozen evaluations, input work, scoring and serialization and should not be forced to sum to the full process wall time. Learning all representations, fitting probes and calculating diagnostics is included in the recorded full execution; probe time alone is not a pipeline cost.

## Hypotheses, limitations and scientific implications

| Prespecified question | Observation | Assessment |
|---|---|---|
| H1: supervised geometry transfers beyond training | All methods lower held-out Q pair risk versus initialization; not all separation measures improve versus B | Partial support, metric-dependent |
| H2: improved state separation benefits restricted VQC prediction | Larger hybrid D_tr/J coexist with poorer predictive means; no joint gain over historical B | A broad geometry-to-prediction explanation is weakened |
| H3: Q differs favorably from classical geometric supervision | NQE mean favors Q, with heterogeneous loss; hybrid mean favors E | Schedule-dependent evidence, no general Q superiority |
| Generalization/stability improves | Strong training specialization, variable fold/seed contrasts and rank contraction remain | No broad improvement established |
| Observable-accessibility ratio captures predictive capacity | NQE-Q ratio near one, while LR-z outperforms its VQC on both main metrics | A complete-capacity interpretation is contradicted |

The most relevant unresolved issue for a future scientific investigation is the gap between supervised state geometry, patient-level class overlap, restricted readout and held-out calibration. The present results motivate investigating a concrete, falsifiable limitation only if it survives classical geometric/readout controls and independent evaluation. They do not establish novelty, a favorable C1 objective, or quantum advantage. No C1 function or implementation is proposed here.

There is one head start per learned representation; seed variability mixes encoder and head effects. Q/E match local curvature but differ globally and in gradient scale. Pair balancing changes the supervision distribution relative to B. The frozen readout is one initial observable, not optimal measurement capacity. Probe and class-reference controls differ in capacity, regularization and probability link. Development folds overlap in training and have already informed project design. These limits preclude independent confirmatory or causal mediator claims.

## Integrity, tests and reproducibility

The preparation condition passed before launch. **40 targeted scientific/isolation/integration tests** passed immediately before execution. After adding read-only summary/notebook safeguards, **44 tests** passed. Tests include exact objectives, label/diagonal conventions, paired streams, frozen blocks, gradients and terminal cancellation, closed config, validation isolation, synthetic lambda-zero compatibility, historical initialization behavior, fold-level aggregation and missing-artifact refusal. Technical tests used synthetic data; their outputs did not alter scientific decisions.

The final independent verifier performed **no fitting**. It checked all sixty saved checkpoints against audited development inputs, all 150 probes, all **870 classification metric rows**, paired streams, fixed epochs/updates and frozen tensors. Maximum absolute errors were **0** for initial tensors, embeddings and primary probabilities; **7.77×10⁻¹⁶** for probe probabilities; **4.44×10⁻¹⁶** for classification metrics. It recomputed trace distances and checked the observable bound. Hash comparison found **zero changed or missing files among 1,950 preexisting artifacts** and unchanged execution sources; **803 run artifact files** have a complete local SHA-256 manifest. See [integrity.json](integrity.json). This is artifact consistency verification, not an independent experimental replication.

The source notebook has no embedded outputs and loads only small aggregate CSV/JSON artifacts. A separate local copy executed top-to-bottom, generating nine figures, with fitting, array/checkpoint/cohort loaders blocked; all figures were visually inspected. Its code does not train, refit preprocessing, load patient-level arrays or access the locked test. The public CSVs contain aggregate fold/seed statistics rather than patient-level predictions. Checkpoints, predictions, complete logs and full manifests remain in ignored execution storage. Existing historical artifacts were preserved. There was no commit or push.

Execution base: Git `20b32d7`. Environment: Python 3.12.11, Torch 2.9.0, PennyLane 0.43.0, NumPy 2.5.3, scikit-learn 1.9.1, pandas 2.3.3, PyArrow 21.0.0. Config: [configs/c0_v1.yaml](../../../configs/c0_v1.yaml), unchanged. Protocol SHA-256: `e80c3117b1a0760bde4e528cd621966671f58f5708acc9f001dd4fe6f338f136`. An earlier design hash differed solely through removal of administrative prose; the local documentation crosswalk verifies scientific sections and requirements unchanged. The protocol, objectives, lambda, sampling, initialization, budgets, final checkpoints and primary contrasts remain intact.

The commands actually executed were:

```bash
MPLCONFIGDIR=/tmp/qarl-c0-mpl XDG_CACHE_HOME=/tmp/qarl-c0-cache .venv/bin/python -m src.phase2.run_c0 --execute-cv --audit-folder runs/phase2/c0_v1_execution_audit --output runs/phase2/c0_v1_full_cv
.venv/bin/python -m src.phase2.analyze_c0 --run runs/phase2/c0_v1_full_cv --output reports/phase2/c0_v1_results
.venv/bin/python -m src.phase2.verify_c0_results --run runs/phase2/c0_v1_full_cv --output runs/phase2/c0_v1_final_verification
MPLCONFIGDIR=/tmp/qarl-c0-mpl XDG_CACHE_HOME=/tmp/qarl-c0-cache .venv/bin/python -m pytest -q tests/phase2/test_c0_results.py tests/phase2/test_c0.py tests/phase2/test_ry_fidelity_identity.py tests/phase2/test_experiments.py::test_paired_quantum_initialization tests/phase2/test_experiments.py::test_quantum_shapes_and_gradients tests/phase2/test_experiments.py::test_angles
```

The analysis/verifier commands require fresh destination directories and original local inputs; they do not silently overwrite this human-edited report. Do not repeat training to inspect figures. Open [the read-only notebook](../../../notebooks/05_c0_development_results.ipynb) and point its configurable summary directory to this report folder. It uses [shared plotting functions](../../../src/phase2/c0_views.py). Exhaustive numeric records are [metrics.csv](metrics.csv), [paired_Q_minus_E.csv](paired_Q_minus_E.csv), [geometry.csv](geometry.csv), [generalization.csv](generalization.csv), [gradient_components.csv](gradient_components.csv), [curves.csv](curves.csv), [costs.csv](costs.csv), and [historical_metrics.csv](historical_metrics.csv). The small aggregate tables make the notebook inspectable without private patient-level artifacts.
