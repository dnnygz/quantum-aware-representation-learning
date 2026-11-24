# Brief methodological review for E0: Quantum Metric Learning and Neural Quantum Embedding

**Status: primary-source review for protocol approval, not an implementation.**
Reviewed on 2026-10-28. The versioned manuscripts below were inspected directly;
statements about our architecture are identified separately as local analysis.
This review makes no performance prediction for TCGA-BRCA.

## 1. Lloyd et al.: Quantum Metric Learning

Seth Lloyd, Maria Schuld, Aroosa Ijaz, Josh Izaac, and Nathan Killoran,
*Quantum embeddings for machine learning*, arXiv:2001.03622v2 (10 February 2020).
Primary text: [Sections I–III and Figure 5](https://arxiv.org/html/2001.03622v2).
Version/date metadata: [arXiv record](https://arxiv.org/abs/2001.03622v2).

For class-averaged embedded states rho and sigma, their practical objective is

\[
C=1-\tfrac12 D_{\rm hs},\qquad
D_{\rm hs}=\operatorname{Tr}(\rho^2)+\operatorname{Tr}(\sigma^2)
-2\operatorname{Tr}(\rho\sigma).
\]

This is squared Hilbert–Schmidt separation, estimated through same-/different-
class overlaps. Their trace-distance formulation instead uses
D_tr(rho,sigma)=||rho−sigma||_1/2. Under the paper's linear-loss formulation,
fidelity and Helstrom classifiers correspond to Hilbert–Schmidt and trace-distance
objectives, respectively. Those statements do not establish optimality for BCE
or for our restricted Pauli-Z readout. [Primary text, Sections II–III](https://arxiv.org/html/2001.03622v2).

Required components are labeled examples, a trainable data-dependent quantum
embedding, overlap estimation, and the corresponding readout. SWAP or inversion
tests estimate overlaps; hybrid classical adaptation is also illustrated. The
paper's low-rank class ensembles concern density operators, not the numerical
rank of a four-column classical latent matrix. [Primary text, Section III and Figure 5](https://arxiv.org/html/2001.03622v2).

## 2. Hur et al.: Neural Quantum Embedding

Tak Hur, Israel F. Araujo, and Daniel K. Park, *Neural Quantum Embedding: Pushing
the Limits of Quantum Supervised Learning*, Physical Review A **110**, 022411
(2024); arXiv:2311.11412v2 (9 August 2024).
Primary text: [Sections II.1–II.2, Appendix A.1 and Appendix C](https://arxiv.org/html/2311.11412v2).
Bibliographic record: [arXiv](https://arxiv.org/abs/2311.11412v2),
[publisher DOI](https://doi.org/10.1103/PhysRevA.110.022411).

NQE learns classical parameters w in |psi_w(x)>=V(g_w(x))|0>. Its implemented
pairwise fidelity surrogate, for labels y in {−1,+1}, is

\[
\ell_{\rm fid}(i,j)=
\left[|\langle\psi_w(x_i)|\psi_w(x_j)\rangle|^2
-\tfrac12(1+y_i y_j)\right]^2.
\]

Same-class pairs target overlap one; different-class pairs target zero.
The motivation uses a linear-risk bound

\[
L_s\ge\tfrac12-D_{\rm tr}(p^-\rho^-,p^+\rho^+),
\]

with empirical class priors p. The implemented surrogate is not the trace
distance itself. [Primary text, Equations 1 and 3](https://arxiv.org/html/2311.11412v2).

It requires a classical network, a differentiable quantum feature map, labeled
pairs, and overlap evaluation; the paper studies ZZ feature embeddings and
subsequent classifier training. Our single-example BCE is not this pairwise
objective. The bound concerns the stated linear risk, not a guarantee of lower
BCE, Brier, or validation error for our model. [Primary text, Section II.2 and Appendix A.1](https://arxiv.org/html/2311.11412v2).

## 3. Differences from our architecture: local analysis

The following deductions use `src/phase2/models.py` and the completed B protocol,
rather than attributing new claims to either paper.

| Aspect | Our B / proposed E0 | Consequence |
|---|---|---|
| Training target | B used single-example BCE jointly on encoder and VQC; E0 freezes B's encoder and refits heads. | E0 is a readout-compatibility study, not quantum metric learning or NQE training. |
| Quantum input map | Four angles pi*tanh(z/2), each encoded once by RY. | Sharing a classical-network/quantum-circuit pattern does not reproduce NQE. |
| Observable | A restricted trained-unitary Pauli-Z readout and fixed probability link. | Neither an optimal Helstrom readout nor a fidelity classifier is being fitted. |
| Available data | Labeled examples and saved four-dimensional embeddings. | No pairwise-fidelity training data flow or geometric objective currently exists. |
| Rank descriptors | Centered classical latent rank and variance entropy rank. | They must not be identified with class density-operator rank or model effective dimension. |

Write our input encoding as R(a), followed by the data-independent head U_theta.
For two examples with the encoder frozen,

\[
\langle\psi(a)|U_\theta^\dagger U_\theta|\psi(a')\rangle
=\langle\psi(a)|\psi(a')\rangle.
\]

Thus refitting this head changes the accessible measurement, while leaving pure-
state overlaps and class-ensemble trace/Hilbert–Schmidt distances invariant under
the common ideal unitary. It cannot optimize a fidelity-separation objective by
changing only those post-encoding head parameters. B's classical encoder could
change that geometry while it was trained; E0 deliberately holds it fixed.
This does not imply that a restricted readout can attain optimal discrimination.

For this particular single-layer product RY encoding, direct calculation gives

\[
K(a,a')=\prod_{j=1}^{4}\cos^2((a_j-a'_j)/2).
\]

This kernel is classically computable from the four angles. Adding the common
unitary head does not make its fidelity evaluation harder. Consequently an
adaptation of a pairwise fidelity objective to our existing encoding would
require explicit classical comparators; it could not establish quantum advantage
from the objective's name. This is an architecture-specific deduction, not a
claim about the embeddings studied in either paper.

## 4. Implication for the proposed sequence

E0.1 tests fitted readouts on fixed B geometry; E0.2 tests linear readability
before/after the fixed angular map. Neither tests whether a metric-learning
objective would improve the encoder. A later adaptation of either reviewed
objective would require a separately approved protocol specifying trainable
components, pair sampling and class-prior conventions, overlap calculation,
classical controls, and evaluation safeguards. These are requirements for a new
experiment, not a proposed definitive regularizer. E0.3, C0, and C1 remain
outside the present approval request.

See [E0 draft v1](e0_protocol_draft_v1.md) for the minimal proposed experimental
design and the limits of its conditional causal interpretation.
