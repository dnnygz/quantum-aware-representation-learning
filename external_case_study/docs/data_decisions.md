# Phase 1 scientific and technical decisions

This document records decisions that materially affect the cohort. Runtime
counts and file lists belong in generated reports, not in this document.

## Authoritative sources

The pipeline uses only official GDC services:

- [GDC API search and retrieval](https://docs.gdc.cancer.gov/API/Users_Guide/Search_and_Retrieval/)
- [GDC file download and manifest endpoints](https://docs.gdc.cancer.gov/API/Users_Guide/Downloading_Files/)
- [GDC mRNA analysis pipeline](https://docs.gdc.cancer.gov/Data/Bioinformatics_Pipelines/Expression_mRNA_Pipeline/)
- [GDC STAR Counts definition](https://docs.gdc.cancer.gov/Encyclopedia/pages/STAR_Counts/)
- live `/status`, `/files`, `/cases`, and `/data/{file_id}` endpoints

The API status response is saved with the snapshot. This is the release anchor;
individual files also retain `data_release`, timestamps, UUID, size and MD5.

## ER status source

### Representations considered

1. **Harmonized GDC case graph:** ESR1 molecular-test nodes under
   `follow_ups.molecular_tests`, typically IHC with `Positive`, `Negative`, or
   `Equivocal` results.
2. **Per-case Clinical Supplement XML:** historical submitted clinical files
   can contain breast-specific receptor fields. They are useful for source
   archaeology but require schema-specific XML parsing and can include multiple
   versions/files per case.
3. **Expression-derived ESR1 or molecular subtype:** rejected as a label source.
   It would redefine a clinical IHC outcome from the predictors and create a
   circular target.

### Superseded decision: harmonized ESR1/IHC aggregation

The original Phase-1 cohort used every harmonized GDC molecular-test node with
`gene_symbol=ESR1` and `molecular_analysis_method=IHC`, excluding cases when
known results conflicted. This definition is preserved as
`v0_harmonized_esr1_ihc` and is no longer canonical.

The targeted source audit found 100% agreement when the harmonized and baseline
sources were both known, but also showed that unrestricted aggregation mixed
the baseline result with ER measurements from recurrence, metastasis or new
tumour events. Eight apparently conflicting cases had a valid baseline ER, and
one patient with baseline ER not evaluated had been labeled from a later event.
The evidence is retained in `reports/data/er_source_audit.md` and its
machine-readable artifacts.

### Approved decision: canonical v1 baseline ER

On 2026-10-07 the primary target was changed to the baseline Clinical
Supplement field `er_status_by_ihc` /
`breast_carcinoma_estrogen_receptor_status` (CDE 2957359). This directly matches
the scientific estimand: **baseline ER status of the primary tumour**.

The source hierarchy is now:

1. per-case BCR Clinical Supplement XML direct patient field as ground truth;
2. BCR patient Biotab `er_status_by_ihc` as a mandatory consistency cross-check;
3. harmonized GDC ESR1/IHC molecular tests as secondary validation only.

Positive maps to 1 and Negative to 0. Missing, not evaluated, indeterminate,
equivocal or unsupported baseline values remain missing. ER measurements from
follow-up, recurrence, metastasis or new tumour events must never fill a missing
baseline result. XML/Biotab disagreement is a fail-fast integrity error rather
than a precedence decision.

The prior cohort and its original provenance, reports, splits and SHA-256
hashes are retained under
`data/processed/versions/superseded/v0_harmonized_esr1_ihc/`. The canonical v1
provenance links both the audit and the superseded manifest. Patient-level
entries, exits and any label changes are recorded in
`reports/data/old_cohort_vs_baseline_er_cohort.json`.

## Expression measure

The processed matrix uses raw `unstranded` counts. GDC states that RNA-seq reads
are treated as unstranded for harmonized analysis and annotates current STAR
outputs with GENCODE v36. Raw counts preserve flexibility for a later,
fold-local modeling pipeline. `stranded_first` and `stranded_second` are not
selected because they conflict with the GDC harmonization convention;
TPM/FPKM/FPKM-UQ are not selected because this phase should not commit the
modeling pipeline to a pre-normalized measure.

The four `N_*` STAR summary rows are QC counters, not genes, and therefore do
not enter the feature matrix. They are parsed separately.

## Multiple samples and aliquots

The experimental unit is the case. All eligible files remain in the metadata
snapshot. One file is selected per case by the configured stable ordering of
sample barcode, numeric portion, aliquot barcode, sample UUID and file UUID.
For TCGA barcodes this normally prefers vial `A` to later vials. This rule:

- is deterministic;
- does not use ER or expression values;
- prevents one patient's samples from crossing model partitions;
- is fully represented in `sample_selection_audit.parquet`.

It does not claim biological superiority. A later, preregistered sensitivity
analysis can compare aggregation or quality-metric-based selection without
changing the primary cohort silently.

## Leakage boundary

Allowed in Phase 1: identity checks, schema checks, finite-value checks, class
counts, per-sample library-size summaries, deterministic sample selection and
patient-level split generation.

Deferred to fold-local training pipelines: gene filtering informed by training
data, normalization with cohort-estimated statistics, scaling, PCA, mutual
information, ANOVA, LASSO and all outcome-guided feature selection.

## Phase 1B primary RNA-seq preprocessing specification

Approved on 2026-10-07, the primary preprocessing sequence is:

`training-fold CPM filter -> training-fitted TMM -> log2-CPM with prior count -> exact-zero-variance removal -> estimator-specific scaling`.

The expression filter retains a gene when it has `CPM >= 1` in at least 10% of
the training patients. This is a prespecified transcriptomic criterion, not a
threshold selected from ER labels or predictive performance. Alternative CPM
or prevalence values may later be supplied declaratively for sensitivity
analysis, but choosing among them using model performance requires nested
training-only validation.

Every operation that depends on the patient distribution follows
`fit(training) / transform(validation or test)`. The gene mask, TMM reference
and geometric-mean calibration, exact-zero-variance mask, and any
model-specific scaler are learned without validation/test patients. A new
patient receives a TMM factor independently against the frozen training
reference; test patients are never jointly recentred. No global normalized
matrix is a valid modeling input.

TMM uses the Robinson--Oshlack weighted trimmed mean with edgeR-compatible
defaults: 30% log-ratio trimming, 5% absolute-expression trimming and a
training-selected reference whose upper-quartile expression is closest to the
training mean. Log2-CPM uses prior count 2, scaled in proportion to the
effective library size. Remaining zeros are not imputed. Only features with
exactly zero transformed variance in training are removed; no top-variance or
near-zero-variance feature budget is imposed.

Scaling is deliberately outside the shared count transform. Linear models and
SVMs may fit a `StandardScaler` on their training partition, tree models may
omit it, and a later VQC encoding scaler must be learned only after fold-local
feature selection. PCA, MI, ANOVA, LASSO, model fitting, and reduction to
4/6/8 features are outside Phase 1B.

## Phase 1C validation boundary

The original `stratified_kfold` entry in the canonical Phase-1 split file spans
all patients and is retained for provenance, but it is not valid for tuning
when the canonical 20% holdout is treated as a locked final test. Phase 1C
therefore creates a separate, versioned five-fold stratified split using only
the canonical development patient IDs. It does not rewrite the canonical split
artifact.

These development-only folds are the sole partitions authorized for
preprocessing fitting, hyperparameter comparison and model selection. The
locked test is evaluated once after the full pipeline and hyperparameters have
been frozen. The protocol, search spaces, default classification thresholds
and bootstrap procedure are preregistered in `docs/phase1c_protocol.md` before
any classifier is trained.

## Known limitations / pending scientific choices

1. Baseline IHC values are site-submitted TCGA clinical data and do not imply a
   centralized pathology re-review or uniform historical positivity threshold.
2. Barcode-order sample selection is neutral and reproducible but not based on
   RNA quality metrics, which are not uniformly exposed with these count files.
3. TMM assumes that most genes are not dominated by unbalanced global shifts.
   Phase-1B QC therefore reports factor distributions, robust outliers and MA
   diagnostics, but test-set diagnostics must never change the primary
   specification.

## Phase 1C classical reference decision

Phase 1C was executed on 2026-10-08 under the approved frozen protocol. Model
families and regularization values were selected only by mean ROC-AUC across
the five development-only folds. Logistic L1 at `C=0.01` had the largest mean
CV ROC-AUC and therefore defines `P_classical-full`; this decision predates
locked-test access. The numerically highest test ROC-AUC belongs to RBF-SVM by
only about 0.001, with overlapping uncertainty, and does not override the
predeclared selection rule.

The selected L1 model is extremely sparse, but its nonzero coefficients are a
property of a supervised fitted classifier. They must not be treated as a
preapproved 4/6/8-gene feature selector or reused outside a properly nested
Phase-2 selection pipeline.

The final report was recovered from atomic checkpoints after a post-inference
Parquet type error in the sparsity audit. No second test inference occurred.
The recovery, missing final per-step timings, repaired type, and one-use access
receipt are recorded in Phase-1C provenance and the durable results report.

## Phase 1C-XGB post-hoc complementary baseline

XGBoost was added only after the four original locked-test results had been
observed. It is permanently labelled `post-hoc complementary classical
baseline` and cannot redefine `P_classical-full(primary)`, which remains
Logistic L1 with locked-test ROC-AUC 0.9660. Six configurations and their
strict argmax selection rule were frozen before fitting; a complexity
tie-break is allowed only for numerical equality within `1e-12`.

The experiment reused the original development/test patients, inner folds and
5,000 bootstrap samples. It omitted StandardScaler, used fold-local balanced
sample weights, and performed one locked-test inference after selection. The
selected C06 configuration and complete audit are documented in
`docs/xgboost_posthoc_protocol.md`. Its tree importance is descriptive only
and is prohibited as a Phase-2 or VQC feature selector.

## ER class and metric semantics

The project-wide encoding is `ER_positive = 1` and `ER_negative = 0`. ER+ is
the positive class for Phase-1C model scores, ROC-AUC and the frozen legacy
`pr_auc` field. The legacy keys therefore mean:

- `pr_auc`: ER+ PR-AUC / average precision;
- `sensitivity`: ER+ sensitivity / recall;
- `specificity`: ER− specificity / recall;
- confusion order `[ER_negative=0, ER_positive=1]`: `(TN, FP, FN, TP)`.

Generic keys remain only in frozen Phase-1C artifacts for hash stability. New
reports and future phases must name the class explicitly and report ER+ and
ER− average precision with their prevalence baselines. ROC-AUC remains the
unchanged primary model-selection metric.
