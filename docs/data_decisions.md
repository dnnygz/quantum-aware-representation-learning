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

## Known limitations / pending scientific choices

1. Baseline IHC values are site-submitted TCGA clinical data and do not imply a
   centralized pathology re-review or uniform historical positivity threshold.
2. Barcode-order sample selection is neutral and reproducible but not based on
   RNA quality metrics, which are not uniformly exposed with these count files.
3. Whether downstream models should use a count-specific transform, TPM, or a
   carefully fitted normalization method is intentionally deferred until the
   modeling protocol is defined inside training folds.
