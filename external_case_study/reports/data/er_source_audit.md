# Scientific audit of the TCGA-BRCA ER-status target

**Audit date:** 2026-10-07  
**GDC release:** 46.0 (2026-08-10)  
**Scope:** 1,098 TCGA-BRCA cases; open-access GDC clinical records  
**Status:** review required; the frozen Phase-1 cohort was not changed

## Conclusion

`gene_symbol=ESR1`, `molecular_analysis_method=IHC`, and a binary
`test_result` is a semantically valid GDC harmonization of an ER IHC result.
It is **not an adequate primary target definition when every matching
`follow_ups.molecular_tests` node is aggregated without clinical timepoint**.

The most specific official representation of the intended target is the
direct patient-level Clinical Supplement field
`breast_carcinoma_estrogen_receptor_status`, exposed in BCR Biotab as
`er_status_by_ihc` (CDE 2957359). It represents the baseline diagnostic ER
status and is distinct from `nte_er_status` / metastatic ER fields. The
recommended policy for review is therefore:

1. use the direct patient-level BCR XML field `er_status_by_ihc` as the primary
   target source;
2. use the BCR patient Biotab as an exact aggregate cross-check;
3. use harmonized ESR1/IHC nodes as secondary validation, not as an
   unrestricted case-level aggregator;
4. keep unknown/indeterminate/not-evaluated values missing; and
5. retain later-event ER values as longitudinal metadata, never as substitutes
   for a missing baseline target.

This recommendation has **not** been applied. All hashes under
`data/processed/` were identical before and after the audit.

## Documentary basis

GDC distinguishes indexed clinical data, which follow the searchable GDC Data
Dictionary, from additional TCGA clinical data stored in XML or Biotab
supplements. Its Clinical Supplement documentation specifically notes that
breast-cancer ER/PR/HER2 information may be present in these supplements
rather than the cross-cancer clinical model:

- [GDC Clinical Data](https://docs.gdc.cancer.gov/Encyclopedia/pages/Clinical_Data/)
- [GDC Clinical Supplement](https://docs.gdc.cancer.gov/Encyclopedia/pages/Clinical_Supplement/)

The GDC API exposes `cases.follow_ups.molecular_tests.gene_symbol`,
`molecular_analysis_method`, and `test_result`, confirming that the current
query uses supported harmonized fields:

- [GDC API available fields](https://docs.gdc.cancer.gov/API/Users_Guide/Appendix_A_Available_Fields/)

The GDC Data Dictionary release history documents the `molecular_test` entity
and the migration of legacy breast clinical fields such as
`estrogen_receptor_result_ihc` into properties including `gene_symbol`,
`molecular_analysis_method`, and `test_result`. This supports the semantic
identity of ESR1/IHC, but does not make all measurements exchangeable across
timepoints:

- [GDC Data Dictionary release notes](https://docs.gdc.cancer.gov/Data_Dictionary/Release_Notes/Data_Dictionary_Release_Notes/)
- [GDC Data Dictionary](https://docs.gdc.cancer.gov/Data_Dictionary/)

Clinical Supplement files are official GDC case-linked files with file UUIDs,
versions, and history:

- [GDC file submission model](https://docs.gdc.cancer.gov/API/Users_Guide/Submission/)

## Sources inspected

The release-scoped API query returned 1,183 Clinical Supplement files:

| Format | Files | Use in this audit |
|---|---:|---|
| BCR XML | 1,097 | Primary case-level field and longitudinal context |
| BCR Biotab | 9 | Patient-table cross-check; one file contains the baseline ER field |
| BCR OMF XML | 77 | Availability recorded; not used as a baseline target |

All 1,097 BCR XML files and the patient Biotab were downloaded from the GDC
data endpoint and checked against the GDC-provided size and MD5. One project
case has no released BCR XML; the patient Biotab still provides the aggregate
project row. The raw API request, response, GDC status, file IDs, sizes, MD5s,
and release are stored in the immutable audit snapshot.

The XML parser used namespace-independent local names and classified ER fields
by their path:

- baseline: direct child of `patient`,
  `breast_carcinoma_estrogen_receptor_status`;
- later event: `metastatic_breast_carcinoma_estrogen_receptor_status` under a
  follow-up or new-tumour event;
- direct patient template fields for new-tumour events were retained but not
  interpreted as baseline.

## Patient-level comparison

| Measure | Harmonized ESR1/IHC aggregate | Baseline XML | Patient Biotab |
|---|---:|---:|---:|
| Known binary ER | 1,038 (94.54%) | 1,046 (95.26%) | 1,046 (95.26%) |
| Missing/ambiguous/conflicting | 60 (5.46%) | 52 (4.74%) | 52 (4.74%) |
| ER positive among known | 801 | 808 | 808 |
| ER negative among known | 237 | 238 | 238 |

Pairwise harmonized-versus-baseline XML results across all 1,098 cases:

| Category | Cases |
|---|---:|
| Both known and concordant | 1,037 |
| Both known and discordant | 0 |
| XML baseline known only | 9 |
| Harmonized aggregate known only | 1 |
| Neither known | 51 |

Concordance among cases known in both sources was **1,037/1,037 (100%)**.
The XML and patient Biotab were also exactly concordant for all 1,046 known
baseline labels; both were unknown for the other 52 project cases. This shows
that the source systems agree when they refer to the same baseline field.

## Conflicts and temporal leakage of the target definition

All eight cases excluded by the frozen pipeline as `conflicting_er_results`
have a single known baseline XML result. In every case, the opposite value is
located in an XML follow-up or new-tumour/metastatic field.

| Case | Baseline ER | Other official ER observations |
|---|---|---|
| TCGA-A2-A0YJ | Positive | Negative follow-up |
| TCGA-A2-A3XU | Negative | Positive and Negative new-tumour events |
| TCGA-A7-A13E | Positive | Negative follow-up |
| TCGA-AO-A03N | Positive | Negative follow-up |
| TCGA-AO-A0JA | Positive | two Negative follow-ups |
| TCGA-B6-A402 | Negative | Negative and Positive new-tumour events |
| TCGA-E2-A10B | Positive | Negative follow-up |
| TCGA-E2-A152 | Positive | Negative new-tumour event |

There is one additional baseline-positive case, TCGA-A7-A0DC, with no
harmonized ESR1/IHC node. Thus the nine XML-only labels comprise seven ER+
and two ER− cases.

Conversely, TCGA-E9-A243 is ER `Not Evaluated` in both baseline XML and patient
Biotab, but has a later `nte_er_status=Negative` value that appears as a
harmonized ESR1/IHC molecular test. It is the only harmonized-only known case
and is currently included in the frozen cohort as ER−. Scientifically, that
node describes a later tumour event and should not substitute for the absent
baseline target.

## Frozen-cohort impact assessment

Within the current 1,036-patient frozen cohort:

- 1,035 labels agree with a known baseline XML label;
- 0 known baseline labels disagree;
- 1 case (TCGA-E9-A243) lacks a baseline label and is labeled only from a
  later event.

All nine XML-only baseline-known cases and the one harmonized-only case have
eligible RNA-seq. If the recommended baseline definition were approved and the
cohort rebuilt, the expected result would be 1,044 cases (807 ER+, 237 ER−):
nine additions and one removal. These numbers are an audit projection, **not a
cohort modification**.

## Limitations

- TCGA receptor status is retrospectively submitted clinical pathology data;
  this audit does not establish central laboratory re-review or uniform assay
  thresholds across tissue-source sites.
- The binary field does not capture percent-positive cells, assay scale, or
  changing clinical thresholds. Those fields remain available for sensitivity
  analyses but should not be used to impute a missing binary status silently.
- Fifty-two cases have no usable baseline binary value, including
  not-evaluated/indeterminate records and one case without a BCR XML file.
- Results are specific to GDC release 46.0. Reproduction against a later
  release must create a new release-scoped snapshot and rerun concordance.

## Reproduction and artifacts

```bash
python -m src.data.er_source_audit --config configs/data.yaml
# or
make audit-er
```

Outputs:

- `reports/data/er_source_audit.json` — machine-readable counts, case-level
  exceptions, hashes, release, endpoints, and projected-but-unapplied impact;
- `data/interim/er_source_comparison.parquet` — one row per project case;
- `data/interim/er_xml_observations.parquet` — every baseline and longitudinal
  XML ER observation with path/context;
- `data/raw/gdc_release_46.0/er_source_audit_b5df3c6ef16d/` — immutable query
  metadata, verified XML files, and verified patient Biotab.

The only pending scientific decision is whether to approve the baseline XML
field as the primary target and rebuild/version the cohort. Until that review,
Phase 1B should not treat the current label set as final.
