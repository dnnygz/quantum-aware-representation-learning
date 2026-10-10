# Phase 1 — Data, Preprocessing and Classical Reference Baselines: FINAL FREEZE

Status: **FINAL_FREEZE_PASS**  
Generated: `2026-10-10T01:29:40.449564+00:00`

## Frozen scientific definition

- **Phase 1A:** TCGA-BRCA canonical v1 contains 1044 patients (807 ER+, 237 ER−) and 60660 raw STAR-count genes. Baseline Clinical Supplement IHC is the primary target; `ER+=1`, `ER−=0`.
- **Phase 1B:** `CPM>=1 in >=10% training -> training-fitted TMM -> log2-CPM prior=2 -> exact training zero variance -> estimator-specific scaling`. All patient-distribution-dependent parameters are training-fitted; no global normalized modeling matrix exists.
- **Phase 1C:** 835 development patients, five frozen inner folds, 209 locked-test patients. The preregistered primary reference remains Logistic L1: ROC-AUC 0.9660 (95% CI 0.9372–0.9896).
- **Semantic audit:** ER+/ER− meanings and confusion order were reconciled from persisted predictions with no discrepancies and no refit/inference.
- **Phase 1C-XGB:** explicitly post-hoc and complementary. Strict CV argmax selected `xgb_c06_shallow_faster` (mean ROC-AUC 0.96316); locked-test ROC-AUC was 0.97413 (95% CI 0.95088–0.99265).

## XGBoost interpretation

Named confusion matrix: `{"true_ER_negative_predicted_ER_negative_TN": 46, "true_ER_negative_predicted_ER_positive_FP": 1, "true_ER_positive_predicted_ER_negative_FN": 11, "true_ER_positive_predicted_ER_positive_TP": 151}`.
The point ROC-AUC difference versus Logistic L1 is 0.00814; its paired-bootstrap 95% interval is -0.00066 to 0.01878. The interval crosses zero. XGBoost therefore does **not** replace `P_classical-full(primary)=0.9660`.

## Integrity and leakage gate

All 12 freeze checks passed. The XGBoost receipt records exactly one locked-test inference; all original Phase-1C hashes match their pre-XGBoost values; the same 5,000 paired bootstrap indices were reused; no StandardScaler or supervised feature selection entered the XGBoost pipeline. Feature importance is descriptive only and prohibited for Phase 2/VQC selection.

Tests: `.venv/bin/pytest -q` → **77 passed, 0 failed**.

## Versions

`{"PyYAML": "6.0.3", "numpy": "2.5.3", "pandas": "2.3.3", "pyarrow": "21.0.0", "python": "3.12.11", "scikit-learn": "1.9.1", "xgboost": "3.4.1"}`  
Git: `{"commit": "996f02a6b1ce450c6b5dac6f9bfb62f7ce29b9a7", "dirty": true}`

## Reconstruction hashes (SHA-256)

- `configs/data.yaml`: `ddd761050b279e88b7ea6691dd642f22535716c38ce535c5cfe7d66e56617fe9`
- `configs/label_semantics.yaml`: `bda3ed38ad2636c6a798d573d867a2996db92a0e20aedd570f3528cc5a480254`
- `configs/xgboost_posthoc.yaml`: `2faf00095011b1e33b055e71514277426a5cff8f1d85c71d3eed5d721fd2fc01`
- `data/processed/expression.parquet`: `08f4166ccc2f23fb95ad235d15f9fdbb274e2d5441a7c20cdebe005063874069`
- `data/processed/genes.parquet`: `c1d8b0d7d7aed08540fbc09e3713bf4efaced7cc036735bb15b623426c4984f0`
- `data/processed/labels.parquet`: `5d18840a4b3f0fe57fbdd1d2e255d4a5c21f4b3fb23131b563f13c6ae408fe69`
- `data/processed/metadata.parquet`: `14562116d29c1049619ce7db67db1619d37faed667053de63be3b6797709878c`
- `data/processed/splits.json`: `167e716ef018552d8f29fcf2976d4df12f718b2841e9d9b82ec884f24bad3523`
- `data/processed/splits/phase1c_inner_cv.json`: `2d58fa5bdf9d7e95875b099c29bcea8fc9fc0b987fb9cc98fc2d10a386b57ff3`
- `data/processed/phase1c_classical_baseline_v1/bootstrap_indices.npz`: `f68d06b2d62321bc935bca91810ac4a7d3b40efad2914ba0b2dce5f60f7386b6`
- `data/processed/phase1c_classical_baseline_v1/bootstrap_patient_order.parquet`: `0008b5af19d3460b2f06bb283ecc6239b6925c7bf6d26f85c28fa8f81df4b37e`
- `data/processed/phase1c_classical_baseline_v1/oof_predictions.parquet`: `52fd35fa4eadb551cb2ab441c9f5918e974ec412d0374a9ba8c9c17806b2c550`
- `data/processed/phase1c_classical_baseline_v1/locked_test_predictions.parquet`: `f6dadd12d3dcd60fd57faf0356abbb80981c52102e8c0f0b5228cec48e6042a8`
- `data/processed/xgboost_posthoc_v1/oof_predictions.parquet`: `4a60c0182342767d82dd93010104e26caa2baf5bb1b9b49dd9e10324a6e688c9`
- `data/processed/xgboost_posthoc_v1/locked_test_predictions.parquet`: `f5e61fe5ec2963643fdcbd7e0bf5c12b317ed14cea48498d44928c176855e650`
- `data/interim/xgboost_posthoc_v1/xgboost_model.json`: `b52aa571ec1a37bf18947cd0656aaa24a154df71ac766ae417de8ed4016f7838`
- `reports/data/cohort_summary.json`: `e3ee0aabbbd05bdcec08e1a053167e6f63e2b8982a79a24c8a03d4ee615e6061`
- `reports/data/provenance.json`: `a4ed6d0db26027831857e0d64b7c8ab21231b91751bb651402fa6502fc707453`
- `reports/data/preprocessing_provenance.json`: `b43835467b2d6e28b998f0e0cfb41a0121b32233d27970113e4db88d457acf0e`
- `reports/data/phase1c_provenance.json`: `4ee9c8142fa574caa634e8a7dd890e307623b2ed92306ff005a8b12ded59ad7b`
- `reports/data/phase1c_label_metric_semantic_audit.json`: `ed35ef8d9e1c39663cc267a706ecda19b8db717dd52b4cfc5b73c8608d483d47`
- `reports/data/xgboost_posthoc_protocol_gate.json`: `04dab1f6f8576cd6e6076298a1a7b064ba5d207425951770c34b848656419178`
- `reports/data/xgboost_posthoc_selected_configuration.json`: `ad66ffb0a92a0eedc0fcbc4a71736f758fc97e32f7ef58a474627a5a87941511`
- `reports/data/xgboost_posthoc_locked_test_performance.json`: `bb78606972a5429419e9324089295e6743598de54f59462ed9cb0e41d8903b7b`
- `reports/data/xgboost_posthoc_leakage_qc.json`: `2420e2bd981ed8806a4e25704e07f80dda380585337d4a6a2a6d53281f674289`
- `reports/data/xgboost_posthoc_provenance.json`: `9fd27b7160c68dd135b1676dda6677659af74e347f8584f3daa2c0f125df7390`

## Boundary after freeze

Phase 2 has not started. PCA, MI, ANOVA, LASSO bottleneck and VQC remain outside this freeze. The fold-local feature-bottleneck protocol requires a separate review and approval.
