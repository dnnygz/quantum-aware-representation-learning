# Explore the saved A/B pilot

Open `01_representation_analysis.ipynb` from the repository root or `notebooks/`.
It reads the six aggregate CSV files in
`reports/phase2/pilot_fold0_50e_verified/` and renders convergence, gradient
norms, parameter gradients, angular saturation, the descriptive loss gap, and
the centered latent spectrum. It never trains, refits preprocessing, loads
checkpoints, or accesses cohort or locked-test data.

The [principal report](../reports/phase2/pilot_fold0_50e_verified/report.md)
contains the protocol, numerical results, conclusions, and historical provenance.
The pilot uses fold 0, 668 training and 167 validation patients, 50 epochs, and
seeds 1729/1730/1731. Seed variability is descriptive, not a confidence interval.

## Install and execute

```bash
uv sync --locked --extra analysis --extra notebooks
uv run --extra analysis --extra notebooks jupyter nbconvert \
  --execute --to notebook --ExecutePreprocessor.timeout=180 \
  --output 01_representation_analysis.executed.ipynb \
  --output-dir runs/phase2/notebook_review \
  notebooks/01_representation_analysis.ipynb
```

Choose an unused output directory for each executed copy. Notebook outputs and
PNG/SVG exports remain local under ignored directories. Shared plotting functions
are in `src/phase2/analysis_views.py`.

`QARL_ANALYSIS_DIR` optionally selects another saved summary directory, relative
to the project root or absolute. Set `QARL_PROJECT_ROOT` when executing outside
the repository. Missing CSV files produce an informative error; no replacement
results are generated.

## Saved inputs

| CSV | Contents |
|---|---|
| `metrics_by_seed.csv` | Aggregate metrics, confusion matrices, and costs |
| `descriptive_summary.csv` | Model means and sample standard deviations |
| `paired_comparisons.csv` | Metric differences and score correlations |
| `learning_curves.csv` | Epoch losses, gradients, and saturation |
| `latent_spectrum.csv` | Singular values and numerical ranks |
| `angular_diagnostics.csv` | Coordinate and pooled angular statistics |

These files contain no patient IDs, individual predictions, or individual latent
vectors. They support everyday visualization without the private run archive.
Reading summaries is not independent verification of their underlying experiment.

## Separate deep verification

To reconcile all six tables and their recorded source hashes with the original
**development** archive, run from the repository root:

```bash
uv run --extra analysis python -c "from src.phase2.analysis_artifacts import load_analysis_artifacts; print(load_analysis_artifacts('runs/phase2/pilot_fold0_50e_paired_v1', 'reports/phase2/pilot_fold0_50e_verified')['checks'])"
```

This requires saved configuration, provenance, resources, representation split
metadata, validation predictions, curve CSVs, and latent NPZ files. It checks
patient alignment, metrics, confusion matrices, shapes, and table agreement at
1e-12 tolerance without loading checkpoints or fitting models.

Exact score scatterplots and angular histograms require private predictions and
individual latent arrays. The existing analysis CLI also checks saved model
states and reads historical preflight/smoke diagnostics. To regenerate the full
analysis into a **new** directory, without training:

```bash
uv run --extra quantum --extra analysis python -m src.phase2.analyze_pilot \
  --run runs/phase2/pilot_fold0_50e_paired_v1 \
  --output runs/phase2/pilot_analysis_review \
  --preflight runs/phase2/preflight_pilot_50e/diagnostics.json \
  --smoke runs/phase2/smoke_verified_1729
```

Do not run the training or diagnostic preflight CLI merely to view figures.
Historical review records, source snapshots, and images remain local; the
withdrawn presentation report and previous guide are preserved under
`runs/phase2/reorganization_archive_20261028/`.
