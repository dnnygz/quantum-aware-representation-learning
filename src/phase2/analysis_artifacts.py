"""Reconstruction and deep verification of saved development pilot outputs.

No training, checkpoint loading, or cohort access. Everyday visualization reads
aggregate CSVs directly and does not depend on this verification module.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from src.data.utils import sha256_file
from .evaluation import metrics
from .analysis_views import MODELS, MEASURES, read_csv, paired_predictions
from .analysis_views import summarize_metrics, spectrum_values, angular_values, paired_values

TABLE_KEYS = {
    'metrics_by_seed': ['experiment','seed'],
    'descriptive_summary': ['experiment'],
    'paired_comparisons': ['seed'],
    'learning_curves': ['experiment','seed','epoch'],
    'latent_spectrum': ['experiment','seed','checkpoint','split'],
    'angular_diagnostics': ['experiment','seed','checkpoint','split','coordinate'],
}


class ArtifactError(ValueError):
    """Missing, inconsistent or unsupported analysis input; never auto-regenerate."""


def _require(paths):
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise ArtifactError('Required saved artifacts are missing:\n- '+'\n- '.join(missing)+
            '\nSet QARL_RUN_DIR / QARL_ANALYSIS_DIR. Nothing will be trained or reconstructed automatically. '
            'If derived artifacts are missing, use the official analysis command with a new output directory; '
            'if original artifacts are missing, restore the run archive.')


def _compare(name, recomputed, saved):
    keys = TABLE_KEYS[name]
    if recomputed.duplicated(keys).any() or saved.duplicated(keys).any():
        raise ArtifactError(f'{name}: duplicate keys')
    def ordered(frame):
        value = frame.copy()
        if 'coordinate' in value:
            value['coordinate'] = value.coordinate.astype(str)
        return value.sort_values(keys).reset_index(drop=True)
    if set(saved.columns) != set(recomputed.columns):
        raise ArtifactError(f'{name}: columns do not match the official calculation')
    try:
        assert_frame_equal(ordered(recomputed)[saved.columns],ordered(saved),check_dtype=False,
                           check_exact=False,atol=1e-12,rtol=1e-12)
    except AssertionError as error:
        raise ArtifactError(f'{name}: the calculation does not match the saved official table. {error}') from error


def load_analysis_artifacts(run_dir, analysis_dir=None):
    """Reconstruct the saved pilot; optionally reconcile an existing analysis.

    This deep verification is separate from everyday CSV-based visualization.
    It never fits models or loads checkpoints.
    """
    run = Path(run_dir).expanduser().resolve()
    analysis = Path(analysis_dir).expanduser().resolve() if analysis_dir is not None else None
    _require([run/'config.json',run/'provenance.json',run/'comparison.csv',run/'resources.json'])
    config = json.loads((run/'config.json').read_text())
    provenance = json.loads((run/'provenance.json').read_text())
    if (config.get('folds') != [0] or config.get('epochs') != 50
            or config.get('seeds') != [1729,1730,1731] or config.get('experiments') != ['A','B']
            or provenance.get('locked_test_evaluated') is not False
            or provenance.get('stage') not in ['development_cv','development_pilot']):
        raise ArtifactError('Only the development A/B pilot is supported: fold 0, 50 epochs, and three seeds; '
                            'no locked test or implicit future experiments.')
    required = ([analysis/f'{name}.csv' for name in TABLE_KEYS] +
                [analysis/'analysis_provenance.json']) if analysis is not None else []
    source_files = [run/'config.json',run/'comparison.csv']
    for name in MODELS:
        base = run/f'fold_0_{"A" if name.startswith("A") else "B"}'
        for seed in config['seeds']:
            folder = base/f'{name}_seed_{seed}'
            source_files.append(folder/'validation_predictions.csv')
            if name in ['A','B','B_classical']:
                source_files += [folder/'curves.csv',folder/'latent_diagnostics.npz']
        required.append(base/'representation.json')
    _require(required+source_files)
    if analysis is not None:
        manifest = json.loads((analysis/'analysis_provenance.json').read_text())
        # Rebase known source paths onto the configurable run root. Never follow
        # arbitrary paths from a provenance manifest (e.g. raw or locked artifacts).
        for path in source_files:
            key = (Path(manifest['run'])/path.relative_to(run)).as_posix()
            expected = manifest.get('source_sha256',{}).get(key)
            if expected is None or sha256_file(path) != expected:
                raise ArtifactError(f'Provenance/hash mismatch: {path}; use the analysis corresponding to this run.')
    table = read_csv(run/'comparison.csv')
    expected_keys = {(name, seed) for name in MODELS for seed in config['seeds']}
    if len(table) != 15 or set(zip(table.experiment,table.seed)) != expected_keys or not (table.fold==0).all():
        raise ArtifactError('comparison.csv: expected the 15 unique rows of the fold-0 pilot')
    curves, spectra, angles, paired = [], [], [], []
    reference_ids, reference_labels = None, None
    for seed in config['seeds']:
        predictions = {}
        for name in MODELS:
            base = run/f'fold_0_{"A" if name.startswith("A") else "B"}'
            representation = json.loads((base/'representation.json').read_text())
            train_ids, validation_ids = representation['train_ids'],representation['validation_ids']
            if len(train_ids)!=668 or len(validation_ids)!=167 or set(train_ids)&set(validation_ids):
                raise ArtifactError('Invalid training/validation partition in the saved representation')
            folder = base/f'{name}_seed_{seed}'
            prediction = read_csv(folder/'validation_predictions.csv')
            if prediction.case_id.duplicated().any() or set(prediction.case_id) != set(validation_ids):
                raise ArtifactError(f'{folder}: prediction IDs do not match validation IDs')
            aligned = prediction.set_index('case_id').sort_index()
            if reference_ids is None:
                reference_ids, reference_labels = aligned.index, aligned.er_label
            if not aligned.index.equals(reference_ids) or not aligned.er_label.equals(reference_labels):
                raise ArtifactError('Patients or labels differ across models/seeds')
            values = metrics(prediction.er_label.to_numpy(),prediction.probability_er_positive.to_numpy())
            row = table[(table.experiment==name)&(table.seed==seed)].iloc[0]
            for measure in MEASURES:
                if not np.isclose(values[measure],row[measure],atol=1e-12,rtol=0):
                    raise ArtifactError(f'Metric could not be reproduced: {name}/{seed}/{measure}')
            if json.loads(row.confusion_matrix) != values['confusion_matrix']:
                raise ArtifactError(f'Confusion matrix could not be reproduced: {name}/{seed}')
            predictions[name] = prediction
            if name not in ['A','B','B_classical']:
                continue
            curve = read_csv(folder/'curves.csv')
            if list(curve.epoch)!=list(range(1,51)) or not np.isfinite(curve.to_numpy()).all():
                raise ArtifactError(f'{folder}: incomplete or nonfinite learning curve')
            curves.append(curve.assign(experiment=name,seed=seed))
            with np.load(folder/'latent_diagnostics.npz',allow_pickle=False) as latent:
                for checkpoint in ['initial','final']:
                    for split,n in [('train',668),('validation',167)]:
                        z = latent[f'{checkpoint}_{split}']
                        if z.shape!=(n,4) or not np.isfinite(z).all():
                            raise ArtifactError(f'{folder}: invalid {checkpoint}/{split} latent representation')
                        spectra.append(dict(experiment=name,seed=seed,checkpoint=checkpoint,
                                            split=split,**spectrum_values(z)))
                        for coordinate in [None,0,1,2,3]:
                            values = z if coordinate is None else z[:,coordinate:coordinate+1]
                            angles.append(dict(experiment=name,seed=seed,checkpoint=checkpoint,split=split,
                                coordinate='pooled' if coordinate is None else coordinate,**angular_values(values)))
        joined,rho = paired_predictions(predictions['B'],predictions['B_frozen_probe'])
        paired.append(paired_values(table[table.seed==seed].set_index('experiment'),seed,rho,len(joined),MEASURES))
    tables = dict(metrics_by_seed=table,
                  descriptive_summary=summarize_metrics(table,MEASURES).reset_index(),
                  paired_comparisons=pd.DataFrame(paired),learning_curves=pd.concat(curves,ignore_index=True),
                  latent_spectrum=pd.DataFrame(spectra),angular_diagnostics=pd.DataFrame(angles))
    checks = []
    if analysis is not None:
        for name, computed in tables.items():
            _compare(name, computed, read_csv(analysis/f'{name}.csv'))
            checks.append(dict(artifact=name, rows=len(computed), official_calculation_matches=True))
    return dict(config=config, tables=tables,
                resources=json.loads((run/'resources.json').read_text()), checks=pd.DataFrame(checks))
