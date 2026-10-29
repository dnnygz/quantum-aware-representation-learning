"""Read-only score diagnostics for the C1 decision; no model or loss fitting."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import metrics


def score_diagnostics(y, p):
    """Describe ranking, fixed-threshold errors, and probability tails separately."""
    y, p = np.asarray(y), np.asarray(p)
    if y.shape != p.shape or set(np.unique(y)) != {0, 1}:
        raise ValueError('Aligned binary labels with both classes are required')
    if not np.isfinite(p).all() or np.any((p <= 0) | (p >= 1)):
        raise ValueError('Finite probabilities strictly inside (0, 1) are required')
    loss = -np.where(y == 1, np.log(p), np.log1p(-p))
    correct = (p >= .5) == y
    pair = p[y == 1, None] - p[None, y == 0]
    result = {k: v for k, v in metrics(y, p).items() if k != 'confusion_matrix'}
    result.update(brier=float(np.mean((p-y)**2)),
                  mean_contrast=float(p[y == 1].mean()-p[y == 0].mean()),
                  reversed_pair_fraction=float(np.mean(pair < 0)),
                  tied_pair_fraction=float(np.mean(pair == 0)),
                  probability_bias=float(p.mean()-y.mean()),
                  wrong_confident_fraction=float(np.mean(np.where(y == 1, p <= .1, p >= .9))),
                  top_10pct_loss_share=float(np.sort(loss)[-int(np.ceil(.1*len(y))):].sum()/loss.sum()))
    for c in (0, 1):
        pc, lc = p[y == c], loss[y == c]
        result.update({f'class_{c}_{k}': float(v) for k, v in dict(
            mean_probability=pc.mean(), sd_probability=pc.std(),
            q05=np.quantile(pc, .05), q50=np.quantile(pc, .5), q95=np.quantile(pc, .95),
            log_loss=lc.mean(), sensitivity=correct[y == c].mean(),
            wrong_confident=np.mean(pc >= .9) if c == 0 else np.mean(pc <= .1)).items()})
    assert abs(result['roc_auc']-(1-result['reversed_pair_fraction']-.5*result['tied_pair_fraction'])) < 1e-12
    return result


def paired_diagnostics(y, vqc, probe):
    """Partition paired ranking and loss differences without fitting a link."""
    vc, pc = (vqc >= .5) == y, (probe >= .5) == y
    vl = -np.where(y == 1, np.log(vqc), np.log1p(-vqc))
    pl = -np.where(y == 1, np.log(probe), np.log1p(-probe))
    vs = np.sign(vqc[y == 1, None]-vqc[None, y == 0])
    ps = np.sign(probe[y == 1, None]-probe[None, y == 0])
    out = dict(lr_correct_vqc_reversed=float(np.mean((ps > 0) & (vs < 0))),
               vqc_correct_lr_reversed=float(np.mean((vs > 0) & (ps < 0))),
               both_reversed=float(np.mean((vs < 0) & (ps < 0))),
               delta_auc=float(np.mean((ps-vs)/2)), delta_loss=float(np.mean(pl-vl)))
    for name, mask in dict(both_correct=vc & pc, both_wrong=~vc & ~pc,
                           lr_only_correct=~vc & pc, vqc_only_correct=vc & ~pc).items():
        out[name+'_fraction'] = float(mask.mean())
        out[name+'_loss_contribution'] = float(np.sum((pl-vl)[mask])/len(y))
    assert abs(sum(v for k, v in out.items() if k.endswith('_loss_contribution'))-out['delta_loss']) < 1e-12
    return out


def audit(output):
    """Audit only allowlisted, hashed development prediction artifacts."""
    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a new output directory; existing reports are preserved')
    accessed = {}

    def checked(path, expected=None):
        path = Path(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected is not None and digest != expected:
            raise ValueError(f'Provenance mismatch: {path}')
        accessed[str(path)] = digest
        return path

    cv = json.loads(checked('reports/phase2/cv_5fold_50e_v1/provenance.json').read_text())
    split = json.loads(checked('data/processed/splits/phase1c_inner_cv.json', cv['protocol_sha256']).read_text())
    dev, locked = set(split['development_case_ids']), set(split['locked_test_case_ids'])
    if len(dev) != 835 or len(locked) != 209 or dev & locked:
        raise ValueError('Invalid development/locked partition metadata')
    labels = {}
    for f in split['folds']:
        root = 'pilot_fold0_50e_paired_v1' if f['fold'] == 0 else 'cv_folds1to4_50e_paired_v1'
        path = f"runs/phase2/{root}/fold_{f['fold']}_B/B_seed_1729/validation_predictions.csv"
        frame = pd.read_csv(checked(path, cv['sources'][path]), float_precision='round_trip')
        if frame.case_id.duplicated().any() or set(frame.case_id) != set(f['validation_case_ids']):
            raise ValueError('Invalid development label source')
        for row in frame.itertuples():
            if row.case_id in labels or row.case_id not in dev or row.er_label not in (0, 1):
                raise ValueError('Invalid or duplicate development label')
            labels[row.case_id] = int(row.er_label)
    if set(labels) != dev:
        raise ValueError('Incomplete development coverage')
    cp = json.loads(checked('reports/phase2/c0_v1_results/provenance.json').read_text())
    manifest = json.loads(checked('runs/phase2/c0_v1_final_verification/run_artifact_hashes.json',
                                  cp['run_hash_manifest_sha256']).read_text())
    run = Path('runs/phase2/c0_v1_full_cv')
    reference = pd.read_csv(checked(run/'metrics.csv', manifest['metrics.csv']))
    rows, pairs, bins = [], [], []
    max_error = 0.
    for f in split['folds']:
        for s in (1729, 1730, 1731):
            for arm in ('C0-NQE-Q', 'C0-NQE-E', 'C0-Hybrid-Q', 'C0-Hybrid-E'):
                folder = f"fold_{f['fold']}_seed_{s}_{arm}"
                paths = [f'{folder}/{n}.npz' for n in ('latent_and_predictions', 'diagnostic_predictions')]
                with np.load(checked(run/paths[0], manifest[paths[0]]), allow_pickle=False) as primary, np.load(
                        checked(run/paths[1], manifest[paths[1]]), allow_pickle=False) as diagnostic:
                    for partition in ('train', 'validation'):
                        ids = f['train_case_ids' if partition == 'train' else 'validation_case_ids']
                        if len(ids) != len(set(ids)) or not set(ids) <= dev or set(ids) & locked:
                            raise ValueError('Non-development prediction scope')
                        y = np.array([labels[i] for i in ids])
                        if (len(y), int(y.sum())) != ((668, 516) if partition == 'train' else (167, 129)):
                            raise ValueError('Unexpected class counts')
                        predictions = {'VQC': primary[partition+'_probability']}
                        predictions.update({r: diagnostic[r+'_'+partition] for r in
                                            ('LR_z', 'LR_angle', 'class_reference_Q', 'class_reference_E')})
                        base = dict(fold=f['fold'], seed=s, arm=arm, split=partition)
                        for readout, p in predictions.items():
                            result = score_diagnostics(y, p)
                            old = reference[(reference.fold == f['fold']) & (reference.seed == s) &
                                            (reference.arm == arm) & (reference.snapshot == 'final') &
                                            (reference.split == partition) & (reference.readout == readout)]
                            assert len(old) == 1
                            max_error = max(max_error, *(abs(result[k]-old.iloc[0][k]) for k in
                                                        ('roc_auc', 'log_loss', 'balanced_accuracy', 'brier')))
                            rows.append(dict(**base, readout=readout, **result))
                            for b in range(10):
                                mask = (p >= b/10) & ((p < (b+1)/10) if b < 9 else (p <= 1))
                                bins.append(dict(**base, readout=readout, bin=b, n=int(mask.sum()),
                                                 probability_sum=float(p[mask].sum()), positive_count=int(y[mask].sum())))
                        pairs.append(dict(**base, **paired_diagnostics(y, predictions['VQC'], predictions['LR_z'])))
    assert max_error < 1e-12
    # Recheck all accessed inputs before creating any derived files.
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in accessed.items())
    output.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(output/'scores.csv', index=False)
    pd.DataFrame(pairs).to_csv(output/'paired.csv', index=False)
    pd.DataFrame(bins).to_csv(output/'reliability_bins.csv', index=False)
    provenance = dict(status='passed', analysis='post-hoc descriptive; no C1 implementation',
                      prediction_groups=len(rows), paired_groups=len(pairs), max_metric_error=max_error,
                      source_hashes=accessed, source_module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                      training_executed=False, locked_features_or_labels_loaded=False,
                      probability_arrays_only=True, historical_inputs_unchanged=True)
    (output/'audit.json').write_text(json.dumps(provenance, indent=2)+'\n')
    return provenance


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='New directory for non-patient-level summaries')
    args = parser.parse_args()
    result = audit(args.output)
    print(json.dumps({k: v for k, v in result.items() if k != 'source_hashes'}, indent=2))
