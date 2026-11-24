"""Verify and summarize the fixed five-fold development experiment without fitting."""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .analysis_views import MODELS, MEASURES, read_csv, spectrum_values, angular_values, paired_predictions
from .data import validate_partitions
from .evaluation import metrics
from .models import Model
from .run import write_json
from .training import seed_all
from src.data.utils import sha256_file


def latent_summary(z):
    """Centered spectrum, variance entropy rank, and participation ratio (not numerical rank)."""
    result = spectrum_values(z)
    s = np.array([result[f'singular_{i}'] for i in range(4)])
    total = np.sum(s*s)
    weights = s*s/total if total else np.zeros(4)
    positive = weights[weights > 0]
    result.update(variance_entropy_rank=float(np.exp(-np.sum(positive*np.log(positive)))) if total else 0.,
                  participation_ratio=float(1/np.sum(weights*weights)) if total else 0.)
    for i in range(4):
        result[f'coordinate_{i}_std'] = float(z[:,i].std(ddof=0))
        result[f'variance_fraction_{i}'] = float(weights[i])
    return result


def paired_table(table):
    """Retain fold and seed pairing; never treat 15 pairs as independent cohorts."""
    if table.duplicated(['fold','seed','experiment']).any():
        raise ValueError('Duplicate model/fold/seed')
    rows = []
    for (fold,seed), part in table.groupby(['fold','seed']):
        by = part.set_index('experiment')
        if set(by.index) != set(MODELS):
            raise ValueError('Incomplete paired model set')
        row = dict(fold=fold,seed=seed)
        for left,right in [('B','A'),('B','B_frozen_probe'),('B','B_classical'),('A','A_classical')]:
            for metric in MEASURES+['brier']:
                row[f'{left}_minus_{right}_{metric}'] = float(by.loc[left,metric]-by.loc[right,metric])
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(table):
    """Separate between-fold variation from within-fold seed variation, descriptively."""
    rows=[]
    for name, part in table.groupby('experiment',sort=False):
        for metric in MEASURES+['brier']:
            fold_means=part.groupby('fold')[metric].mean()
            seed_sd=part.groupby('fold')[metric].std(ddof=1)
            rows.append(dict(experiment=name,metric=metric,mean=fold_means.mean(),
                between_fold_sd=fold_means.std(ddof=1),mean_within_fold_seed_sd=seed_sd.mean(),
                minimum=part[metric].min(),maximum=part[metric].max()))
    return pd.DataFrame(rows)


def analyze(pilot, remaining, protocol_path, output):
    """Read only trusted local experiment artifacts; reject mismatched provenance."""
    output=Path(output)
    if output.exists():
        raise FileExistsError(output)
    roots=[Path(pilot),Path(remaining)]
    configs=[json.loads((r/'config.json').read_text()) for r in roots]
    provenance=[json.loads((r/'provenance.json').read_text()) for r in roots]
    reference={k:v for k,v in configs[0].items() if k!='folds'}
    if configs[0]['folds'] != [0] or configs[1]['folds'] != [1,2,3,4]:
        raise ValueError('Expected historical fold 0 and new folds 1–4')
    if reference != {k:v for k,v in configs[1].items() if k!='folds'}:
        raise ValueError('Protocol changed beyond fold selection')
    if reference['seeds'] != [1729,1730,1731] or reference['epochs'] != 50:
        raise ValueError('Unexpected seeds or epoch budget')
    for key in ['versions','input_sha256']:
        if provenance[0][key] != provenance[1][key]:
            raise ValueError(f'Historical/new {key} mismatch')
    frozen=['data.py','models.py','training.py','run.py','evaluation.py','diagnostics.py']
    if any(provenance[0]['source_sha256'][f]!=provenance[1]['source_sha256'][f] for f in frozen):
        raise ValueError('Execution source changed since pilot')
    if any(p['locked_test_evaluated'] is not False for p in provenance):
        raise ValueError('Locked evaluation is prohibited')
    protocol_path=Path(protocol_path)
    protocol=json.loads(protocol_path.read_text())
    validate_partitions(protocol)
    if len(protocol['development_case_ids'])!=835 or len(protocol['locked_test_case_ids'])!=209:
        raise ValueError('Unexpected canonical partition sizes')
    expected_protocol_hash=next(h for p,h in provenance[0]['input_sha256'].items() if p.endswith('phase1c_inner_cv.json'))
    if sha256_file(protocol_path)!=expected_protocol_hash:
        raise ValueError('Protocol metadata hash mismatch')
    folds={f['fold']:f for f in protocol['folds']}
    table=pd.concat([read_csv(r/'comparison.csv') for r in roots],ignore_index=True)
    expected={(f,s,m) for f in range(5) for s in reference['seeds'] for m in MODELS}
    if len(table)!=75 or set(zip(table.fold,table.seed,table.experiment))!=expected:
        raise ValueError('Expected exactly 75 unique completed results')
    sources={}
    def record(path):
        sources[str(path)]=sha256_file(path)
        return path
    predictions={}
    labels={}
    for root,config in zip(roots,configs):
        for file in ['config.json','comparison.csv','resources.json','provenance.json']:
            record(root/file)
        for fold in config['folds']:
            for name in MODELS:
                for seed in reference['seeds']:
                    folder=root/f'fold_{fold}_{"A" if name.startswith("A") else "B"}'/f'{name}_seed_{seed}'
                    pred=read_csv(record(folder/'validation_predictions.csv'))
                    if pred.case_id.duplicated().any() or set(pred.case_id)!=set(folds[fold]['validation_case_ids']):
                        raise ValueError('Prediction partition mismatch')
                    if not set(pred.er_label.unique()) <= {0,1}:
                        raise ValueError('Unexpected labels')
                    for row in pred.itertuples():
                        if row.case_id in labels and labels[row.case_id] != row.er_label:
                            raise ValueError('Inconsistent development labels')
                        labels[row.case_id]=row.er_label
                    predictions[fold,seed,name]=pred
    if set(labels)!=set(protocol['development_case_ids']):
        raise ValueError('Development coverage mismatch')
    curves=[]; spectra=[]; angles=[]; training=[]; calibration=[]; checks=[]
    for root,config in zip(roots,configs):
        for fold in config['folds']:
            f=folds[fold]
            for experiment in ['A','B']:
                base=root/f'fold_{fold}_{experiment}'
                rep=json.loads(record(base/'representation.json').read_text())
                if rep['train_ids']!=f['train_case_ids'] or rep['validation_ids']!=f['validation_case_ids']:
                    raise ValueError('Representation partition mismatch')
                with record(base/'representation.pkl').open('rb') as stream:
                    representation=pickle.load(stream)
                if (representation.training_ids!=rep['train_ids'] or representation.pca.n_samples_!=668
                        or representation.rna.tmm_.reference_sample_id_ not in rep['train_ids']
                        or not np.all(np.asarray(representation.scaler.n_samples_seen_)==668)):
                    raise ValueError('Persisted training-only preprocessing state mismatch')
            for seed in reference['seeds']:
                initial={}
                for name in MODELS:
                    base=root/f'fold_{fold}_{"A" if name.startswith("A") else "B"}'
                    folder=base/f'{name}_seed_{seed}'
                    key=(fold,seed,name)
                    pred=predictions[key]
                    p=pred.probability_er_positive.to_numpy(); y=pred.er_label.to_numpy()
                    result=metrics(y,p)
                    index=table.index[(table.fold==fold)&(table.seed==seed)&(table.experiment==name)][0]
                    saved=json.loads(record(folder/'metrics.json').read_text())
                    for measure in MEASURES:
                        if not np.isclose(result[measure],table.loc[index,measure],atol=1e-12,rtol=0) or not np.isclose(result[measure],saved[measure],atol=1e-12,rtol=0):
                            raise ValueError(f'Unreproducible metric {key}/{measure}')
                    if result['confusion_matrix']!=saved['confusion_matrix']:
                        raise ValueError('Confusion matrix mismatch')
                    table.loc[index,'brier']=np.mean((p-y)**2)
                    table.loc[index,'n_validation']=len(y)
                    table.loc[index,'n_er_positive']=y.sum()
                    # Fixed descriptive bins only; no calibrator or tuning is fitted.
                    bins=np.minimum((p*10).astype(int),9)
                    for bin_id in range(10):
                        subset=bins==bin_id
                        calibration.append(dict(fold=fold,seed=seed,experiment=name,bin=bin_id,n=int(subset.sum()),
                            mean_probability=float(p[subset].mean()) if subset.any() else np.nan,
                            observed_positive_fraction=float(y[subset].mean()) if subset.any() else np.nan))
                    source_name=name if name in ['A','B','B_classical'] else ('A' if name=='A_classical' else 'B')
                    source_folder=base/f'{source_name}_seed_{seed}'
                    with np.load(record(source_folder/'latent_diagnostics.npz'),allow_pickle=False) as latent:
                        ztrain=latent['final_train'].copy(); zval=latent['final_validation'].copy()
                        if name in ['A','B','B_classical']:
                            for checkpoint in ['initial','final']:
                                for split,n in [('train',668),('validation',167)]:
                                    z=latent[f'{checkpoint}_{split}']
                                    if z.shape!=(n,4) or not np.isfinite(z).all():
                                        raise ValueError('Invalid latent diagnostic')
                                    info=dict(fold=fold,seed=seed,experiment=name,checkpoint=checkpoint,split=split)
                                    spectra.append(dict(**info,**latent_summary(z)))
                                    for coordinate in [None,0,1,2,3]:
                                        angles.append(dict(**info,coordinate='pooled' if coordinate is None else str(coordinate),
                                            **angular_values(z if coordinate is None else z[:,coordinate:coordinate+1])))
                    if name in ['A','B','B_classical']:
                        curve=read_csv(record(folder/'curves.csv'))
                        if list(curve.epoch)!=list(range(1,51)) or not np.isfinite(curve.to_numpy()).all():
                            raise ValueError('Incomplete/nonfinite optimization history')
                        curves.append(curve.assign(fold=fold,seed=seed,experiment=name))
                        initial[name]=torch.load(record(folder/'initial_model.pt'),weights_only=True)
                        seed_all(seed)
                        model=Model(4 if name=='A' else 32,encoder=name!='A',quantum=name!='B_classical')
                        model.load_state_dict(torch.load(record(folder/'model.pt'),weights_only=True))
                        model.eval()
                        with torch.no_grad():
                            pt=model.head(torch.tensor(ztrain)).reshape(-1).numpy()
                            pv=model.head(torch.tensor(zval)).reshape(-1).numpy()
                    else:
                        with record(folder/'model.pkl').open('rb') as stream:
                            estimator=pickle.load(stream)
                        pt=estimator.predict_proba(ztrain)[:,1]; pv=estimator.predict_proba(zval)[:,1]
                    np.testing.assert_allclose(pv,p,atol=1e-12,rtol=0)
                    yt=np.array([labels[c] for c in f['train_case_ids']])
                    train_metrics=metrics(yt,pt)
                    training.append(dict(fold=fold,seed=seed,experiment=name,
                        **{f'train_{m}':train_metrics[m] for m in MEASURES},
                        train_brier=float(np.mean((pt-yt)**2)),
                        **{f'validation_minus_train_{m}':result[m]-train_metrics[m] for m in MEASURES}))
                if not torch.equal(initial['A']['head.weights'],initial['B']['head.weights']):
                    raise ValueError('Unpaired VQC initialization')
                for k in initial['B']:
                    if k.startswith('encoder.') and not torch.equal(initial['B'][k],initial['B_classical'][k]):
                        raise ValueError('Unpaired encoder initialization')
                _,rho=paired_predictions(predictions[fold,seed,'B'],predictions[fold,seed,'B_frozen_probe'])
                checks.append(dict(fold=fold,seed=seed,paired_initialization=True,spearman_B_probe=rho))
    pairs=paired_table(table).merge(pd.DataFrame(checks),on=['fold','seed'],validate='one_to_one')
    tables=dict(metrics=table,summary=summarize(table),paired=pairs,
        paired_fold_means=pairs.groupby('fold').mean(numeric_only=True).drop(columns=['seed','paired_initialization']).reset_index(),
        curves=pd.concat(curves,ignore_index=True),latent=pd.DataFrame(spectra),angles=pd.DataFrame(angles),
        generalization=pd.DataFrame(training),calibration=pd.DataFrame(calibration))
    output.mkdir(parents=True,exist_ok=False)
    for name,frame in tables.items():
        frame.to_csv(output/f'{name}.csv',index=False)
    write_json(output/'provenance.json',dict(sources=sources,protocol_sha256=sha256_file(protocol_path),
        analysis_source_sha256=sha256_file(Path(__file__)),
        dependencies_sha256={n:sha256_file(Path(__file__).parent/n) for n in ['analysis_views.py','evaluation.py','models.py','data.py']},
        runs=[str(r) for r in roots],versions=provenance[0]['versions'],
        frozen_execution_sources_identical=True,input_hashes_identical=True,
        locked_test_evaluated=False,validation_metrics_recomputed=75,
        final_checkpoint_training_metrics_reconstructed=75,
        training_labels_source='Existing development validation predictions across all five folds; no cohort data loaded',
        resources=[json.loads((r/'resources.json').read_text()) for r in roots]))
    return tables


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--pilot',default='runs/phase2/pilot_fold0_50e_paired_v1')
    parser.add_argument('--remaining',default='runs/phase2/cv_folds1to4_50e_paired_v1')
    parser.add_argument('--protocol',default='data/processed/splits/phase1c_inner_cv.json')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    tables=analyze(args.pilot,args.remaining,args.protocol,args.output)
    print(tables['summary'].to_string(index=False))


if __name__=='__main__':
    main()
