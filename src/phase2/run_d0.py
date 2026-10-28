"""Execute exploratory D0 using trusted saved development artifacts only; never fit."""
import argparse
import importlib.metadata
import json
import pickle
import subprocess
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .analysis_views import read_csv, MEASURES
from .data import validate_partitions
from .d0 import (FrozenHead, ALPHAS, DONOR_SEEDS, phi, training_reference, interventions,
                 displacement, predictive_change, response_summary, geometry)
from .evaluation import metrics
from .run import write_json
from src.data.utils import sha256_file


TABLES = ['geometry','coordinates','functional','directions','checks','baseline_metrics',
          'shared_encoder','intervention_summary','paired_interventions']


def aggregate_interventions(frame):
    """Collapse the fixed intervention design inside each fold/seed, never across patients."""
    measures=[c for c in frame if c.startswith('delta_')]+[
        'probability_change_abs_mean','decision_flip_fraction','z_l2_mean','angle_l2_mean',
        'outside_training_bounds_before','outside_training_bounds_after',
        'nearest_training_distance_before','nearest_training_distance_after','train_scaled_l2_mean',
        'linearization_error_abs_mean']
    keys=['fold','seed','model','split','family','alpha']
    grouped=frame.groupby(keys,dropna=False)[measures].agg(['mean','min','max'])
    grouped.columns=['_'.join(c) for c in grouped.columns]
    result=grouped.reset_index()
    sizes=frame.groupby(keys,dropna=False).size().rename('design_cells').reset_index()
    return result.merge(sizes,on=keys,validate='one_to_one')


def run(output, summaries, cv_analysis='reports/phase2/cv_5fold_50e_v1',
        protocol_path='data/processed/splits/phase1c_inner_cv.json'):
    output, summaries, cv_analysis = Path(output), Path(summaries), Path(cv_analysis)
    if output.exists() or summaries.exists():
        raise FileExistsError('Use new D0 run and summary directories; existing artifacts are never overwritten')
    started=time.perf_counter()
    torch.set_num_threads(1)
    manifest=json.loads((cv_analysis/'provenance.json').read_text())
    source_hashes=manifest['sources']
    accessed={}
    def checked(path):
        path=Path(path)
        expected=source_hashes.get(str(path))
        if expected is None or sha256_file(path)!=expected:
            raise ValueError(f'Saved CV provenance mismatch: {path}')
        accessed[str(path)]=expected
        return path
    protocol_path=Path(protocol_path)
    protocol=json.loads(protocol_path.read_text())
    validate_partitions(protocol)
    if (len(protocol['development_case_ids'])!=835 or len(protocol['locked_test_case_ids'])!=209
            or sha256_file(protocol_path)!=manifest['protocol_sha256']):
        raise ValueError('Unexpected canonical protocol metadata')
    folds={f['fold']:f for f in protocol['folds']}
    if set(folds)!=set(range(5)):
        raise ValueError('D0 requires all five development folds')
    roots=[Path(r) for r in manifest['runs']]
    root_by_fold={}
    configs=[]
    for root in roots:
        config=json.loads(checked(root/'config.json').read_text());configs.append(config)
        provenance=json.loads(checked(root/'provenance.json').read_text())
        if provenance['locked_test_evaluated'] is not False:
            raise ValueError('Locked-test evaluation is prohibited')
        if config['seeds']!=[1729,1730,1731] or config['epochs']!=50:
            raise ValueError('Unexpected source protocol')
        for fold in config['folds']:
            if fold in root_by_fold:
                raise ValueError('Duplicate fold source')
            root_by_fold[fold]=root
    if set(root_by_fold)!=set(range(5)):
        raise ValueError('Incomplete source folds')
    # Reconstruct development labels from saved validation predictions, not raw labels.
    labels={}
    for fold,root in root_by_fold.items():
        pred=read_csv(checked(root/f'fold_{fold}_A/A_seed_1729/validation_predictions.csv'))
        if pred.case_id.duplicated().any() or set(pred.case_id)!=set(folds[fold]['validation_case_ids']):
            raise ValueError('Invalid saved development validation predictions')
        for row in pred.itertuples():
            if row.case_id in labels:
                raise ValueError('Validation coverage must be disjoint')
            labels[row.case_id]=int(row.er_label)
    if set(labels)!=set(protocol['development_case_ids']) or set(labels.values())!={0,1}:
        raise ValueError('Invalid development-only label coverage')
    output.mkdir(parents=True,exist_ok=False)
    config=dict(protocol='D0-v1 exploratory; approved revisions',folds=list(range(5)),seeds=[1729,1730,1731],
        checkpoints_D0_A=['initial','final'],checkpoints_D0_B=['final'],alphas=list(ALPHAS),
        donor_seeds=list(DONOR_SEEDS),donors='Training permutation; validation uses training donors without replacement',
        bases=['coordinate','training_principal_axis'],coordinate_replacements=['training_mean','zero','training_donor'],
        train_standard_deviation_ddof=0,rank_relative_threshold=1e-6,
        fd_step=1e-5,fd_count_per_split=19,chain_absolute_tolerance=1e-10,
        fd_absolute_tolerance=1e-7,forward_absolute_tolerance=1e-12,
        discovery_recurrence_rule=None,training_or_calibration=False,locked_test_access=False,
        comparison='B/probe share z; B_classical comparisons use aggregate properties only',
        interpretation='Variation, functional sensitivity, and predictive effects are distinct',
        cv_analysis=str(cv_analysis),source_configurations=configs)
    write_json(output/'config.json',config)
    # Preserve a source snapshot and hashes before the first numerical diagnostic.
    source=output/'source';source.mkdir()
    for filename in ['d0.py','run_d0.py','models.py','evaluation.py','analyze_cv.py','analysis_views.py']:
        (source/filename).write_bytes((Path(__file__).parent/filename).read_bytes())
    rows={name:[] for name in TABLES if name not in ['intervention_summary','paired_interventions']}
    detailed=[]
    for fold in range(5):
        root=root_by_fold[fold];f=folds[fold]
        ysplit={split:np.array([labels[c] for c in f[key]]) for split,key in
                [('train','train_case_ids'),('validation','validation_case_ids')]}
        for representation in ['A','B']:
            saved=json.loads(checked(root/f'fold_{fold}_{representation}/representation.json').read_text())
            if saved['train_ids']!=f['train_case_ids'] or saved['validation_ids']!=f['validation_case_ids']:
                raise ValueError('Latent row ordering does not match the fold')
        for seed in [1729,1730,1731]:
            target=output/f'fold_{fold}_seed_{seed}';target.mkdir()
            shared={};all_changed={}
            for name in ['A','B','B_frozen_probe','B_classical']:
                base=root/f'fold_{fold}_{"A" if name=="A" else "B"}'
                folder=base/f'{name}_seed_{seed}'
                latent_name='B' if name=='B_frozen_probe' else name
                with np.load(checked(base/f'{latent_name}_seed_{seed}/latent_diagnostics.npz'),allow_pickle=False) as archive:
                    latent={key:archive[key].copy() for key in archive.files}
                if any(v.shape!=(668 if key.endswith('train') else 167,4) or not np.isfinite(v).all()
                       for key,v in latent.items()):
                    raise ValueError('Invalid latent arrays')
                prediction=read_csv(checked(folder/'validation_predictions.csv'))
                indexed=prediction.set_index('case_id')
                if prediction.case_id.duplicated().any() or set(prediction.case_id)!=set(f['validation_case_ids']):
                    raise ValueError('Invalid validation prediction IDs')
                ordered=indexed.loc[f['validation_case_ids']]
                np.testing.assert_array_equal(ordered.er_label,ysplit['validation'])
                estimator=None
                if name=='B_frozen_probe':
                    with checked(folder/'model.pkl').open('rb') as stream:
                        estimator=pickle.load(stream)
                for checkpoint in (['final'] if estimator is not None else ['initial','final']):
                    state=None
                    if estimator is None:
                        path=folder/('initial_model.pt' if checkpoint=='initial' else 'model.pt')
                        state=torch.load(checked(path),weights_only=True)
                    head=FrozenHead(name,state,estimator)
                    before={k:v.clone() for k,v in state.items()} if state is not None else None
                    training=latent[f'{checkpoint}_train']
                    reference=training_reference(training,ysplit['train'])
                    np.savez_compressed(target/f'{name}_{checkpoint}_training_reference.npz',**reference)
                    for split in ['train','validation']:
                        z=latent[f'{checkpoint}_{split}'];y=ysplit[split]
                        info=dict(fold=fold,seed=seed,model=name,checkpoint=checkpoint,split=split)
                        derivative=head.derivatives(z)
                        p=derivative['probability'];gp=derivative['probability_gradient']
                        if checkpoint=='final' and split=='validation':
                            np.testing.assert_allclose(p,ordered.probability_er_positive,atol=1e-12,rtol=0)
                        rows['baseline_metrics'].append(dict(**info,**{k:v for k,v in metrics(y,p).items() if k!='confusion_matrix'},
                            brier=float(np.mean((p-y)**2))))
                        checks=head.numerical_checks(z)
                        if checks['fd_probability_max_abs_error']>config['fd_absolute_tolerance']:
                            raise ValueError('Finite-difference verification failed')
                        if head.quantum:
                            if derivative['chain_absolute_error'].max()>config['chain_absolute_tolerance'] or checks['final_rz_perturbation_max_abs_error']>1e-10:
                                raise ValueError('Chain rule or structural negative control failed')
                            checks['chain_max_abs_error']=float(derivative['chain_absolute_error'].max())
                        rows['checks'].append(dict(**info,**checks))
                        np.savez_compressed(target/f'{name}_{checkpoint}_{split}_derivatives.npz',**derivative)
                        if name!='B_frozen_probe':
                            for space,value in [('z',z),('angle',phi(z))]:
                                rows['geometry'].append(dict(**info,space=space,angular_map_actual=head.quantum,**geometry(value)))
                        for j in range(4):
                            co=dict(**info,coordinate=j,training_sd=float(reference['sd'][j]),
                                z_sd=float(z[:,j].std()),angle_sd=float(phi(z)[:,j].std()),
                                probability_gradient_rms=float(np.sqrt(np.mean(gp[:,j]**2))),
                                training_sd_scaled_probability_gradient_rms=float(reference['sd'][j]*np.sqrt(np.mean(gp[:,j]**2))))
                            if head.quantum:
                                d=derivative['map_derivative'][:,j]
                                ga=derivative['angle_gradient'][:,j];gz=derivative['expectation_gradient'][:,j]
                                co.update(map_derivative_mean=float(d.mean()),map_derivative_q05=float(np.quantile(d,.05)),
                                    map_derivative_median=float(np.median(d)),map_derivative_q95=float(np.quantile(d,.95)),
                                    map_derivative_relative_max_mean=float(d.mean()/(np.pi/2)),
                                    angle_expectation_gradient_rms=float(np.sqrt(np.mean(ga*ga))),
                                    z_expectation_gradient_rms=float(np.sqrt(np.mean(gz*gz))),
                                    chain_max_abs_error=float(derivative['chain_absolute_error'][:,j].max()))
                            rows['coordinates'].append(co)
                        for kind,gradient in [('probability_z',gp)]+(
                                [('expectation_angle',derivative['angle_gradient']),('expectation_z',derivative['expectation_gradient'])] if head.quantum else []):
                            summary=response_summary(gradient,reference)
                            # Scaling angle gradients by z SD would mix spaces; omit this descriptor.
                            if kind=='expectation_angle':
                                summary['training_sd_scaled_gradient_norm_rms']=np.nan
                            rows['functional'].append(dict(**info,derivative_kind=kind,**summary))
                        for j in range(5):
                            direction=reference['axes'][:,j] if j<4 else reference['class_difference']
                            scale=np.sqrt(reference['eigenvalues'][j]) if j<4 else 1.
                            projection=gp@direction*scale
                            rows['directions'].append(dict(**info,direction=f'pc_{j}' if j<4 else 'training_class_mean_difference',
                                scale=float(scale),direction_norm=float(np.linalg.norm(direction)),
                                probability_directional_derivative_mean=float(projection.mean()),
                                probability_directional_derivative_rms=float(np.sqrt(np.mean(projection**2)))))
                        if checkpoint=='final' and name in ['B','B_frozen_probe']:
                            shared[name,split]=(gp,p)
                        if checkpoint!='final':
                            continue
                        changes={}
                        for intervention,altered in interventions(z,training,reference,fold,seed,split):
                            after=head.predict(altered);changes[intervention['intervention']]=after
                            first_order=np.sum(gp*(altered-z),axis=1)
                            actual=after-p
                            row=dict(**info,**intervention,**predictive_change(y,p,after),
                                **displacement(z,altered,training,reference,split),
                                linearization_error_abs_mean=float(np.mean(np.abs(actual-first_order))))
                            detailed.append(row)
                        np.savez_compressed(target/f'{name}_{split}_intervention_predictions.npz',**changes)
                        all_changed[name,split]=changes
                    if before is not None:
                        for key,value in state.items():
                            if not torch.equal(value,before[key]):
                                raise ValueError('A checkpoint tensor was mutated')
            for split in ['train','validation']:
                gb,pb=shared['B',split];gl,pl=shared['B_frozen_probe',split]
                norm=np.linalg.norm(gb,axis=1)*np.linalg.norm(gl,axis=1)
                valid=norm>1e-15
                cosine=np.sum(gb[valid]*gl[valid],axis=1)/norm[valid]
                rows['shared_encoder'].append(dict(fold=fold,seed=seed,split=split,
                    gradient_cosine_mean=float(cosine.mean()),gradient_cosine_q05=float(np.quantile(cosine,.05)),
                    gradient_cosine_median=float(np.median(cosine)),gradient_cosine_q95=float(np.quantile(cosine,.95)),
                    defined_cosine_fraction=float(valid.mean()),probability_abs_difference_mean=float(np.mean(np.abs(pb-pl)))))
            print(f'Completed frozen D0: fold={fold}, seed={seed}',flush=True)
    detail=pd.DataFrame(detailed)
    detail.to_csv(output/'interventions.csv',index=False)
    tables={k:pd.DataFrame(v) for k,v in rows.items()}
    tables['intervention_summary']=aggregate_interventions(detail)
    a=detail[detail.model=='B'];b=detail[detail.model=='B_frozen_probe']
    keys=['fold','seed','split','intervention','family','direction','alpha','sign','donor_seed']
    values=[c for c in detail if c.startswith('delta_')]+['probability_change_abs_mean','linearization_error_abs_mean']
    paired=a[keys+values].merge(b[keys+values],on=keys,suffixes=('_B','_probe'),validate='one_to_one')
    for c in values:
        paired[f'B_minus_probe_{c}']=paired[c+'_B']-paired[c+'_probe']
    paired.to_csv(output/'paired_interventions.csv',index=False)
    # Keep paired summary at fold/seed/design-family grain; individual intervention cells remain private.
    paired['model']='B_minus_probe';paired['checkpoint']='final'
    for c in values:
        paired[c]=paired[f'B_minus_probe_{c}']
    numeric=[c for c in paired if c.startswith('B_minus_probe_')]
    group=paired.groupby(['fold','seed','split','family','alpha'])[numeric].agg(['mean','min','max'])
    group.columns=['_'.join(c) for c in group.columns]
    tables['paired_interventions']=group.reset_index()
    summaries.mkdir(parents=True,exist_ok=False)
    for name,table in tables.items():
        table.to_csv(output/f'{name}.csv',index=False) if name!='paired_interventions' else None
        table.to_csv(summaries/f'{name}.csv',index=False)
    for path,expected in accessed.items():
        if sha256_file(Path(path))!=expected:
            raise ValueError(f'Historical artifact changed during D0: {path}')
    provenance=dict(protocol='D0-v1',stage='exploratory frozen-checkpoint diagnostics',
        locked_test_access=False,model_fitting_performed=False,
        labels_source='Existing development validation predictions only',sources=accessed,
        source_sha256={p.name:sha256_file(p) for p in source.glob('*.py')},
        output=str(output),summary_directory=str(summaries),
        git_commit=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip(),
        versions={n:importlib.metadata.version(n) for n in ['torch','pennylane','numpy','pandas','scikit-learn','scipy']},
        D0_A_model_checkpoint_split_groups=len(tables['baseline_metrics']),
        D0_B_model_split_groups=120,intervention_cells=len(detail),
        wall_seconds=time.perf_counter()-started,historical_accessed_files_unchanged=len(accessed),
        missing_intermediate_checkpoints=True,discovery_recurrence_rule=None)
    write_json(output/'provenance.json',provenance);write_json(summaries/'provenance.json',provenance)
    return tables,detail


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--summaries',required=True)
    parser.add_argument('--cv-analysis',default='reports/phase2/cv_5fold_50e_v1')
    args=parser.parse_args()
    run(args.output,args.summaries,args.cv_analysis)


if __name__=='__main__':
    main()
