"""Recompute completed E0 predictions and paired primary contrasts without fitting."""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .analysis_views import read_csv
from .d0 import phi
from .e0 import initial_weights, validate_config
from .e0_data import load_inputs
from .evaluation import metrics
from .models import VQC
from .run import write_json
from src.data.utils import sha256_file

METRICS=['roc_auc','ap_er_negative','balanced_accuracy','log_loss','brier','ap_er_positive','accuracy']
TABLES=['metrics','head_results','e01_paired','e01_diagonal','e02_paired','summary',
        'primary_summary','fold_primary','initialization_spread','curves','costs']


def results_wide(table):
    keys=['fold','encoder_seed','head_seed','model','diagonal']
    if table.duplicated(keys+['split']).any(): raise ValueError('Duplicate E0 result')
    result=table.pivot(index=keys,columns='split',values=METRICS)
    result.columns=[f'{split}_{metric}' for metric,split in result.columns]
    result=result.reset_index()
    for metric in METRICS:
        result[f'validation_minus_train_{metric}']=result[f'validation_{metric}']-result[f'train_{metric}']
    return result


def paired_tables(wide):
    keys=['fold','encoder_seed']
    base=wide[wide.model=='B_historical'].set_index(keys)
    z=wide[wide.model=='LR_z_historical'].set_index(keys)
    angular=wide[wide.model=='LR_angle'].set_index(keys)
    e01=wide[wide.model=='VQC_refit'].copy()
    e02=wide[wide.model=='LR_angle'].copy()
    for part,references in [(e01,[('B',base),('LR_z',z),('LR_angle',angular)]),(e02,[('LR_z',z)])]:
        for label,reference in references:
            for split in ['train','validation']:
                for metric in METRICS:
                    column=f'{split}_{metric}'
                    values=part[keys].merge(reference[[column]].reset_index(),on=keys,validate='many_to_one')[column].to_numpy()
                    part[f'{column}_reference_{label}']=values
                    part[f'{column}_minus_{label}']=part[column].to_numpy()-values
    return e01,e02


def summarize_results(table):
    records=[]
    for (model,split),part in table.groupby(['model','split']):
        for metric in METRICS:
            encoder_means=part.groupby(['fold','encoder_seed'])[metric].mean()
            fold_means=encoder_means.groupby('fold').mean()
            head_sd=part.groupby(['fold','encoder_seed'])[metric].std(ddof=1).fillna(0)
            records.append(dict(model=model,split=split,metric=metric,mean=fold_means.mean(),
                between_fold_sd=fold_means.std(ddof=1),mean_within_fold_encoder_sd=encoder_means.groupby('fold').std(ddof=1).mean(),
                mean_within_encoder_head_sd=head_sd.mean(),minimum=part[metric].min(),maximum=part[metric].max()))
    return pd.DataFrame(records)


def summarize_primary(e01,e02):
    records=[];fold_rows=[]
    for scope,part in [('all_initializations',e01),('diagonal',e01[e01.diagonal]),('off_diagonal',e01[~e01.diagonal])]:
        for endpoint in ['train_log_loss_minus_B','validation_log_loss_minus_B']:
            encoder=part.groupby(['fold','encoder_seed'])[endpoint].mean()
            folds=encoder.groupby('fold').mean()
            records.append(dict(experiment='E0.1',scope=scope,endpoint=endpoint,mean=folds.mean(),
                between_fold_sd=folds.std(ddof=1),minimum=part[endpoint].min(),maximum=part[endpoint].max(),
                within_fold_encoder_sd_mean=encoder.groupby('fold').std(ddof=1).mean()))
            fold_rows.extend(dict(experiment='E0.1',scope=scope,endpoint=endpoint,fold=int(f),effect=float(v)) for f,v in folds.items())
    for endpoint in ['validation_roc_auc_minus_LR_z','validation_log_loss_minus_LR_z']:
        folds=e02.groupby('fold')[endpoint].mean()
        records.append(dict(experiment='E0.2',scope='paired_encoders',endpoint=endpoint,mean=folds.mean(),
            between_fold_sd=folds.std(ddof=1),minimum=e02[endpoint].min(),maximum=e02[endpoint].max(),
            within_fold_encoder_sd_mean=e02.groupby('fold')[endpoint].std(ddof=1).mean()))
        fold_rows.extend(dict(experiment='E0.2',scope='paired_encoders',endpoint=endpoint,fold=int(f),effect=float(v)) for f,v in folds.items())
    return pd.DataFrame(records),pd.DataFrame(fold_rows)


def analyze(run,output):
    run,output=Path(run),Path(output)
    if output.exists(): raise FileExistsError(output)
    provenance=json.loads((run/'provenance.json').read_text())
    config=json.loads((run/'config.json').read_text())
    from .e0 import CLOSED
    validate_config({k:config[k] for k in CLOSED})
    if sha256_file(run/'config.json')!=provenance['config_sha256']:
        raise ValueError('E0 configuration hash mismatch')
    if sha256_file(run/'approved_protocol.md')!=provenance['protocol_sha256']:
        raise ValueError('Approved protocol snapshot mismatch')
    statuses=json.loads((run/'statuses.json').read_text())
    expected={(f,s,m,h) for f in range(5) for s in [1729,1730,1731]
              for m,hs in [('VQC_refit',[1729,1730,1731]),('LR_angle',[-1])] for h in hs}
    if (len(statuses)!=60 or {(s['fold'],s['encoder_seed'],s['model'],s['head_seed']) for s in statuses}!=expected
            or any(s['status']!='completed' for s in statuses)):
        raise ValueError('E0 must retain all 60 planned completed fits; failed fits cannot be silently excluded')
    inputs,_,_=load_inputs()
    saved=read_csv(run/'metrics.csv');rows=[];sources={}
    def record(path): sources[str(path)]=sha256_file(path);return path
    for path in ['config.json','approved_protocol.md','provenance.json','statuses.json','planned_design.json','costs.csv','curves.csv','metrics.csv']:
        record(run/path)
    for item in inputs:
        f,s=item['fold'],item['encoder_seed'];base=run/f'fold_{f}_encoder_{s}'
        for name,reference in item['historical'].items():
            for split,p in reference['probabilities'].items():
                result=metrics(item['y'][split],p)
                rows.append(dict(fold=f,encoder_seed=s,head_seed=-1,model=name,split=split,diagonal=False,
                    **{k:result[k] for k in METRICS if k!='brier'},brier=float(np.mean((p-item['y'][split])**2))))
        shuffle=[]
        for name,hs in [('VQC_refit',[1729,1730,1731]),('LR_angle',[-1])]:
            for h in hs:
                folder=base/(f'VQC_head_{h}' if name=='VQC_refit' else 'LR_angle')
                if name=='VQC_refit':
                    initial=torch.load(record(folder/'initial_model.pt'),weights_only=True)
                    if not torch.equal(initial['weights'],initial_weights(h)):
                        raise ValueError('Unexpected head initialization')
                    with torch.random.fork_rng(devices=[]): head=VQC()
                    head.load_state_dict(torch.load(record(folder/'model.pt'),weights_only=True))
                    curve=read_csv(record(folder/'curves.csv'))
                    if (curve.epoch.tolist()!=list(range(1,51)) or not np.isfinite(curve.to_numpy()).all()
                            or curve.optimizer_steps.tolist()!=list(range(21,1051,21))):
                        raise ValueError('Fixed epoch or optimizer budget violated')
                else:
                    with record(folder/'model.pkl').open('rb') as stream: estimator=pickle.load(stream)
                    if estimator.get_params(deep=False)!=item['probe_parameters']:
                        raise ValueError('LR-angle settings differ from historical probe')
                cost=json.loads(record(folder/'cost.json').read_text())
                record(folder/'status.json');record(folder/'metrics.json')
                if name=='VQC_refit':
                    if cost['optimizer_steps']!=1050 or cost['training_qnode_calls']!=1100 or cost['total_qnode_calls']!=1102:
                        raise ValueError('Unexpected head optimization/evaluation counters')
                    shuffle.append(cost['shuffle_sha256'])
                for split in ['train','validation']:
                    pred=read_csv(record(folder/f'{split}_predictions.csv'))
                    if pred.case_id.duplicated().any() or pred.case_id.tolist()!=item['ids'][split]:
                        raise ValueError('New prediction row order does not match canonical split')
                    np.testing.assert_array_equal(pred.er_label,item['y'][split])
                    p=pred.probability_er_positive.to_numpy()
                    if name=='VQC_refit':
                        with torch.no_grad(): reconstructed=head(torch.tensor(np.array(item['z'][split],copy=True))).numpy()
                    else: reconstructed=estimator.predict_proba(phi(item['z'][split]))[:,1]
                    np.testing.assert_allclose(p,reconstructed,atol=1e-12,rtol=0)
                    metric=metrics(pred.er_label,p)
                    rows.append(dict(fold=f,encoder_seed=s,head_seed=h,model=name,split=split,
                        diagonal=name=='VQC_refit' and s==h,**{k:metric[k] for k in METRICS if k!='brier'},
                        brier=float(np.mean((p-item['y'][split])**2))))
            if name=='VQC_refit' and len(set(shuffle))!=1:
                raise ValueError('Batch order differs across head initializations')
    table=pd.DataFrame(rows)
    keys=['fold','encoder_seed','head_seed','model','split']
    left=table.sort_values(keys).reset_index(drop=True);right=saved.sort_values(keys).reset_index(drop=True)
    if len(table)!=180: raise ValueError('Expected 180 split-level metric sets including historical references')
    for metric in METRICS:
        np.testing.assert_allclose(left[metric],right[metric],atol=1e-12,rtol=0)
    wide=results_wide(table);e01,e02=paired_tables(wide)
    primary,fold_primary=summarize_primary(e01,e02)
    spread=table[table.model=='VQC_refit'].groupby(['fold','encoder_seed','split'])[METRICS].agg(['mean','std','min','max'])
    spread.columns=['_'.join(c) for c in spread.columns]
    tables=dict(metrics=table,head_results=wide,e01_paired=e01,e01_diagonal=e01[e01.diagonal].copy(),
        e02_paired=e02,summary=summarize_results(table),primary_summary=primary,fold_primary=fold_primary,
        initialization_spread=spread.reset_index(),curves=read_csv(run/'curves.csv'),costs=read_csv(run/'costs.csv'))
    output.mkdir(parents=True,exist_ok=False)
    for name,frame in tables.items(): frame.to_csv(output/f'{name}.csv',index=False)
    write_json(output/'analysis_provenance.json',dict(run=str(run),sources=sources,
        protocol_sha256=provenance['protocol_sha256'],config_sha256=provenance['config_sha256'],
        analysis_source_sha256=sha256_file(Path(__file__)),
        all_180_metric_sets_recomputed=True,all_120_new_split_predictions_reproduced=True,
        diagonal_fits=15,all_vqc_fits=45,lr_angle_fits=15,all_15_shuffle_groups_paired=True,
        raw_cohort_loaded=False,locked_test_access=False,model_fitting_performed=False))
    return tables


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--run',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();tables=analyze(args.run,args.output)
    print(tables['primary_summary'].to_string(index=False))


if __name__=='__main__': main()
