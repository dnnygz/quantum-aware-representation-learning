"""Aggregate completed C0 artifacts without fitting, training, or cohort access."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

ARMS=['C0-NQE-Q','C0-NQE-E','C0-Hybrid-Q','C0-Hybrid-E']
MEASURES=['roc_auc','log_loss','ap_er_negative','balanced_accuracy','brier']


def flatten_geometry(folder, fold, seed, arm, snapshot, frozen=False):
    path=folder/('frozen_observable_geometry.json' if frozen else 'diagnostics.json')
    value=json.loads(path.read_text())
    rows=[]
    for split,g in value['geometry'].items():
        row=dict(fold=fold,seed=seed,arm=arm,snapshot=snapshot,split=split,
                 head='initial_frozen' if frozen else 'trained' if snapshot=='final' else 'initial')
        row.update(g['class_states']);row.update(g['observable'])
        for kind,summary in g['pair_geometry'].items():
            for key in ['balanced_pair_risk','balanced_separation']:row[kind+'_'+key]=summary[key]
            for group,stats in summary['strata'].items():
                for key in ['mean','sd','fraction_below_001','fraction_above_099']:
                    row[f'{kind}_{group}_{key}']=stats[key]
        for space in ['latent','angular']:
            for key in ['numerical_rank','entropy_rank','participation_ratio']:
                row[space+'_'+key]=g[space+'_spectrum'][key]
        row.update(saturation_95=g['saturation_95'],saturation_99=g['saturation_99'],
                   chain_rule_max_error=value['responses'][split]['chain_rule_max_error'])
        row['rms_angular_response_norm']=float(np.linalg.norm(value['responses'][split]['rms_expectation_angular_response']))
        rows.append(row)
    return rows


def summarize(frame, keys, measures):
    folds=frame.groupby(keys+['fold'],as_index=False)[measures].mean()
    stats=folds.groupby(keys)[measures].agg(['mean','std','min','max'])
    stats.columns=['_'.join(c) for c in stats.columns]
    return folds,stats.reset_index()


def analyze(run, output):
    run,output=Path(run),Path(output);output.mkdir(parents=True,exist_ok=False)
    status=json.loads((run/'run_status.json').read_text())
    metrics=pd.read_csv(run/'metrics.csv')
    metrics.to_csv(output/'metrics.csv',index=False)
    paired=pd.read_csv(run/'paired_Q_minus_E.csv');paired.to_csv(output/'paired_Q_minus_E.csv',index=False)
    fields=['log_loss_Q_minus_E','roc_auc_Q_minus_E']
    fold,summary=summarize(paired,['schedule','split'],fields)
    fold.to_csv(output/'paired_fold_means.csv',index=False);summary.to_csv(output/'primary_summary.csv',index=False)
    _,scores=summarize(metrics,['arm','snapshot','readout','split'],MEASURES)
    scores.to_csv(output/'score_summary.csv',index=False)
    geometric=[];frozen=[];curves=[];costs=[];gradients=[];probe_timing=[]
    for f in range(5):
        for s in [1729,1730,1731]:
            initial=run/f'fold_{f}_seed_{s}_initial'
            if initial.exists():
                geometric.extend(flatten_geometry(initial,f,s,'shared_initial','initial'))
                d=json.loads((initial/'diagnostics.json').read_text())
                probe_timing.append(dict(fold=f,seed=s,arm='shared_initial',**d['timing'],warnings=len(d['warnings'])))
            for arm in ARMS:
                folder=run/f'fold_{f}_seed_{s}_{arm}'
                if not (folder/'status.json').exists() or json.loads((folder/'status.json').read_text())['status']!='completed':continue
                geometric.extend(flatten_geometry(folder,f,s,arm,'final'))
                frozen.extend(flatten_geometry(folder,f,s,arm,'final',True))
                c=pd.read_csv(folder/'curves.csv').assign(fold=f,seed=s,arm=arm);curves.append(c)
                r=json.loads((folder/'resources.json').read_text())
                resource={k:v for k,v in r.items() if not isinstance(v,(dict,list))}
                for stage in r['stages']:
                    resource[stage['stage']+'_seconds']=stage['seconds']
                    resource[stage['stage']+'_steps']=stage['optimizer_steps']
                costs.append(dict(fold=f,seed=s,arm=arm,**resource))
                d=json.loads((folder/'diagnostics.json').read_text())
                probe_timing.append(dict(fold=f,seed=s,arm=arm,**d['timing'],warnings=len(d['warnings'])))
                for g in json.loads((folder/'gradient_components.json').read_text()):gradients.append(dict(fold=f,seed=s,arm=arm,**g))
    geometry=pd.DataFrame(geometric);geometry.to_csv(output/'geometry.csv',index=False)
    pd.DataFrame(frozen).to_csv(output/'frozen_observable_geometry.csv',index=False)
    pd.concat(curves,ignore_index=True).to_csv(output/'curves.csv',index=False)
    pd.DataFrame(costs).to_csv(output/'costs.csv',index=False)
    pd.DataFrame(probe_timing).to_csv(output/'diagnostic_timing.csv',index=False)
    pd.DataFrame(gradients).to_csv(output/'gradient_components.csv',index=False)
    final=metrics[(metrics.readout=='VQC') & (metrics.snapshot=='final')]
    gaps=final.pivot(index=['arm','fold','seed'],columns='split',values=MEASURES)
    gap=pd.DataFrame(index=gaps.index)
    for m in MEASURES:
        gap['train_'+m]=gaps[m]['train'];gap['validation_'+m]=gaps[m]['validation']
        gap['validation_minus_train_'+m]=gaps[m]['validation']-gaps[m]['train']
    gap.reset_index().to_csv(output/'generalization.csv',index=False)
    # Historical comparisons remain contextual and use completed aggregate artifacts only.
    historical=pd.read_csv('reports/phase2/cv_5fold_50e_v1/metrics.csv')
    historical.to_csv(output/'historical_metrics.csv',index=False)
    context=[]
    for reference in ['B','B_frozen_probe','B_classical','A_classical']:
        ref=historical[historical.experiment==reference]
        merged=final[final.split=='validation'].merge(ref,on=['fold','seed'],suffixes=('_C0','_reference'),validate='many_to_one')
        for row in merged.itertuples():
            context.append(dict(arm=row.arm,fold=row.fold,seed=row.seed,reference=reference,
                roc_auc_C0_minus_reference=row.roc_auc_C0-row.roc_auc_reference,
                log_loss_C0_minus_reference=row.log_loss_C0-row.log_loss_reference,
                equal_total_budget=False))
    pd.DataFrame(context).to_csv(output/'contextual_historical.csv',index=False)
    # Same-encoder readout contrasts never fit or select a diagnostic head.
    controls=metrics[(metrics.snapshot=='final') & (metrics.readout.isin(['LR_z','LR_angle','class_reference_Q','class_reference_E','frozen_initial_VQC']))]
    same=final.merge(controls,on=['arm','fold','seed','snapshot','split'],suffixes=('_VQC','_control'),validate='one_to_many')
    same['roc_auc_VQC_minus_control']=same.roc_auc_VQC-same.roc_auc_control
    same['log_loss_VQC_minus_control']=same.log_loss_VQC-same.log_loss_control
    same.to_csv(output/'same_encoder_controls.csv',index=False)
    geometry.merge(final,on=['arm','fold','seed','snapshot','split'],validate='one_to_one').to_csv(output/'geometry_prediction.csv',index=False)
    (output/'execution_summary.json').write_text(json.dumps(dict(status=status,source_run=str(run),
        scope='development_cv_not_locked_test',summary_unit='seed_mean_within_fold_then_five_fold_means',
        standard_deviation='descriptive_between_fold_not_confidence_interval',
        patient_level_outputs_exported=False),indent=2)+'\n')
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();analyze(a.run,a.output)
