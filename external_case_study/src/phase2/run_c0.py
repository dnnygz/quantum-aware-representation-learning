"""C0-v1 technical smoke by default; full CV requires an explicit separate flag."""
import argparse
import copy
import importlib.metadata
import json
import pickle
import platform
from pathlib import Path
import resource
import time
import warnings

import numpy as np
import pandas as pd
import torch
import yaml

from .c0 import ARMS, CLOSED, fit_pipeline, initialized_model, synthetic_fixture, validate_config
from .c0_data import load_audited, sha256
from .c0_metrics import classification, calibration, evaluate_encoder


def write_json(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def _evaluate(folder, z, y, head, initial_head, seed, *, fit_probes=True):
    """Persist diagnostic predictions only inside the private run destination."""
    started=time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        view,estimators=evaluate_encoder(z,dict(y,seed=seed),head,fit_probes=fit_probes)
    with (folder/'probes.pkl').open('wb') as stream:pickle.dump(estimators,stream)
    with torch.no_grad():
        view['readouts']['frozen_initial_VQC']={split:initial_head(torch.tensor(values)).numpy() for split,values in z.items()}
    arrays={f'{name}_{split}':p for name,values in view.pop('readouts').items() for split,p in values.items()}
    np.savez_compressed(folder/'diagnostic_predictions.npz',**arrays)
    rows=[]
    for key,p in arrays.items():
        split='validation' if key.endswith('_validation') else 'train'
        name=key.removesuffix('_'+split)
        rows.append(dict(readout=name,split=split,**classification(y[split],p)))
    view['warnings']=[str(w.message) for w in caught]
    view['probe_iterations']={name:est.n_iter_.tolist() for name,est in estimators.items()}
    view['diagnostic_seconds']=time.perf_counter()-started
    write_json(folder/'diagnostics.json',view)
    return rows,view['diagnostic_seconds']


def paired_contrasts(frame):
    records=[]
    data=frame[(frame.readout=='VQC') & (frame.snapshot=='final')]
    for schedule in ['NQE','Hybrid']:
        q=data[data.arm==f'C0-{schedule}-Q']
        e=data[data.arm==f'C0-{schedule}-E']
        paired=q.merge(e,on=['fold','seed','split'],suffixes=('_Q','_E'),validate='one_to_one')
        for row in paired.itertuples():
            records.append(dict(schedule=schedule,fold=row.fold,seed=row.seed,split=row.split,
                log_loss_Q_minus_E=row.log_loss_Q-row.log_loss_E,
                roc_auc_Q_minus_E=row.roc_auc_Q-row.roc_auc_E))
    return pd.DataFrame(records)


def run(output, *, execute_cv=False, audit_folder='runs/phase2/c0_v1_availability_audit'):
    validate_config(yaml.safe_load(Path('configs/c0_v1.yaml').read_text()))
    design=json.loads(Path('reports/phase2/c0_v1_design/design_provenance.json').read_text())
    expected=design['consulted_inputs_sha256']['src/phase2/models.py']
    if sha256('src/phase2/models.py')!=expected:
        raise ValueError('The frozen B architecture source changed after protocol design')
    destination=Path(output)
    # No tracked patient-level output destinations, including accidental report paths.
    resolved=destination.resolve()
    if not resolved.is_relative_to(Path('runs').resolve()):
        raise ValueError('C0 outputs must remain inside the ignored runs directory')
    destination.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    mode='development_cv' if execute_cv else 'synthetic_technical_smoke'
    versions={name:importlib.metadata.version(name) for name in ['torch','pennylane','numpy','scikit-learn','pandas','pyarrow']}
    write_json(destination/'config.json',dict(CLOSED,execution_mode=mode,technical_epochs=None if execute_cv else 2))
    write_json(destination/'provenance.json',dict(mode=mode,python=platform.python_version(),versions=versions,
        protocol_sha256=sha256('reports/phase2/c0_protocol_v1.md'),
        source_sha256={str(p):sha256(p) for p in Path('src/phase2').glob('*.py')},
        locked_test_evaluated=False,synthetic_only=not execute_cv,
        primary_comparisons='Q_minus_E_within_schedule',historical_B='contextual_only',
        computational_quantum_advantage_claim=False))
    if execute_cv:
        inputs,audit=load_audited(audit_folder)
        write_json(destination/'availability_reference.json',dict(audit_sha256=sha256(Path(audit_folder)/'audit.json'),status=audit['status']))
        historical_versions=json.loads(Path('reports/phase2/cv_5fold_50e_v1/provenance.json').read_text())['versions']
        for name,version in versions.items():
            if name in historical_versions and version!=historical_versions[name]:
                raise ValueError('Environment differs from historical provenance: '+name)
    else:
        x,y,xv,yv=synthetic_fixture()
        model,_=initialized_model(1729)
        inputs=[dict(fold=0,seed=1729,x=dict(train=x,validation=xv),
                     y=dict(train=y,validation=yv),initial_state=copy.deepcopy(model.state_dict()))]
    all_rows=[];statuses=[]
    for data in inputs:
        f,s=data['fold'],data['seed']
        common=destination/f'fold_{f}_seed_{s}_initial';common.mkdir()
        original,_=initialized_model(s)
        original.load_state_dict(data['initial_state'])
        initial_head=copy.deepcopy(original.head)
        with torch.no_grad():
            initial_z={split:original.encoder(torch.tensor(x,dtype=torch.float64)).numpy() for split,x in data['x'].items()}
        initial_rows,_=_evaluate(common,initial_z,data['y'],original.head,initial_head,s)
        for row in initial_rows:all_rows.append(dict(fold=f,seed=s,arm='shared_initial',snapshot='initial',**row))
        for arm in ARMS:
            folder=destination/f'fold_{f}_seed_{s}_{arm}';folder.mkdir()
            write_json(folder/'status.json',dict(status='running',mode=mode))
            try:
                result=fit_pipeline(data['x']['train'],data['y']['train'],data['x']['validation'],data['y']['validation'],
                    f,s,arm,data['initial_state'],CLOSED,synthetic=not execute_cv,
                    technical_epochs=None if execute_cv else 2)
                torch.save(result['initial_state'],folder/'initial_model.pt')
                torch.save(result['model'].state_dict(),folder/'model.pt')
                if result['geometry_state'] is not None:
                    torch.save(result['geometry_state'],folder/'geometry_model.pt')
                np.savez_compressed(folder/'latent_and_predictions.npz',
                    **{f'{split}_z':z for split,z in result['z'].items()},
                    **{f'{split}_probability':p for split,p in result['probabilities'].items()})
                pd.DataFrame(result['curves']).to_csv(folder/'curves.csv',index=False)
                write_json(folder/'gradient_components.json',result['diagnostics'])
                rows,diagnostic_seconds=_evaluate(folder,result['z'],data['y'],result['model'].head,initial_head,s)
                for split,p in result['probabilities'].items():
                    rows.append(dict(readout='VQC',split=split,**classification(data['y'][split],p)))
                # Initial and final fixed-observable responses use the same source head.
                frozen_geometry,_=evaluate_encoder(result['z'],data['y'],initial_head,fit_probes=False)
                write_json(folder/'frozen_observable_geometry.json',dict(
                    geometry=frozen_geometry['geometry'],responses=frozen_geometry['responses']))
                write_json(folder/'calibration.json',{
                    split:calibration(data['y'][split],p) for split,p in result['probabilities'].items()})
                write_json(folder/'resources.json',dict(result['resources'],stages=result['stages'],
                    final_encoder_diagnostic_direct_qnode_calls=8,
                    final_encoder_diagnostic_direct_sample_states=4*sum(len(a) for a in result['z'].values()),
                    frozen_initial_scoring_qnode_calls=2,
                    frozen_initial_scoring_sample_states=sum(len(a) for a in result['z'].values()),
                    diagnostic_seconds=diagnostic_seconds,scientific_epochs=50,
                    executed_epochs_per_stage=result['executed_epochs_per_stage']))
                for row in rows:all_rows.append(dict(fold=f,seed=s,arm=arm,snapshot='final',**row))
                statuses.append(dict(fold=f,seed=s,arm=arm,status='completed',mode=mode))
                write_json(folder/'status.json',statuses[-1])
            except Exception as exc:
                status=dict(fold=f,seed=s,arm=arm,status='failed',mode=mode,
                            exception=type(exc).__name__,message=str(exc))
                write_json(folder/'status.json',status);statuses.append(status)
                # A technical failure is never replaced by another seed or configuration.
                if not execute_cv:raise
    frame=pd.DataFrame(all_rows)
    frame.to_csv(destination/'metrics.csv',index=False)
    contrasts=paired_contrasts(frame)
    contrasts.to_csv(destination/'paired_Q_minus_E.csv',index=False)
    if len(contrasts):
        fold_means=contrasts.groupby(['schedule','fold','split'],as_index=False)[['log_loss_Q_minus_E','roc_auc_Q_minus_E']].mean()
        fold_means.to_csv(destination/'paired_fold_means.csv',index=False)
        fold_means.groupby(['schedule','split'])[['log_loss_Q_minus_E','roc_auc_Q_minus_E']].agg(['mean','std','min','max']).to_csv(destination/'primary_summary.csv')
    write_json(destination/'run_status.json',dict(mode=mode,planned_primary_pipelines=len(inputs)*4,
        completed=sum(s['status']=='completed' for s in statuses),failures=[s for s in statuses if s['status']!='completed'],
        statuses=statuses,wall_seconds=time.perf_counter()-started,
        peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        peak_rss_unit='bytes' if platform.system()=='Darwin' else 'KiB',torch_threads=1,
        historical_experiments_rerun=False,locked_test_evaluated=False))
    return destination


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--execute-cv',action='store_true',help='Run all 60 pipelines only after separate execution approval')
    parser.add_argument('--audit-folder',default='runs/phase2/c0_v1_availability_audit')
    args=parser.parse_args()
    run(args.output,execute_cv=args.execute_cv,audit_folder=args.audit_folder)


if __name__=='__main__':main()
