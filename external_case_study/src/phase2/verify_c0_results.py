"""Independently reconcile saved C0 checkpoints, metrics, streams and history hashes."""
import argparse
import copy
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score,log_loss,average_precision_score,balanced_accuracy_score,brier_score_loss
from sklearn.linear_model import LogisticRegression
from .c0 import ARMS,CLOSED,initialized_model
from .c0_data import load_audited,sha256
from .c0_metrics import geometry,responses


def verify(run,output,audit_folder='runs/phase2/c0_v1_execution_audit'):
    run,output=Path(run),Path(output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    torch.set_num_threads(1)
    inputs,audit=load_audited(audit_folder)
    groups={(r['fold'],r['seed']):r for r in inputs}
    status=json.loads((run/'run_status.json').read_text())
    assert len(status['statuses'])==60
    assert {(r['fold'],r['seed'],r['arm']) for r in status['statuses']}=={(f,s,a) for f in range(5) for s in [1729,1730,1731] for a in ARMS}
    cfg=json.loads((run/'config.json').read_text())
    assert {key:cfg[key] for key in CLOSED}==CLOSED and cfg['technical_epochs'] is None
    saved=pd.read_csv(run/'metrics.csv')
    errors=dict(initial_tensor=0.,encoder_embedding=0.,primary_probability=0.,probe_probability=0.,metrics=0.,trace_distance=0.)
    checked=0;probe_count=0;geometry_rows=[]
    original_fit=LogisticRegression.fit
    def forbidden(*a,**k):raise AssertionError('Fitting is forbidden during result verification')
    LogisticRegression.fit=forbidden
    try:
        for item in status['statuses']:
            if item['status']!='completed':continue
            f,s,arm=item['fold'],item['seed'],item['arm'];data=groups[f,s]
            folder=run/f'fold_{f}_seed_{s}_{arm}'
            model,_=initialized_model(s)
            initial=torch.load(folder/'initial_model.pt',weights_only=True)
            assert all(torch.equal(v,initial[k]) for k,v in data['initial_state'].items())
            final=torch.load(folder/'model.pt',weights_only=True);model.load_state_dict(final)
            if 'NQE' in arm:
                stage=torch.load(folder/'geometry_model.pt',weights_only=True)
                assert torch.equal(stage['head.weights'],initial['head.weights'])
                assert all(torch.equal(v,final[k]) for k,v in stage.items() if k.startswith('encoder.'))
            r=json.loads((folder/'resources.json').read_text());curves=pd.read_csv(folder/'curves.csv')
            stages=['geometry','head'] if 'NQE' in arm else ['joint']
            assert set(curves.stage)==set(stages)
            for stage in stages:
                c=curves[curves.stage==stage]
                assert c.epoch.tolist()==list(range(1,51)) and c.stage_steps.tolist()==[21*k for k in range(1,51)]
            assert r['optimizer_steps']==(2100 if 'NQE' in arm else 1050)
            assert r['pair_draws']==67200 and r['pair_endpoint_occurrences']==134400
            assert r['bce_patient_occurrences']==33400
            assert r['training_qnode_calls']==1050 and r['training_sample_states']==33400
            assert r['validation_qnode_calls']==50 and r['validation_sample_states']==8350
            assert r['executed_epochs_per_stage']==50
            with np.load(folder/'latent_and_predictions.npz',allow_pickle=False) as values:
                for split in ['train','validation']:
                    x=torch.tensor(data['x'][split],dtype=torch.float64)
                    with torch.no_grad():
                        z=model.encoder(x).numpy();p=model(x).numpy()
                    errors['encoder_embedding']=max(errors['encoder_embedding'],float(np.abs(z-values[split+'_z']).max()))
                    errors['primary_probability']=max(errors['primary_probability'],float(np.abs(p-values[split+'_probability']).max()))
                    # Independent 16-dimensional class density calculation from elementary RY amplitudes.
                    a=np.pi*np.tanh(z/2);state=np.ones((len(a),1))
                    for j in range(4):
                        local=np.stack([np.cos(a[:,j]/2),np.sin(a[:,j]/2)],axis=-1)
                        state=(state[:,:,None]*local[:,None,:]).reshape(len(a),-1)
                    y=data['y'][split];rho=[state[y==c].T@state[y==c]/sum(y==c) for c in [0,1]]
                    distance=float(np.abs(np.linalg.eigvalsh(rho[1]-rho[0])).sum()/2)
                    recorded=json.loads((folder/'diagnostics.json').read_text())['geometry'][split]
                    errors['trace_distance']=max(errors['trace_distance'],abs(distance-recorded['class_states']['trace_distance']))
                    assert abs(recorded['observable']['raw_class_mean_contrast'])<=distance+1e-12
                    assert recorded['observable']['ratio_is_complete_discriminative_capacity'] is False
            checked+=1
        # Recompute every saved metric from its stored probabilities, including all fixed readouts.
        for (f,s,arm,snapshot),rows in saved.groupby(['fold','seed','arm','snapshot']):
            data=groups[f,s]
            folder=run/(f'fold_{f}_seed_{s}_initial' if snapshot=='initial' else f'fold_{f}_seed_{s}_{arm}')
            with np.load(folder/'diagnostic_predictions.npz',allow_pickle=False) as preds, (folder/'probes.pkl').open('rb') as stream:
                estimators=pickle.load(stream)
                assert set(estimators)=={'LR_z','LR_angle'};probe_count+=2
                if snapshot=='initial':
                    model,_=initialized_model(s)
                    with torch.no_grad():z={split:model.encoder(torch.tensor(x,dtype=torch.float64)).numpy() for split,x in data['x'].items()}
                else:
                    with np.load(folder/'latent_and_predictions.npz',allow_pickle=False) as a:z={split:a[split+'_z'].copy() for split in ['train','validation']}
                for name,estimator in estimators.items():
                    params=estimator.get_params()
                    for key,value in dict(C=1.,max_iter=2000,solver='lbfgs',tol=1e-4,fit_intercept=True,class_weight=None,warm_start=False,random_state=s).items():assert params[key]==value
                    for split in z:
                        features=z[split] if name=='LR_z' else np.pi*np.tanh(z[split]/2)
                        p=estimator.predict_proba(features)[:,1]
                        errors['probe_probability']=max(errors['probe_probability'],float(np.abs(p-preds[name+'_'+split]).max()))
                for row in rows.itertuples():
                    if row.readout=='VQC':
                        with np.load(folder/'latent_and_predictions.npz',allow_pickle=False) as a:p=a[row.split+'_probability'].copy()
                    else:p=preds[row.readout+'_'+row.split]
                    y=data['y'][row.split]
                    independent=dict(roc_auc=roc_auc_score(y,p),log_loss=log_loss(y,p,labels=[0,1]),
                        ap_er_positive=average_precision_score(y,p),ap_er_negative=average_precision_score(1-y,1-p),
                        balanced_accuracy=balanced_accuracy_score(y,p>=.5),brier=brier_score_loss(y,p),accuracy=np.mean((p>=.5)==y))
                    errors['metrics']=max(errors['metrics'],max(abs(value-getattr(row,key)) for key,value in independent.items()))
        assert probe_count==150
        assert all(value<=1e-12 for value in errors.values()),errors
        costs=[json.loads((run/f'fold_{f}_seed_{s}_{a}'/'resources.json').read_text()) for f,s in groups for a in ARMS]
        for f,s in groups:
            subset=[json.loads((run/f'fold_{f}_seed_{s}_{a}'/'resources.json').read_text()) for a in ARMS]
            for name in ['pair_stream_sha256','patient_stream_sha256']:assert len({r[name] for r in subset})==1
        # New frozen diagnostics on saved historical B embeddings; no original runner is invoked.
        history=json.loads(Path('reports/phase2/cv_5fold_50e_v1/provenance.json').read_text())['sources']
        for f,s in groups:
            root=Path('runs/phase2/pilot_fold0_50e_paired_v1' if f==0 else 'runs/phase2/cv_folds1to4_50e_paired_v1')/f'fold_{f}_B/B_seed_{s}'
            for path in [root/'model.pt',root/'latent_diagnostics.npz']:assert sha256(path)==history[str(path)]
            model,_=initialized_model(s);model.load_state_dict(torch.load(root/'model.pt',weights_only=True))
            with np.load(root/'latent_diagnostics.npz',allow_pickle=False) as a:
                for split in ['train','validation']:
                    g=geometry(a['final_'+split],groups[f,s]['y'][split],model.head)
                    geometry_rows.append(dict(fold=f,seed=s,split=split,Q_balanced_pair_risk=g['pair_geometry']['Q']['balanced_pair_risk'],
                        Q_balanced_separation=g['pair_geometry']['Q']['balanced_separation'],**g['class_states'],**g['observable']))
        pd.DataFrame(geometry_rows).to_csv(output/'historical_B_geometry.csv',index=False)
    finally:LogisticRegression.fit=original_fit
    before=json.loads(Path('runs/phase2/c0_v1_execution_preflight/preservation_before.json').read_text())
    changed=[path for path,value in before.items() if not Path(path).is_file() or sha256(path)!=value]
    assert not changed,changed
    provenance=json.loads((run/'provenance.json').read_text())
    assert all(sha256(path)==value for path,value in provenance['source_sha256'].items())
    run_hashes={str(p.relative_to(run)):sha256(p) for p in run.rglob('*') if p.is_file()}
    (output/'run_artifact_hashes.json').write_text(json.dumps(run_hashes,indent=2)+'\n')
    result=dict(status='passed',successful_primary_pipelines_verified=checked,failed_primary_pipelines=status['failures'],
        metric_rows_recomputed=len(saved),probe_estimators_verified=probe_count,max_absolute_errors=errors,
        initial_and_frozen_block_tensors_verified=True,epochs_and_updates_verified=True,
        pair_and_patient_stream_pairing_verified=True,classification_and_geometry_distinct=True,
        preexisting_files_checked=len(before),changed_or_missing_preexisting_files=changed,
        execution_sources_unchanged=True,run_artifact_files_hashed=len(run_hashes),
        locked_features_or_labels_loaded=False,model_fitting_during_verification=False,
        scope='All 60 saved checkpoints and all saved metric rows; not an independent replication or proof of scientific generalization')
    (output/'integrity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();verify(a.run,a.output)
