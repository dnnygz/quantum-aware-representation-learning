"""Execute the closed approved E0.1/E0.2 experiment on saved B embeddings."""
import argparse
import importlib.metadata
import json
import pickle
import platform
import resource
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.base import clone
from .d0 import phi
from .e0 import validate_config, fit_head
from .e0_data import load_inputs, array_sha256
from .evaluation import metrics
from .run import write_json
from src.data.utils import sha256_file


def score(y,p):
    result=metrics(y,p)
    return {k:v for k,v in result.items() if k!='confusion_matrix'}|dict(brier=float(np.mean((p-y)**2)))


def run(config_path,output,protocol_path='reports/phase2/e0_protocol_v1.md'):
    started=time.perf_counter();output=Path(output)
    if output.exists(): raise FileExistsError(output)
    config=yaml.safe_load(Path(config_path).read_text());validate_config(config)
    protocol_path=Path(protocol_path)
    if not protocol_path.is_file(): raise FileNotFoundError('Approved protocol is required before fitting')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    load_started=time.perf_counter()
    inputs,sources,versions=load_inputs()
    current={n:importlib.metadata.version(n) for n in versions}
    if current!=versions: raise ValueError('Dependency versions differ from the historical paired experiment')
    loading_seconds=time.perf_counter()-load_started
    output.mkdir(parents=True,exist_ok=False)
    snapshot=output/'source';snapshot.mkdir()
    for path in Path(__file__).parent.glob('*.py'): shutil.copy2(path,snapshot/path.name)
    shutil.copy2(protocol_path,output/'approved_protocol.md')
    resolved=dict(**config,source_config_sha256=sha256_file(Path(config_path)),
        protocol_sha256=sha256_file(protocol_path),versions=current,python=platform.python_version(),
        classical_settings={f"fold_{x['fold']}_encoder_{x['encoder_seed']}":x['probe_parameters'] for x in inputs},
        source_loading_seconds=loading_seconds,device='CPU default.qubit',dtype='float64',
        shots=None,head_only=True,locked_test_access=False)
    write_json(output/'config.json',resolved)
    plan=[dict(fold=x['fold'],encoder_seed=x['encoder_seed'],model=name,head_seed=h)
          for x in inputs for name,hs in [('VQC_refit',config['head_seeds']),('LR_angle',[-1])] for h in hs]
    write_json(output/'planned_design.json',plan)
    rows=[];costs=[];statuses=[];curves_all=[]
    for item in inputs:
        f,s=item['fold'],item['encoder_seed']
        base=output/f'fold_{f}_encoder_{s}';base.mkdir()
        # Input snapshots are references/hashes, not duplicated raw cohort data.
        write_json(base/'input_reference.json',dict(original_checkpoint=item['original_checkpoint'],
            embedding_sha256=item['embedding_sha256'],encoder_executed=False,preprocessing_fitted=False))
        mapping_start=time.perf_counter()
        angular={split:phi(z) for split,z in item['z'].items()}
        angular_seconds=time.perf_counter()-mapping_start
        for name,reference in item['historical'].items():
            for split,p in reference['probabilities'].items():
                rows.append(dict(fold=f,encoder_seed=s,head_seed=-1,model=name,split=split,diagonal=False,
                                 **score(item['y'][split],p)))
            write_json(base/f'{name}_source_metrics.json',reference['metrics'])
        for name,heads in [('VQC_refit',config['head_seeds']),('LR_angle',[-1])]:
            for h in heads:
                target=base/(f'VQC_head_{h}' if name=='VQC_refit' else 'LR_angle');target.mkdir()
                key=dict(fold=f,encoder_seed=s,head_seed=h,model=name)
                start=time.perf_counter()
                try:
                    if name=='VQC_refit':
                        head,initial,predictions,curve,timing=fit_head(item['z']['train'],item['y']['train'],
                            item['z']['validation'],item['y']['validation'],s,h,config)
                        torch.save(initial,target/'initial_model.pt');torch.save(head.state_dict(),target/'model.pt')
                        curve_table=pd.DataFrame(curve);curve_table.to_csv(target/'curves.csv',index=False)
                        curves_all.append(curve_table.assign(**key))
                        warning_text=[];iterations=-1
                    else:
                        estimator=clone(item['probe'])
                        fit_start=time.perf_counter()
                        with warnings.catch_warnings(record=True) as recorded:
                            warnings.simplefilter('always')
                            estimator.fit(angular['train'],item['y']['train'])
                        fit_seconds=time.perf_counter()-fit_start
                        prediction_start=time.perf_counter()
                        predictions={split:estimator.predict_proba(angular[split])[:,1] for split in ['train','validation']}
                        prediction_seconds=time.perf_counter()-prediction_start
                        warning_text=[dict(category=w.category.__name__,message=str(w.message)) for w in recorded]
                        iterations=int(np.max(estimator.n_iter_))
                        with (target/'model.pkl').open('wb') as stream: pickle.dump(estimator,stream)
                        timing=dict(fit_update_seconds=fit_seconds,epoch_validation_seconds=0.,
                            final_prediction_seconds=prediction_seconds,training_qnode_calls=0,
                            training_sample_state_evaluations=0,total_qnode_calls=0,total_sample_state_evaluations=0,
                            optimizer_steps=-1,trainable_parameters=5,shuffle_sha256='not_applicable')
                    result=[]
                    for split,p in predictions.items():
                        result.append(dict(**key,split=split,diagonal=name=='VQC_refit' and h==s,
                                           **score(item['y'][split],p)))
                        pd.DataFrame(dict(case_id=item['ids'][split],er_label=item['y'][split],
                            probability_er_positive=p)).to_csv(target/f'{split}_predictions.csv',index=False)
                    rows.extend(result);write_json(target/'metrics.json',result)
                    cost=dict(**key,**timing,angular_reference_seconds=angular_seconds,
                        elapsed_fit_and_save_seconds=time.perf_counter()-start,iterations=iterations)
                    costs.append(cost);write_json(target/'cost.json',cost)
                    status=dict(**key,status='completed',warnings=warning_text,iterations=iterations)
                except Exception as error:
                    status=dict(**key,status='failed',error_type=type(error).__name__,message=str(error))
                statuses.append(status);write_json(target/'status.json',status)
                write_json(output/'statuses.json',statuses)
                print(f"{status['status']}: {name}, fold={f}, encoder_seed={s}, head_seed={h}",flush=True)
        for split,z in item['z'].items():
            if array_sha256(z)!=item['embedding_sha256'][split]:
                raise ValueError('A frozen embedding changed during head fitting')
    pd.DataFrame(rows).to_csv(output/'metrics.csv',index=False)
    pd.DataFrame(costs).to_csv(output/'costs.csv',index=False)
    if curves_all: pd.concat(curves_all,ignore_index=True).to_csv(output/'curves.csv',index=False)
    for path,digest in sources.items():
        if sha256_file(Path(path))!=digest: raise ValueError(f'Historical source changed: {path}')
    write_json(output/'provenance.json',dict(protocol='E0-v1-approved-closed',
        stage='development head-fitting controls on frozen historical B representations',
        sources=sources,source_sha256={p.name:sha256_file(p) for p in snapshot.glob('*.py')},
        config_sha256=sha256_file(output/'config.json'),protocol_sha256=sha256_file(output/'approved_protocol.md'),
        git_commit=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip(),
        versions=current,python=platform.python_version(),planned_fits=60,
        completed_fits=sum(s['status']=='completed' for s in statuses),failed_fits=sum(s['status']=='failed' for s in statuses),
        historical_reference_heads=30,encoder_or_preprocessing_fitted=False,raw_cohort_loaded=False,
        locked_test_features_or_labels_accessed=False,
        primary_e01=config['primary_e01'],primary_e02=config['primary_e02'],
        phase_closed=True,extended_search_performed=False,
        wall_seconds=time.perf_counter()-started,source_loading_seconds=loading_seconds,
        process_peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        peak_rss_unit='bytes' if sys.platform=='darwin' else 'KiB',torch_threads=torch.get_num_threads()))
    return pd.DataFrame(rows),statuses


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--config',default='configs/e0_v1.yaml')
    parser.add_argument('--output',required=True)
    args=parser.parse_args();_,statuses=run(args.config,args.output)
    if any(s['status']!='completed' for s in statuses): raise SystemExit('E0 contains recorded failed fits; no replacements were trained')


if __name__=='__main__': main()
