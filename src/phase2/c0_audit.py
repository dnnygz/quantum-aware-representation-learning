"""Audit historical development inputs without fitting or changing source artifacts."""
from pathlib import Path
import hashlib,json,pickle,time,traceback,sys
import numpy as np
import torch
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from src.phase2.data import validate_partitions
from src.phase2.models import Model


def audit(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    report={'status':'running','training_executed':False,'locked_rows_materialized':False,'accessed':{},'folds':[],'reconstruction_tolerance':1e-12,'protocol_sha256':sha('reports/phase2/c0_protocol_v1.md')}
    start=time.perf_counter()
    try:
     manifest=json.loads(Path('reports/phase2/cv_5fold_50e_v1/provenance.json').read_text())
     def checked(p):
      p=Path(p);digest=sha(p)
      if manifest['sources'].get(str(p))!=digest:raise ValueError(f'Historical hash mismatch: {p}')
      report['accessed'][str(p)]=digest
      return p
     splitpath=Path('data/processed/splits/phase1c_inner_cv.json')
     assert sha(splitpath)==manifest['protocol_sha256'],'Split metadata hash mismatch'
     protocol=json.loads(splitpath.read_text());validate_partitions(protocol)
     assert len(protocol['development_case_ids'])==835 and len(protocol['locked_test_case_ids'])==209
     pending=[]
     for fold in protocol['folds']:
      f=fold['fold'];root=Path('runs/phase2/pilot_fold0_50e_paired_v1' if f==0 else 'runs/phase2/cv_folds1to4_50e_paired_v1')
      base=root/f'fold_{f}_B'
      config=json.loads(checked(root/'config.json').read_text())
      prov=json.loads(checked(root/'provenance.json').read_text())
      assert config['epochs']==50 and config['seeds']==[1729,1730,1731] and f in config['folds']
      assert prov['locked_test_evaluated'] is False
      meta=json.loads(checked(base/'representation.json').read_text())
      assert meta['train_ids']==fold['train_case_ids'] and meta['validation_ids']==fold['validation_case_ids']
      with checked(base/'representation.pkl').open('rb') as stream: rep=pickle.load(stream)
      assert rep.training_ids==fold['train_case_ids']
      assert rep.pca.n_components_==32 and rep.pca.n_samples_==668
      assert rep.scaler.n_samples_seen_==668
      assert rep.rna.tmm_.reference_sample_id_ in fold['train_case_ids']
      states=[]
      for s in [1729,1730,1731]:
       directory=base/f'B_seed_{s}'
       state=torch.load(checked(directory/'initial_model.pt'),weights_only=True)
       with torch.random.fork_rng(devices=[]):
        torch.manual_seed(s);model=Model(32,encoder=True)
       assert all(torch.equal(v,state[k]) for k,v in model.state_dict().items()),f'Initialization mismatch fold {f} seed {s}'
       with np.load(checked(directory/'latent_diagnostics.npz'),allow_pickle=False) as z:
        saved={split:z[f'initial_{split}'].copy() for split in ['train','validation']}
       assert saved['train'].shape==(668,4) and saved['validation'].shape==(167,4)
       states.append((s,model,saved))
      pending.append((fold,rep,states,prov))
     print('Verified five historical preprocessors and all 15 initial checkpoints before data transformation.',flush=True)
     expression=Path('data/processed/expression.parquet')
     # Compare only opaque bytes to the historical fingerprint; never materialize locked rows.
     digest=sha(expression)
     assert all(p[3]['input_sha256'][str(expression)]==digest for p in pending),'Expression provenance mismatch'
     schema=pq.read_schema(expression);index=json.loads(schema.metadata[b'pandas'])['index_columns'][0]
     frame=ds.dataset(expression,format='parquet').to_table(filter=ds.field(index).isin(protocol['development_case_ids'])).to_pandas()
     frame.index=frame.index.astype(str)
     assert set(frame.index)==set(protocol['development_case_ids']) and len(frame)==835
     report['development_expression_sha256']=digest
     def forbidden(*args,**kwargs):raise AssertionError('Fitting forbidden during availability audit')
     torch.set_num_threads(1)
     for fold,rep,states,prov in pending:
      for obj in [rep,rep.rna,rep.pca,rep.scaler,rep.rna.expression_filter_,rep.rna.tmm_,rep.rna.zero_variance_filter_,rep.rna.scaler_]:obj.fit=forbidden
      arrays={split:rep.transform(frame.loc[fold[key]]) for split,key in [('train','train_case_ids'),('validation','validation_case_ids')]}
      row={'fold':fold['fold'],'train_shape':list(arrays['train'].shape),'validation_shape':list(arrays['validation'].shape),'seeds':[]}
      for s,model,saved in states:
       errors={}
       with torch.no_grad():
        for split,x in arrays.items():
         error=float(np.abs(model.encoder(torch.tensor(x,dtype=torch.float64)).numpy()-saved[split]).max())
         errors[split]=error
         if error>1e-12:raise ValueError(f'Initial embedding mismatch fold {fold["fold"]} seed {s} split {split}: {error}')
       row['seeds'].append({'seed':s,'max_initial_embedding_errors':errors})
      target=out/f'fold_{fold["fold"]}_pca32.npz'
      np.savez_compressed(target,**arrays,train_ids=np.array(fold['train_case_ids']),validation_ids=np.array(fold['validation_case_ids']))
      row['pca32_path']=str(target);row['pca32_sha256']=sha(target)
      report['folds'].append(row)
      print('Reconstructed and verified fold',fold['fold'],flush=True)
     report['status']='passed'
    except Exception as exc:
     report['status']='failed';report['error_type']=type(exc).__name__;report['error']=str(exc)
     traceback.print_exc()
    finally:
     report['wall_seconds']=time.perf_counter()-start
     (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
     print(json.dumps({k:v for k,v in report.items() if k not in ['accessed','folds']},indent=2),flush=True)
    if report['status']!='passed':raise RuntimeError(report['error'])
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    audit(parser.parse_args().output)
