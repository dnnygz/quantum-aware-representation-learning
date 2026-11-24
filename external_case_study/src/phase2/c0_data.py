"""Load only audited development PCA32 caches and verified initial checkpoints."""
from pathlib import Path
import hashlib
import json

import numpy as np
import torch

from .data import validate_partitions
from .c0 import initialized_model


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_audited(audit_folder='runs/phase2/c0_v1_availability_audit', repository='.'):
    repository=Path(repository)
    folder=repository/audit_folder
    audit=json.loads((folder/'audit.json').read_text())
    if audit['status']!='passed' or audit['training_executed'] or audit['locked_rows_materialized']:
        raise ValueError('A successful training-free, locked-isolated audit is required')
    if sha256(repository/'reports/phase2/c0_protocol_v1.md')!=audit['protocol_sha256']:
        raise ValueError('Scientific protocol changed after audit')
    manifest=json.loads((repository/'reports/phase2/cv_5fold_50e_v1/provenance.json').read_text())
    split=repository/'data/processed/splits/phase1c_inner_cv.json'
    if sha256(split)!=manifest['protocol_sha256']:
        raise ValueError('Split metadata provenance mismatch')
    protocol=json.loads(split.read_text());validate_partitions(protocol)
    if len(protocol['development_case_ids'])!=835 or len(protocol['locked_test_case_ids'])!=209:
        raise ValueError('Unexpected canonical partition sizes')
    def checked(path):
        digest=sha256(repository/path)
        if manifest['sources'].get(str(path))!=digest or audit['accessed'].get(str(path))!=digest:
            raise ValueError(f'Historical source provenance mismatch: {path}')
        return repository/path
    rows={row['fold']:row for row in audit['folds']}
    if set(rows)!=set(range(5)) or len(audit['folds'])!=5:
        raise ValueError('Exactly five audited folds required')
    labels={}
    import pandas as pd
    for fold in protocol['folds']:
        f=fold['fold'];root=Path('runs/phase2/pilot_fold0_50e_paired_v1' if f==0 else 'runs/phase2/cv_folds1to4_50e_paired_v1')
        # The audit need not read prediction labels; the CV provenance authenticates this source.
        p=root/f'fold_{f}_B/B_seed_1729/validation_predictions.csv'
        if sha256(repository/p)!=manifest['sources'].get(str(p)):
            raise ValueError('Development label source provenance mismatch')
        frame=pd.read_csv(repository/p,dtype={'case_id':str},float_precision='round_trip')
        if frame.case_id.duplicated().any() or set(frame.case_id)!=set(fold['validation_case_ids']):
            raise ValueError('Development label IDs differ from canonical validation partition')
        for row in frame.itertuples():
            if row.case_id in labels or row.er_label not in [0,1]:raise ValueError('Invalid development labels')
            labels[row.case_id]=int(row.er_label)
    if set(labels)!=set(protocol['development_case_ids']):raise ValueError('Development coverage mismatch')
    inputs=[]
    for fold in protocol['folds']:
        f=fold['fold'];row=rows[f]
        expected=folder/f'fold_{f}_pca32.npz'
        if Path(row['pca32_path'])!=Path(audit_folder)/expected.name or sha256(expected)!=row['pca32_sha256']:
            raise ValueError('Audited PCA32 cache provenance mismatch')
        with np.load(expected,allow_pickle=False) as data:
            x={split:data[split].copy() for split in ['train','validation']}
            for split,key,n in [('train','train_case_ids',668),('validation','validation_case_ids',167)]:
                if data[split+'_ids'].tolist()!=fold[key] or x[split].shape!=(n,32) or not np.isfinite(x[split]).all():
                    raise ValueError('PCA32 cache IDs, shape or finite-value check failed')
        y={split:np.array([labels[c] for c in fold[key]]) for split,key in [('train','train_case_ids'),('validation','validation_case_ids')]}
        if sum(y['train'])!=516 or sum(y['validation'])!=129:raise ValueError('Class count mismatch')
        root=Path('runs/phase2/pilot_fold0_50e_paired_v1' if f==0 else 'runs/phase2/cv_folds1to4_50e_paired_v1')
        for s in [1729,1730,1731]:
            source=root/f'fold_{f}_B/B_seed_{s}'
            initial=torch.load(checked(source/'initial_model.pt'),weights_only=True)
            model,_=initialized_model(s)
            if any(not torch.equal(v,initial[k]) for k,v in model.state_dict().items()):raise ValueError('Initial tensor mismatch')
            with np.load(checked(source/'latent_diagnostics.npz'),allow_pickle=False) as old, torch.no_grad():
                for split in x:
                    error=np.abs(model.encoder(torch.tensor(x[split],dtype=torch.float64)).numpy()-old['initial_'+split]).max()
                    if error>1e-12:raise ValueError('Initial embedding reconstruction exceeds tolerance')
            inputs.append(dict(fold=f,seed=s,x=x,y=y,initial_state=initial))
    return inputs,audit
