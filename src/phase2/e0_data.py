"""Verified development-only saved inputs for E0; no cohort reads or fitting."""
import json
import pickle
from pathlib import Path
import numpy as np
import torch
from .analysis_views import read_csv
from .data import validate_partitions
from .d0 import FrozenHead
from src.data.utils import sha256_file


def array_sha256(array):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def initial_weights(seed):
    generator=torch.Generator(device='cpu').manual_seed(seed)
    return .1*torch.randn((2,4,2),dtype=torch.float64,generator=generator)


def load_inputs(cv_analysis='reports/phase2/cv_5fold_50e_v1',
                protocol_path='data/processed/splits/phase1c_inner_cv.json'):
    manifest=json.loads((Path(cv_analysis)/'provenance.json').read_text())
    hashes=manifest['sources'];accessed={}
    def checked(path):
        path=Path(path);digest=sha256_file(path)
        if hashes.get(str(path))!=digest:
            raise ValueError(f'Historical provenance mismatch: {path}')
        accessed[str(path)]=digest
        return path
    protocol_path=Path(protocol_path)
    if sha256_file(protocol_path)!=manifest['protocol_sha256']:
        raise ValueError('Canonical split metadata hash mismatch')
    protocol=json.loads(protocol_path.read_text());validate_partitions(protocol)
    if len(protocol['development_case_ids'])!=835 or len(protocol['locked_test_case_ids'])!=209:
        raise ValueError('Unexpected development/locked partition sizes')
    folds={f['fold']:f for f in protocol['folds']}
    roots={}
    for root in map(Path,manifest['runs']):
        config=json.loads(checked(root/'config.json').read_text())
        provenance=json.loads(checked(root/'provenance.json').read_text())
        if provenance['locked_test_evaluated'] is not False:
            raise ValueError('Locked-test source prohibited')
        if config['epochs']!=50 or config['seeds']!=[1729,1730,1731]:
            raise ValueError('Historical protocol mismatch')
        for f in config['folds']:
            if f in roots: raise ValueError('Duplicate source fold')
            roots[f]=root
    if set(roots)!=set(range(5)) or set(folds)!=set(range(5)):
        raise ValueError('Exactly five source folds are required')
    labels={}
    for f,root in roots.items():
        frame=read_csv(checked(root/f'fold_{f}_A/A_seed_1729/validation_predictions.csv'))
        if frame.case_id.duplicated().any() or set(frame.case_id)!=set(folds[f]['validation_case_ids']):
            raise ValueError('Saved prediction partition mismatch')
        for row in frame.itertuples():
            if row.case_id in labels: raise ValueError('Repeated development validation coverage')
            labels[row.case_id]=int(row.er_label)
    if set(labels)!=set(protocol['development_case_ids']) or set(labels.values())!={0,1}:
        raise ValueError('Invalid development labels')
    inputs=[]
    for f in range(5):
        fold=folds[f];base=roots[f]/f'fold_{f}_B'
        representation=json.loads(checked(base/'representation.json').read_text())
        if (representation['train_ids']!=fold['train_case_ids'] or
                representation['validation_ids']!=fold['validation_case_ids']):
            raise ValueError('Latent row order mismatch')
        for s in [1729,1730,1731]:
            original=base/f'B_seed_{s}';probe_dir=base/f'B_frozen_probe_seed_{s}'
            initial=torch.load(checked(original/'initial_model.pt'),weights_only=True)
            if not torch.equal(initial['head.weights'],initial_weights(s)):
                raise ValueError('Generated head initialization differs from historical weights')
            state=torch.load(checked(original/'model.pt'),weights_only=True)
            with checked(probe_dir/'model.pkl').open('rb') as stream:
                probe=pickle.load(stream)
            parameters=probe.get_params(deep=False)
            required=dict(C=1.,max_iter=2000,solver='lbfgs',tol=1e-4,fit_intercept=True,
                          class_weight=None,warm_start=False,random_state=s)
            if any(parameters[k]!=v for k,v in required.items()) or parameters.get('l1_ratio',0.)!=0.:
                raise ValueError('Historical logistic-regression settings differ from approval')
            with np.load(checked(original/'latent_diagnostics.npz'),allow_pickle=False) as archive:
                z={split:archive[f'final_{split}'].copy() for split in ['train','validation']}
            y={split:np.array([labels[c] for c in fold[key]]) for split,key in
                [('train','train_case_ids'),('validation','validation_case_ids')]}
            for split,n in [('train',668),('validation',167)]:
                if z[split].shape!=(n,4) or not np.isfinite(z[split]).all():
                    raise ValueError('Invalid saved B embeddings')
                z[split].setflags(write=False);y[split].setflags(write=False)
            historical={}
            for name,folder,head in [('B_historical',original,FrozenHead('B',state)),
                                     ('LR_z_historical',probe_dir,None)]:
                saved=read_csv(checked(folder/'validation_predictions.csv'))
                if saved.case_id.duplicated().any() or set(saved.case_id)!=set(fold['validation_case_ids']):
                    raise ValueError('Historical validation IDs mismatch')
                ordered=saved.set_index('case_id').loc[fold['validation_case_ids']]
                np.testing.assert_array_equal(ordered.er_label,y['validation'])
                predictions={split:head.predict(z[split]) if head else probe.predict_proba(z[split])[:,1]
                             for split in ['train','validation']}
                np.testing.assert_allclose(predictions['validation'],ordered.probability_er_positive,atol=1e-12,rtol=0)
                historical[name]=dict(probabilities=predictions,metrics=json.loads(checked(folder/'metrics.json').read_text()))
            inputs.append(dict(fold=f,encoder_seed=s,z=z,y=y,probe=probe,probe_parameters=parameters,
                historical=historical,ids={split:fold[key] for split,key in
                    [('train','train_case_ids'),('validation','validation_case_ids')]},
                original_checkpoint=str(original/'model.pt'),
                embedding_sha256={split:array_sha256(a) for split,a in z.items()}))
    return inputs,accessed,manifest['versions']
