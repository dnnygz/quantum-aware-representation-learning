"""Synthetic C0-v1 scientific invariants, isolation, and technical integration."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.phase2 import c0
from src.phase2.c0_metrics import (kernel, state_vectors, reference_scores,
                                 geometry, responses, evaluate_encoder)


@pytest.fixture
def fixture():
    return c0.synthetic_fixture()


def fit(arm, fixture, **kwargs):
    x,y,xv,yv=fixture
    model,_=c0.initialized_model(1729)
    return c0.fit_pipeline(x,y,xv,yv,0,1729,arm,model.state_dict(),c0.CLOSED,
                           synthetic=True,technical_epochs=2,**kwargs)


@pytest.mark.parametrize('field,value', [('lambda_aux',.5),('epochs',25),('pair_count',32),
    ('gaussian_denominator',2.),('seeds',[1729]),('selection','best_validation')])
def test_closed_protocol(field,value):
    changed=copy.deepcopy(c0.CLOSED);changed[field]=value
    with pytest.raises(ValueError):c0.validate_config(changed)


def test_yaml_matches_closed_protocol():
    c0.validate_config(yaml.safe_load(Path('configs/c0_v1.yaml').read_text()))


def test_pair_sampling_strata_diagonals_and_reproducibility():
    y=np.array([1]*516+[0]*152)
    a=np.random.Generator(np.random.PCG64(np.random.SeedSequence([1729,0,1])))
    b=np.random.Generator(np.random.PCG64(np.random.SeedSequence([1729,0,1])))
    for _ in range(8):
        pairs=c0.pair_indices(y,a)
        np.testing.assert_array_equal(pairs,c0.pair_indices(y,b))
        assert pairs.shape==(64,2) and np.all(pairs[:,0]!=pairs[:,1])
        assert np.all(y[pairs[:16]]==1) and np.all(y[pairs[16:32]]==0)
        assert np.all(y[pairs[32:,0]]==1) and np.all(y[pairs[32:,1]]==0)
    with pytest.raises(ValueError):c0.pair_indices(np.array([0,1,1]),a)


@pytest.mark.parametrize('kind',['Q','E'])
def test_exact_objective_labels_both_endpoint_gradients(kind):
    encoder=torch.nn.Linear(32,4).double()
    generator=torch.Generator().manual_seed(1729)
    x=torch.randn(20,32,generator=generator,dtype=torch.float64,requires_grad=True)
    y=torch.tensor([0,1]*10,dtype=torch.float64)
    pairs=c0.pair_indices(y.numpy(),np.random.default_rng(1729))
    actual=c0.pair_loss(encoder,x,y,pairs,kind)
    a=np.pi*np.tanh(encoder(x).detach().numpy()/2)
    expected=(kernel(a,a,kind)[pairs[:,0],pairs[:,1]]-(y.numpy()[pairs[:,0]]==y.numpy()[pairs[:,1]]))**2
    np.testing.assert_allclose(float(actual.detach()),expected.mean(),atol=1e-14)
    signed=2*y-1
    assert torch.equal((1+signed[pairs[:,0]]*signed[pairs[:,1]])/2,
                       (y[pairs[:,0]]==y[pairs[:,1]]).double())
    gx,gw=torch.autograd.grad(actual,(x,encoder.weight))
    assert gx.norm()>0 and gw.norm()>0 and torch.isfinite(gx).all()
    # An endpoint that appears only in the second position must still differentiate.
    pair=np.tile([0,1],(64,1))
    x2=x.detach().clone().requires_grad_()
    gradient=torch.autograd.grad(c0.pair_loss(encoder,x2,y,pair,kind),x2)[0]
    assert gradient[0].norm()>0 and gradient[1].norm()>0
    bad=pairs.copy();bad[0,1]=bad[0,0]
    with pytest.raises(ValueError):c0.pair_loss(encoder,x,y,bad,kind)


def test_local_gaussian_scale_and_density_geometry():
    a=np.zeros((1,4));b=np.array([[.001,-.001,.001,-.001]])
    assert abs(kernel(a,b,'Q')[0,0]-kernel(a,b,'E')[0,0])<1e-12
    points=np.array([[0.,0.,0.,0.],[1.,-.7,.4,.2]])
    states=state_vectors(points)
    np.testing.assert_allclose(kernel(points,points,'Q'),(states@states.T)**2,atol=1e-14)
    with pytest.raises(ValueError):c0.similarity(torch.zeros(2,4),torch.zeros(2,4),'X')


@pytest.mark.parametrize('arm',c0.ARMS)
def test_pipeline_stage_membership_fixed_budgets_and_streams(arm,fixture):
    result=fit(arm,fixture)
    r=result['resources']
    assert r['pair_draws']==256 and r['pair_endpoint_occurrences']==512
    assert r['bce_patient_occurrences']==128
    assert r['optimizer_steps']==(8 if 'NQE' in arm else 4)
    assert r['training_qnode_calls']==4 and r['training_sample_states']==128
    assert r['validation_qnode_calls']==2 and r['validation_sample_states']==32
    assert r['total_qnode_calls']==sum(r[k] for k in ['training_qnode_calls','validation_qnode_calls','diagnostic_qnode_calls','final_prediction_qnode_calls'])
    if 'NQE' in arm:
        stage=result['geometry_state'];initial=result['initial_state'];final=result['model'].state_dict()
        assert torch.equal(stage['head.weights'],initial['head.weights'])
        assert all(torch.equal(v,final[k]) for k,v in stage.items() if k.startswith('encoder.'))
        assert not torch.equal(final['head.weights'],initial['head.weights'])
    assert result['status']=='synthetic_technical'
    assert result['scientific_epochs']==50 and result['executed_epochs_per_stage']==2


def test_determinism_pairing_and_validation_label_isolation(fixture):
    q=fit('C0-Hybrid-Q',fixture)
    again=fit('C0-Hybrid-Q',fixture)
    flipped=(*fixture[:3],1-fixture[3])
    changed_labels=fit('C0-Hybrid-Q',flipped)
    e=fit('C0-Hybrid-E',fixture)
    for other in [again,changed_labels]:
        for key,value in q['model'].state_dict().items():assert torch.equal(value,other['model'].state_dict()[key])
    assert q['resources']['pair_stream_sha256']==e['resources']['pair_stream_sha256']
    assert q['resources']['patient_stream_sha256']==e['resources']['patient_stream_sha256']
    nqe=fit('C0-NQE-Q',fixture)
    assert q['resources']['pair_stream_sha256']==nqe['resources']['pair_stream_sha256']
    assert q['resources']['patient_stream_sha256']==nqe['resources']['patient_stream_sha256']


def test_optimizer_excludes_frozen_parameters_even_with_weight_decay():
    model,_=c0.initialized_model(1729)
    optimizer=c0._optimizer(model,True,False)
    assert {id(p) for g in optimizer.param_groups for p in g['params']}=={id(p) for p in model.encoder.parameters()}
    initial=model.head.weights.detach().clone()
    for p in model.encoder.parameters():p.grad=torch.ones_like(p)
    optimizer.step()
    assert torch.equal(initial,model.head.weights)


def test_component_diagnostic_no_rng_parameter_or_gradient_mutation(fixture):
    x,y,_,_=fixture;model,_=c0.initialized_model(1729)
    state=c0.tensor_hash(model.state_dict());rng=torch.random.get_rng_state().clone()
    result=c0.component_gradients(model,torch.tensor(x),torch.tensor(y,dtype=torch.float64),0,1729,'Q')
    assert state==c0.tensor_hash(model.state_dict()) and torch.equal(rng,torch.random.get_rng_state())
    assert all(p.grad is None for p in model.parameters())
    assert result['ratio']>=0


def test_probe_fitting_uses_training_only_and_references_exclude_self(monkeypatch):
    rng=np.random.default_rng(1729)
    z=dict(train=rng.normal(size=(12,4)),validation=rng.normal(size=(6,4)))
    labels=dict(train=np.array([0,1]*6),validation=np.array([0,1]*3))
    from sklearn.linear_model import LogisticRegression
    original=LogisticRegression.fit;seen=[]
    def checked(self,x,y,*args,**kwargs):
        assert len(x)==12 and np.array_equal(y,labels['train']);seen.append(len(x))
        return original(self,x,y,*args,**kwargs)
    monkeypatch.setattr(LogisticRegression,'fit',checked)
    model,_=c0.initialized_model(1729)
    result,_=evaluate_encoder(z,labels,model.head)
    assert seen==[12,12]
    a=np.pi*np.tanh(z['train']/2)
    p=reference_scores(a,labels['train'],a,'Q',training=True)
    k=kernel(a,a,'Q')
    for i in range(len(a)):
        plus=[j for j in range(len(a)) if labels['train'][j]==1 and j!=i]
        minus=[j for j in range(len(a)) if labels['train'][j]==0 and j!=i]
        assert abs(p[i]-(1+k[i,plus].mean()-k[i,minus].mean())/2)<1e-12
    g=result['geometry']['validation']
    assert not g['observable']['ratio_is_complete_discriminative_capacity']
    assert abs(g['observable']['raw_class_mean_contrast'])<=g['class_states']['trace_distance']+1e-12
    assert result['responses']['train']['chain_rule_max_error']<1e-10


def test_short_real_budget_and_unsafe_output_are_rejected(fixture):
    model,_=c0.initialized_model(1729)
    with pytest.raises(ValueError):c0.fit_pipeline(*fixture,0,1729,'C0-NQE-Q',model.state_dict(),c0.CLOSED,technical_epochs=2)
    from src.phase2.run_c0 import run
    with pytest.raises(ValueError):run('reports/unsafe_c0_outputs')


def test_technical_smoke_never_loads_patient_inputs(monkeypatch,tmp_path):
    import src.phase2.run_c0 as runner
    def forbidden(*a,**k):raise AssertionError('Patient inputs accessed during smoke')
    monkeypatch.setattr(runner,'load_audited',forbidden)
    # A repository-local technical artifact destination remains ignored.
    target=Path('runs/phase2')/('pytest_c0_'+tmp_path.parent.name+'_'+tmp_path.name)
    destination=runner.run(target)
    status=json.loads((destination/'run_status.json').read_text())
    assert status['mode']=='synthetic_technical_smoke' and status['completed']==4
    assert status['planned_primary_pipelines']==4 and status['failures']==[]
    assert json.loads((destination/'provenance.json').read_text())['synthetic_only']


def test_zero_auxiliary_matches_original_updates_on_synthetic_fixture(fixture):
    from src.phase2.training import train
    x,y,xv,yv=[torch.tensor(a,dtype=torch.float64) for a in fixture]
    original,stream=c0.initialized_model(1729)
    candidate,_=c0.initialized_model(1729)
    config=dict(epochs=1,batch_size=32,lr=.01,weight_decay=.001)
    with torch.random.fork_rng(devices=[]):
        torch.random.set_rng_state(stream)
        train(original,x,y,xv,yv,config)
    optimizer=c0._optimizer(candidate,True,True)
    generator=torch.Generator().set_state(stream)
    pairs_rng=np.random.default_rng(1729)
    for batch in torch.randperm(len(x),generator=generator).split(32):
        optimizer.zero_grad()
        bce=torch.nn.functional.binary_cross_entropy(candidate(x[batch]),y[batch])
        auxiliary=c0.pair_loss(candidate.encoder,x,y,c0.pair_indices(y.numpy(),pairs_rng),'Q')
        (bce+0*auxiliary).backward();optimizer.step()
    for name,value in candidate.state_dict().items():
        assert torch.equal(value,original.state_dict()[name])


@pytest.fixture
def synthetic_audit(tmp_path):
    """Build the full canonical dimensions using invented IDs and random PCA32 arrays."""
    import hashlib
    from src.phase2.c0_data import sha256
    protocol=dict(development_case_ids=[f'dev{i}' for i in range(835)],
                  locked_test_case_ids=[f'locked{i}' for i in range(209)],folds=[])
    label_map={}
    for f in range(5):
        validation=protocol['development_case_ids'][167*f:167*(f+1)]
        training=[p for p in protocol['development_case_ids'] if p not in validation]
        protocol['folds'].append(dict(fold=f,train_case_ids=training,validation_case_ids=validation))
        label_map.update({p:int(i<129) for i,p in enumerate(validation)})
    def dump(path,value):
        path=tmp_path/path;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value));return path
    split=dump(Path('data/processed/splits/phase1c_inner_cv.json'),protocol)
    scientific=tmp_path/'reports/phase2/c0_protocol_v1.md'
    scientific.parent.mkdir(parents=True,exist_ok=True);scientific.write_text('Synthetic protocol fixture')
    sources={};audit=dict(status='passed',training_executed=False,locked_rows_materialized=False,
                         protocol_sha256=sha256(scientific),accessed={},folds=[])
    cache=Path('runs/phase2/c0_v1_availability_audit')
    rng=np.random.default_rng(1729)
    for fold in protocol['folds']:
        f=fold['fold'];root=Path('runs/phase2/pilot_fold0_50e_paired_v1' if f==0 else 'runs/phase2/cv_folds1to4_50e_paired_v1')
        arrays=dict(train=rng.normal(size=(668,32)),validation=rng.normal(size=(167,32)))
        p=tmp_path/cache/f'fold_{f}_pca32.npz';p.parent.mkdir(parents=True,exist_ok=True)
        np.savez(p,**arrays,train_ids=fold['train_case_ids'],validation_ids=fold['validation_case_ids'])
        audit['folds'].append(dict(fold=f,pca32_path=str(cache/p.name),pca32_sha256=sha256(p)))
        for s in [1729,1730,1731]:
            folder=root/f'fold_{f}_B/B_seed_{s}'
            (tmp_path/folder).mkdir(parents=True,exist_ok=True)
            model,_=c0.initialized_model(s)
            checkpoint=folder/'initial_model.pt';torch.save(model.state_dict(),tmp_path/checkpoint)
            latent=folder/'latent_diagnostics.npz'
            with torch.no_grad():np.savez(tmp_path/latent,**{
                'initial_'+k:model.encoder(torch.tensor(v)).numpy() for k,v in arrays.items()})
            for source in [checkpoint,latent]:
                sources[str(source)]=sha256(tmp_path/source);audit['accessed'][str(source)]=sources[str(source)]
        import pandas as pd
        labels= root/f'fold_{f}_B/B_seed_1729/validation_predictions.csv'
        pd.DataFrame(dict(case_id=fold['validation_case_ids'],er_label=[label_map[p] for p in fold['validation_case_ids']])).to_csv(tmp_path/labels,index=False)
        sources[str(labels)]=sha256(tmp_path/labels)
    dump(Path('reports/phase2/cv_5fold_50e_v1/provenance.json'),dict(sources=sources,protocol_sha256=sha256(split)))
    dump(cache/'audit.json',audit)
    return tmp_path,cache


def test_audited_loader_positive_without_cohort_or_locked_features(synthetic_audit):
    from src.phase2.c0_data import load_audited
    root,cache=synthetic_audit
    assert not (root/'data/processed/expression.parquet').exists()
    inputs,_=load_audited(cache,root)
    assert len(inputs)==15
    assert all(row['x']['train'].shape==(668,32) and row['x']['validation'].shape==(167,32) for row in inputs)


@pytest.mark.parametrize('corruption',['cache_hash','cache_ids','locked_overlap','initial_checkpoint','protocol_hash'])
def test_audited_loader_rejects_corrupt_provenance_and_partitions(synthetic_audit,corruption):
    from src.phase2.c0_data import load_audited,sha256
    root,cache=synthetic_audit
    audit_path=root/cache/'audit.json';audit=json.loads(audit_path.read_text())
    if corruption.startswith('cache'):
        p=root/cache/'fold_0_pca32.npz'
        with np.load(p) as old:data={k:old[k].copy() for k in old.files}
        if corruption=='cache_hash':data['train'][0,0]+=1
        else:data['train_ids'][0],data['train_ids'][1]=data['train_ids'][1],data['train_ids'][0]
        np.savez(p,**data)
        if corruption=='cache_ids':audit['folds'][0]['pca32_sha256']=sha256(p)
    elif corruption=='locked_overlap':
        p=root/'data/processed/splits/phase1c_inner_cv.json';split=json.loads(p.read_text())
        split['folds'][0]['train_case_ids'][0]='locked0';p.write_text(json.dumps(split))
        manifest=root/'reports/phase2/cv_5fold_50e_v1/provenance.json'
        value=json.loads(manifest.read_text());value['protocol_sha256']=sha256(p);manifest.write_text(json.dumps(value))
    elif corruption=='initial_checkpoint':
        p=root/'runs/phase2/pilot_fold0_50e_paired_v1/fold_0_B/B_seed_1729/initial_model.pt'
        state=torch.load(p,weights_only=True);state['head.weights']+=1;torch.save(state,p)
    else:(root/'reports/phase2/c0_protocol_v1.md').write_text('Changed protocol')
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(ValueError):load_audited(cache,root)


@pytest.mark.parametrize('kind',['Q','E'])
def test_auxiliary_head_gradient_is_absent_and_joint_head_gradient_is_bce(kind,fixture):
    x,y,_,_=fixture
    model,_=c0.initialized_model(1729)
    x=torch.tensor(x);y=torch.tensor(y,dtype=torch.float64)
    aux=c0.pair_loss(model.encoder,x,y,c0.pair_indices(y.numpy(),np.random.default_rng(1729)),kind)
    assert torch.autograd.grad(aux,model.head.weights,allow_unused=True,retain_graph=True)[0] is None
    bce=torch.nn.functional.binary_cross_entropy(model(x),y)
    bce_head=torch.autograd.grad(bce,model.head.weights,retain_graph=True)[0]
    joint_head=torch.autograd.grad(bce+aux,model.head.weights)[0]
    torch.testing.assert_close(joint_head,bce_head,atol=0,rtol=0)
    generator=torch.Generator().manual_seed(1729)
    a=torch.randn(2,4,generator=generator,dtype=torch.float64,requires_grad=True)
    b=torch.randn(2,4,generator=generator,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda u,v:c0.similarity(u,v,kind),(a,b),eps=1e-6,atol=1e-6)


def test_primary_contrast_direction_and_no_readout_selection():
    import pandas as pd
    from src.phase2.run_c0 import paired_contrasts
    rows=[]
    for schedule in ['NQE','Hybrid']:
        for kind,value in [('Q',.2),('E',.3)]:
            rows.append(dict(arm=f'C0-{schedule}-{kind}',snapshot='final',readout='VQC',
                             fold=0,seed=1729,split='validation',log_loss=value,roc_auc=1-value))
        rows.append(dict(rows[-1],readout='LR_z',log_loss=999.))
    contrast=paired_contrasts(pd.DataFrame(rows))
    np.testing.assert_allclose(contrast.log_loss_Q_minus_E,-.1)
    np.testing.assert_allclose(contrast.roc_auc_Q_minus_E,.1)


def test_architecture_source_guard_precedes_any_fitting(monkeypatch):
    import src.phase2.run_c0 as runner
    original=runner.sha256
    monkeypatch.setattr(runner,'sha256',lambda p:'changed' if str(p)=='src/phase2/models.py' else original(p))
    with pytest.raises(ValueError,match='architecture source changed'):
        runner.run('runs/phase2/must_not_be_created_c0_source_guard')
    assert not Path('runs/phase2/must_not_be_created_c0_source_guard').exists()
