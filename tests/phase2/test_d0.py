"""D0 numerical identities, training-only interventions, pairing, and preservation guards."""
from pathlib import Path
import numpy as np
import pytest
import torch
from src.phase2.models import Model
from src.phase2.d0 import (FrozenHead, training_reference, interventions, phi,
                          displacement, predictive_change)
from src.phase2.run_d0 import run


def samples():
    rng=np.random.default_rng(1729)
    z=rng.normal(size=(20,4));z[:,3]=.7
    return z,np.array([0,1]*10)


def head(name='B'):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1729)
        state=Model(32,encoder=True,quantum=name!='B_classical').state_dict()
    return FrozenHead(name,state)


def test_chain_rule_separates_map_from_circuit_and_matches_fd():
    h=head();z,_=samples();z[0,:]=[0,2,10,-10]
    d=h.derivatives(z)
    expected=np.pi/2*(1-np.tanh(z/2)**2)
    np.testing.assert_allclose(d['map_derivative'],expected,rtol=0,atol=1e-14)
    np.testing.assert_allclose(d['expectation_gradient'],d['angle_gradient']*expected,rtol=1e-10,atol=1e-12)
    assert expected[0,2]<expected[0,0]*.001
    checks=h.numerical_checks(z)
    assert checks['fd_probability_max_abs_error']<1e-8
    assert checks['final_rz_perturbation_max_abs_error']<1e-12
    # An independent full Jacobian confirms absence of cross-patient derivatives.
    t=torch.tensor(z[:3],dtype=torch.float64,requires_grad=True)
    jac=torch.autograd.functional.jacobian(h.tensor_probability,t).detach().numpy()
    for i in range(3):
        for j in range(3):
            np.testing.assert_allclose(jac[i,j],d['probability_gradient'][i] if i==j else 0,atol=1e-12)


def test_intervention_scales_and_donors_use_only_training():
    training,y=samples();reference=training_reference(training,y)
    validation=np.full((5,4),1000.)
    plan=list(interventions(validation,training,reference,0,1729,'validation'))
    assert len(plan)==68
    for info,altered in plan:
        j=int(info['direction'].rsplit('_',1)[1])
        if info['family']=='local_coordinate':
            delta=altered-validation
            expected=np.zeros(4);expected[j]=info['sign']*info['alpha']*reference['sd'][j]
            np.testing.assert_allclose(delta,np.broadcast_to(expected,delta.shape),atol=1e-12)
        if info['family']=='scramble':
            assert set(altered[:,j])<=set(training[:,j])
        if info['family']=='mean':
            np.testing.assert_allclose(altered[:,j],reference['mean'][j])
    again=list(interventions(validation,training,reference,0,1729,'validation'))
    for (a,x),(b,w) in zip(plan,again):
        assert a==b
        np.testing.assert_array_equal(x,w)
    assert reference['sd'][3]<1e-14
    np.testing.assert_allclose(reference['axes'].T@reference['axes'],np.eye(4),atol=1e-12)


def test_constant_offset_is_distinct_from_variation():
    z,y=samples();reference=training_reference(z,y)
    variants={info['intervention']:altered for info,altered in interventions(z,z,reference,0,1729,'train')}
    np.testing.assert_allclose(variants['mean_3'],z,atol=1e-14)
    assert np.max(np.abs(phi(variants['zero_3'])-phi(z)))>.1
    d=displacement(z,variants['zero_3'],z,reference,'train')
    assert d['z_l2_mean']==pytest.approx(.7)
    assert d['outside_training_bounds_after']==1.
    assert d['angle_l2_mean']>0


def test_identity_and_classical_head_derivatives():
    z,y=samples();h=head('B_classical')
    p=h.predict(z);d=h.derivatives(z)
    expected=(p*(1-p))[:,None]*h.weight.numpy()
    np.testing.assert_allclose(d['probability_gradient'],expected,atol=1e-12)
    assert all(v==0 for v in predictive_change(y,p,p).values())
    before=h.weight.clone();h.derivatives(z);h.predict(z)
    assert torch.equal(h.weight,before)


def test_new_output_directories_are_required(tmp_path):
    with pytest.raises(FileExistsError):
        run(tmp_path,tmp_path/'new','missing')


def test_training_pc_variances_do_not_inflate_nearly_null_directions():
    rng=np.random.default_rng(4)
    z=rng.normal(size=(100,4));z[:,3]=z[:,0]+z[:,1]+1e-10*z[:,3]
    reference=training_reference(z,np.array([0,1]*50))
    assert reference['eigenvalues'][-1]<1e-20
    direct=((z-z.mean(axis=0))@reference['axes']).var(axis=0)
    np.testing.assert_allclose(reference['eigenvalues'],direct,rtol=1e-5,atol=1e-25)


def test_linear_head_translation_preserves_ranking():
    z,y=samples();h=head('B_classical')
    translated=z+np.array([.1,-.1,.05,.03])
    effect=predictive_change(y,h.predict(z),h.predict(translated))
    assert effect['delta_roc_auc']==pytest.approx(0,abs=1e-12)
    assert effect['probability_change_abs_mean']>0


def test_summary_reproduction_and_shared_input_pairing():
    import pandas as pd
    from src.phase2.analysis_views import read_csv
    from src.phase2.run_d0 import aggregate_interventions
    root=Path('runs/phase2/d0_v1_verified')
    if not (root/'interventions.csv').exists():
        pytest.skip('Private completed D0 archive unavailable')
    detail=read_csv(root/'interventions.csv')
    assert len(detail)==8160
    actual=aggregate_interventions(detail)
    saved=read_csv('reports/phase2/d0_v1_verified/intervention_summary.csv')
    pd.testing.assert_frame_equal(actual,saved,check_exact=False,atol=1e-12,rtol=1e-12)
    keys=['fold','seed','split','intervention']
    b=detail.query("model=='B'").set_index(keys)
    probe=detail.query("model=='B_frozen_probe'").set_index(keys)
    for column in ['z_l2_mean','angle_l2_mean','outside_training_bounds_after','nearest_training_distance_after']:
        np.testing.assert_array_equal(b[column],probe[column])


def test_notebook_reads_only_aggregate_summaries(monkeypatch):
    import builtins
    import json
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import src.phase2.data
    import src.phase2.training
    from sklearn.linear_model import LogisticRegression
    root=Path.cwd().resolve()
    notebook=json.loads((root/'notebooks/03_d0_compatibility.ipynb').read_text())
    if not (root/'reports/phase2/d0_v1_verified/coordinate_ablation.csv').exists():
        pytest.skip('Completed aggregate D0 summaries unavailable')
    def forbidden(*args,**kwargs):
        raise AssertionError('Read-only D0 notebook must not fit, load checkpoints, or access cohort')
    monkeypatch.setattr(src.phase2.data,'load_development',forbidden)
    monkeypatch.setattr(src.phase2.training,'train',forbidden)
    monkeypatch.setattr(LogisticRegression,'fit',forbidden)
    monkeypatch.setattr(torch,'load',forbidden)
    original_open=builtins.open;original_path_open=Path.open
    def check(path):
        if isinstance(path,(str,Path)):
            path=Path(path).resolve()
            assert not path.is_relative_to(root/'runs')
            assert not path.is_relative_to(root/'data')
            assert path.suffix not in {'.npz','.pt','.pkl'}
    def guarded_open(path,*args,**kwargs):
        check(path);return original_open(path,*args,**kwargs)
    def guarded_path_open(path,*args,**kwargs):
        check(path);return original_path_open(path,*args,**kwargs)
    monkeypatch.setattr(builtins,'open',guarded_open)
    monkeypatch.setattr(Path,'open',guarded_path_open)
    rendered=[]
    def show():
        plt.gcf().canvas.draw();rendered.append(plt.gcf());plt.close('all')
    monkeypatch.setattr(plt,'show',show)
    namespace={'__name__':'__main__','display':lambda value:None}
    for i,cell in enumerate(notebook['cells']):
        if cell['cell_type']=='code':
            exec(compile(''.join(cell['source']),f'd0-notebook-cell-{i}','exec'),namespace)
    assert len(rendered)==7
    assert len(namespace['tables']['baseline_metrics'])==210


def test_locked_source_is_rejected_before_checkpoint_reads(tmp_path,monkeypatch):
    import json
    from src.data.utils import sha256_file
    protocol=Path('data/processed/splits/phase1c_inner_cv.json')
    if not protocol.exists():
        pytest.skip('Local canonical split metadata unavailable')
    archive=tmp_path/'archive';archive.mkdir()
    config=archive/'config.json';config.write_text('{}')
    provenance=archive/'provenance.json'
    provenance.write_text(json.dumps({'locked_test_evaluated':True}))
    cv=tmp_path/'cv';cv.mkdir()
    (cv/'provenance.json').write_text(json.dumps(dict(
        sources={str(p):sha256_file(p) for p in [config,provenance]},
        runs=[str(archive)],protocol_sha256=sha256_file(protocol))))
    def forbidden(*args,**kwargs):
        raise AssertionError('Checkpoint read before partition gate')
    monkeypatch.setattr(torch,'load',forbidden)
    with pytest.raises(ValueError,match='Locked-test evaluation is prohibited'):
        run(tmp_path/'new_run',tmp_path/'summaries',cv,protocol)
    assert not (tmp_path/'new_run').exists()
