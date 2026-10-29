"""Closed E0 design, paired RNG streams, head-only fitting, and isolation tests."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from src.phase2.e0 import CLOSED, validate_config, new_head, fit_head
from src.phase2.e0_data import initial_weights, load_inputs
from src.phase2.d0 import phi


def toy():
    rng=np.random.default_rng(1729)
    return rng.normal(size=(12,4)),np.array([0.,1.]*6)


@pytest.mark.parametrize('key,value',[('epochs',100),('lr',.02),('head_seeds',[1729]),('phase_closed',False)])
def test_closed_protocol_rejects_extensions(key,value):
    config=copy.deepcopy(CLOSED);config[key]=value
    with pytest.raises(ValueError,match='independent approved protocol'): validate_config(config)
    validate_config(copy.deepcopy(CLOSED))


def test_initialization_and_angular_map_are_exact():
    z,_=toy();before=torch.random.get_rng_state().clone()
    head=new_head(1730)
    assert torch.equal(torch.random.get_rng_state(),before)
    assert torch.equal(head.weights,initial_weights(1730))
    x=torch.tensor(z,dtype=torch.float64)
    with torch.no_grad():
        actual=head(x).numpy()
        expected=((1+head.circuit(torch.tensor(phi(z)),head.weights))/2).clamp(1e-7,1-1e-7).numpy()
    np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=0)


def test_validation_labels_never_change_fitted_head_and_inputs_remain_frozen():
    z,y=toy();original=z.copy();config=copy.deepcopy(CLOSED);config.update(epochs=2,batch_size=4)
    a,_,pa,ca,ta=fit_head(z,y,z,y,1729,1730,config)
    b,_,pb,cb,tb=fit_head(z,y,z,1-y,1729,1730,config)
    assert torch.equal(a.weights,b.weights)
    np.testing.assert_array_equal(pa['train'],pb['train'])
    np.testing.assert_array_equal(z,original)
    assert ta['shuffle_sha256']==tb['shuffle_sha256']
    assert ta['optimizer_steps']==6 and len(ca)==2
    assert ca[-1]['validation_loss']!=cb[-1]['validation_loss']
    assert set(dict(a.named_parameters()))=={'weights'}


def test_shuffle_is_paired_across_head_initializations():
    z,y=toy();config=copy.deepcopy(CLOSED);config.update(epochs=2,batch_size=4)
    runs=[fit_head(z,y,z,y,1729,h,config) for h in [1729,1730,1731]]
    assert len({r[-1]['shuffle_sha256'] for r in runs})==1
    assert not torch.equal(runs[0][0].weights,runs[1][0].weights)


def test_locked_source_gate_precedes_checkpoint_loading(tmp_path,monkeypatch):
    from src.data.utils import sha256_file
    canonical=Path('data/processed/splits/phase1c_inner_cv.json')
    if not canonical.exists(): pytest.skip('Canonical metadata is unavailable')
    archive=tmp_path/'archive';archive.mkdir()
    config=archive/'config.json';config.write_text('{}')
    provenance=archive/'provenance.json';provenance.write_text(json.dumps(dict(locked_test_evaluated=True)))
    cv=tmp_path/'cv';cv.mkdir()
    (cv/'provenance.json').write_text(json.dumps(dict(runs=[str(archive)],
        sources={str(p):sha256_file(p) for p in [config,provenance]},protocol_sha256=sha256_file(canonical))))
    def forbidden(*args,**kwargs): raise AssertionError('Checkpoint loaded before isolation gate')
    monkeypatch.setattr(torch,'load',forbidden)
    with pytest.raises(ValueError,match='Locked-test source prohibited'): load_inputs(cv,canonical)


def test_primary_contrasts_and_diagonal_are_paired_not_selected():
    import pandas as pd
    from src.phase2.analyze_e0 import METRICS,results_wide,paired_tables,summarize_primary,summarize_results
    rows=[]
    for f in range(5):
        for s in [1729,1730,1731]:
            for model,offset,hs in [('B_historical',0,[-1]),('LR_z_historical',1,[-1]),
                                    ('LR_angle',2,[-1]),('VQC_refit',3,[1729,1730,1731])]:
                for h in hs:
                    for split in ['train','validation']:
                        value=f+.1*(s-1729)+offset+(.01*(h-1729) if h>=0 else 0)+(.5 if split=='validation' else 0)
                        rows.append(dict(fold=f,encoder_seed=s,head_seed=h,model=model,split=split,
                            diagonal=model=='VQC_refit' and s==h,**{m:value for m in METRICS}))
    table=pd.DataFrame(rows).sample(frac=1,random_state=4)
    wide=results_wide(table);e01,e02=paired_tables(wide)
    assert len(e01)==45 and len(e01[e01.diagonal])==15 and len(e02)==15
    np.testing.assert_allclose(e01.train_log_loss_minus_B,3+.01*(e01.head_seed-1729))
    np.testing.assert_allclose(e02.validation_roc_auc_minus_LR_z,1)
    primary,_=summarize_primary(e01,e02)
    value=primary.query("scope=='all_initializations' and endpoint=='train_log_loss_minus_B'")['mean'].item()
    assert value==pytest.approx(3.01)
    summary=summarize_results(table)
    assert summary.query("model=='VQC_refit' and split=='train' and metric=='log_loss'")['mean'].item()==pytest.approx(5.11)


def test_saved_e0_reanalysis_never_fits(tmp_path,monkeypatch):
    import pandas as pd
    from src.phase2.analyze_e0 import analyze
    from src.phase2.analysis_views import read_csv
    import src.phase2.e0
    import src.phase2.data
    from sklearn.linear_model import LogisticRegression
    run=Path('runs/phase2/e0_v1')
    if not (run/'provenance.json').exists(): pytest.skip('Completed private E0 archive absent')
    def forbidden(*args,**kwargs): raise AssertionError('Analysis attempted fitting or cohort loading')
    monkeypatch.setattr(src.phase2.e0,'fit_head',forbidden)
    monkeypatch.setattr(src.phase2.data,'load_development',forbidden)
    monkeypatch.setattr(LogisticRegression,'fit',forbidden)
    result=analyze(run,tmp_path/'analysis')
    assert len(result['metrics'])==180 and len(result['e01_diagonal'])==15
    assert len(result['curves'])==2250
    pd.testing.assert_frame_equal(result['primary_summary'],read_csv('reports/phase2/e0_v1/primary_summary.csv'),
                                 check_exact=False,atol=1e-12,rtol=1e-12)
    with pytest.raises(FileExistsError): analyze(run,tmp_path/'analysis')


def test_notebook_is_read_only_and_uses_aggregate_summaries(monkeypatch):
    import builtins
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import src.phase2.e0
    import src.phase2.data
    from sklearn.linear_model import LogisticRegression
    root=Path.cwd().resolve()
    notebook=json.loads((root/'notebooks/04_e0_head_compatibility.ipynb').read_text())
    if not (root/'reports/phase2/e0_v1/metrics.csv').exists(): pytest.skip('Completed E0 summaries unavailable')
    def forbidden(*args,**kwargs): raise AssertionError('Read-only notebook attempted fitting, checkpoint loading, or cohort access')
    monkeypatch.setattr(src.phase2.e0,'fit_head',forbidden)
    monkeypatch.setattr(src.phase2.data,'load_development',forbidden)
    monkeypatch.setattr(LogisticRegression,'fit',forbidden)
    monkeypatch.setattr(torch,'load',forbidden)
    original_open=builtins.open;original_path_open=Path.open
    def check(path):
        if isinstance(path,(str,Path)):
            path=Path(path).resolve()
            assert not path.is_relative_to(root/'runs') and not path.is_relative_to(root/'data')
            assert path.suffix not in {'.pt','.pkl','.npz'}
    def guarded_open(path,*args,**kwargs): check(path);return original_open(path,*args,**kwargs)
    def guarded_path_open(path,*args,**kwargs): check(path);return original_path_open(path,*args,**kwargs)
    monkeypatch.setattr(builtins,'open',guarded_open);monkeypatch.setattr(Path,'open',guarded_path_open)
    rendered=[]
    def show():
        plt.gcf().canvas.draw();rendered.append(plt.gcf());plt.close('all')
    monkeypatch.setattr(plt,'show',show)
    namespace={'__name__':'__main__','display':lambda value:None}
    for i,cell in enumerate(notebook['cells']):
        if cell['cell_type']=='code': exec(compile(''.join(cell['source']),f'e0-notebook-cell-{i}','exec'),namespace)
    assert len(rendered)==8
    assert len(namespace['tables']['e01_diagonal'])==15
