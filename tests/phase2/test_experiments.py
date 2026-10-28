import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
torch = pytest.importorskip("torch")
pytest.importorskip("pennylane")
import yaml
from src.phase2.data import Representation, validate_partitions
from src.phase2.models import Model, angles
from src.phase2.training import train, seed_all
from src.phase2.run import run


def protocol():
    return dict(development_case_ids=['a','b','c','d'], locked_test_case_ids=['locked'], folds=[
        dict(fold=0, train_case_ids=['a','b'], validation_case_ids=['c','d']),
        dict(fold=1, train_case_ids=['c','d'], validation_case_ids=['a','b'])])


@pytest.mark.parametrize('kind', ['overlap', 'locked', 'missing', 'duplicate'])
def test_partition_rejection(kind):
    p = protocol()
    if kind == 'overlap': p['folds'][0]['train_case_ids'].append('c')
    if kind == 'locked': p['folds'][0]['validation_case_ids'][0] = 'locked'
    if kind == 'missing': p['folds'][0]['validation_case_ids'].pop()
    if kind == 'duplicate': p['development_case_ids'].append('a')
    with pytest.raises(ValueError): validate_partitions(p)


def test_existing_protocol():
    path = Path('data/processed/splits/phase1c_inner_cv.json')
    if not path.exists():
        pytest.skip('Canonical data are local, unversioned artifacts')
    p = json.loads(path.read_text())
    validate_partitions(p)
    assert len(p['development_case_ids']) == 835
    assert len(p['locked_test_case_ids']) == 209


def counts():
    rng = np.random.default_rng(1729)
    return pd.DataFrame(rng.poisson(40, (24, 50)), index=[f'p{i}' for i in range(24)],
                        columns=[f'g{i}' for i in range(50)])


def test_training_only_transforms():
    c = yaml.safe_load(Path('configs/data.yaml').read_text())['preprocessing']
    x = counts()
    r = Representation(c, 4).fit(x.iloc[:18])
    mean = r.pca.mean_.copy()
    reference = r.rna.tmm_.reference_sample_id_
    assert reference in x.index[:18]
    assert r.pca.n_samples_ == 18
    assert r.training_ids == list(x.index[:18])
    # Any fit on validation must fail, including nested components.
    def forbidden(*args, **kwargs): raise AssertionError('fit during validation')
    for item in [r.rna, r.pca, r.scaler, r.rna.expression_filter_, r.rna.tmm_,
                 r.rna.zero_variance_filter_, r.rna.scaler_]:
        item.fit = forbidden
    result = r.transform(x.iloc[18:] * 100)
    assert result.shape == (6, 4)
    np.testing.assert_array_equal(r.pca.mean_, mean)
    assert np.isfinite(result).all()


@pytest.mark.parametrize('encoder', [False, True])
def test_quantum_shapes_and_gradients(encoder):
    seed_all(1729)
    model = Model(6 if encoder else 4, encoder=encoder)
    x = torch.randn(5, 6 if encoder else 4, dtype=torch.float64, requires_grad=True)
    prediction = model(x)
    assert prediction.shape == (5,)
    assert torch.all((prediction > 0) & (prediction < 1))
    torch.nn.functional.binary_cross_entropy(prediction, torch.tensor([0.,1.,0.,1.,1.], dtype=torch.float64)).backward()
    assert torch.isfinite(x.grad).all() and x.grad.norm() > 0
    assert model.head.weights.grad.norm() > 0
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    if encoder:
        assert sum(p.grad.square().sum() for p in model.encoder.parameters()) > 0


def test_angles():
    a = angles(torch.tensor([-1e100, -2., 0., 2., 1e100], dtype=torch.float64))
    assert torch.isfinite(a).all() and (a.abs() <= np.pi).all()


def test_reproducible_training():
    results = []
    for _ in range(2):
        seed_all(1729)
        model = Model(6, encoder=True)
        x = torch.randn(8, 6, dtype=torch.float64)
        y = torch.tensor([0., 1.] * 4, dtype=torch.float64)
        curves, _ = train(model, x, y, x, y,
                          dict(epochs=2, batch_size=4, lr=.01, weight_decay=.001))
        assert np.isfinite(pd.DataFrame(curves).to_numpy()).all()
        results.append(model(x).detach().numpy())
    np.testing.assert_array_equal(*results)


def test_persistence(tmp_path, monkeypatch):
    import src.phase2.run as runner
    root = Path.cwd()
    c = yaml.safe_load((root/'configs/data.yaml').read_text())
    x = counts()
    tr, va = list(x.index[:18]), list(x.index[18:])
    p = dict(folds=[dict(fold=0, train_case_ids=tr, validation_case_ids=va)])
    y = pd.Series([0,1]*12, index=x.index)
    monkeypatch.setattr(runner, 'load_development', lambda _: (c, root, p, x, y))
    config = dict(data_config='configs/data.yaml', experiments=['A','B'], folds=[0],
                  seeds=[1729], intermediate_components=6, epochs=1, batch_size=18,
                  lr=.01, weight_decay=.001)
    table = run(config, tmp_path/'run')
    assert len(table) == 5
    assert set(table.experiment) == {'A','B','A_classical','B_classical','B_frozen_probe'}
    assert (tmp_path/'run/config.json').exists()
    assert len(list((tmp_path/'run').rglob('validation_predictions.csv'))) == 5
    assert len(list((tmp_path/'run').rglob('model.pt'))) == 3
    with pytest.raises(FileExistsError): run(config, tmp_path/'run')
