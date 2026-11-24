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


@pytest.mark.parametrize('seed', [1729, 1730, 1731])
def test_paired_quantum_initialization(seed):
    seed_all(seed)
    a = Model(4)
    state_a = torch.random.get_rng_state()
    seed_all(seed)
    b = Model(32, encoder=True)
    assert torch.equal(a.head.weights, b.head.weights)
    # Thus A/B also start minibatch shuffling at the same RNG state.
    assert torch.equal(state_a, torch.random.get_rng_state())
    seed_all(seed)
    classical = Model(32, encoder=True, quantum=False)
    for x, y in zip(b.encoder.parameters(), classical.encoder.parameters()):
        assert torch.equal(x, y)


def test_relevant_and_redundant_quantum_parameters():
    from src.phase2.diagnostics import circuit_sensitivity
    result = circuit_sensitivity()
    rms = np.array(result['jacobian_rms'])
    assert (rms[0] > 1e-5).all()
    assert (rms[1, :, 0] > 1e-5).all()
    assert (rms[1, :, 1] < 1e-12).all()
    assert result['max_finite_difference_error'] < 1e-8


def test_circuit_can_learn_synthetic_task():
    from src.phase2.diagnostics import synthetic_learning
    result = synthetic_learning()
    assert result['final_loss'] < .15
    assert result['final_loss'] < result['initial_loss'] * .4
    assert result['accuracy'] == 1.


def test_angular_saturation_definition():
    from src.phase2.diagnostics import angular_summary
    result = angular_summary(torch.tensor([[0., 1., 10., -10.]], dtype=torch.float64))
    assert result['angular_saturation_95'] == .5
    assert result['angular_saturation_99'] == .5
    assert result['tanh_derivative_below_001'] == .5


def test_final_rz_invariance_for_finite_perturbations():
    from src.phase2.models import VQC, angles
    seed_all(1729)
    model = VQC()
    x = angles(torch.randn(13, 4, dtype=torch.float64))
    w = torch.randn_like(model.weights)
    baseline = model.circuit(x, w)
    perturbed = w.clone()
    perturbed[1, :, 1] += torch.tensor([.8, -1.2, 2.7, -.4], dtype=torch.float64)
    torch.testing.assert_close(model.circuit(x, perturbed), baseline, atol=1e-12, rtol=0)


def test_score_pairing_uses_patient_ids():
    from src.phase2.analyze_pilot import paired_predictions
    frame = pd.DataFrame(dict(case_id=['a','b','c'],er_label=[0,1,1],
                              probability_er_positive=[.1,.9,.6]))
    joined,rho = paired_predictions(frame,frame.iloc[::-1])
    assert len(joined) == 3 and rho == 1.
    with pytest.raises(ValueError): paired_predictions(frame,frame.iloc[:2])
    changed = frame.copy()
    changed.loc[0,'er_label'] = 1
    with pytest.raises(ValueError): paired_predictions(frame,changed)


def test_probability_csv_round_trip_preserves_log_loss(tmp_path):
    from src.phase2.analyze_pilot import read_csv
    from src.phase2.evaluation import metrics
    p = np.array([.01,.6,.9999999999999991])
    y = np.array([0,1,0])
    path = tmp_path/'probabilities.csv'
    pd.DataFrame(dict(probability=p)).to_csv(path,index=False)
    restored = read_csv(path).probability.to_numpy()
    np.testing.assert_array_equal(restored,p)
    assert metrics(y,p)['log_loss'] == metrics(y,restored)['log_loss']
