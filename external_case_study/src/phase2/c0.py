"""Closed C0-v1 objectives and deterministic fitting with the unchanged B model."""
import copy
import hashlib
import math
import time

import numpy as np
import torch

from .models import Model, angles


ARMS = ('C0-NQE-Q', 'C0-NQE-E', 'C0-Hybrid-Q', 'C0-Hybrid-E')
CLOSED = dict(protocol_id='C0-v1', arms=list(ARMS), folds=list(range(5)),
              seeds=[1729, 1730, 1731], epochs=50, batch_size=32,
              pair_count=64, pair_strata=[16, 16, 32], lambda_aux=1.,
              lr=.01, weight_decay=.001, gaussian_denominator=4.,
              selection='fixed_final_epoch', two_stage_encoder='frozen',
              primary_contrasts=['C0-NQE-Q_minus_C0-NQE-E',
                                 'C0-Hybrid-Q_minus_C0-Hybrid-E'])


def validate_config(config):
    if config != CLOSED:
        raise ValueError('C0-v1 is fixed; changes require a separately approved protocol')


def tensor_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def initialized_model(seed):
    """Return the original B initialization and its independent patient RNG stream."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = Model(32, encoder=True)
        stream = torch.random.get_rng_state().clone()
    return model, stream


def pair_indices(labels, generator):
    """Sample ordered non-diagonal pairs; orientation is irrelevant to the loss."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.isin(labels, [0, 1]).all():
        raise ValueError('Binary labels are required')
    classes = {c: np.flatnonzero(labels == c) for c in [0, 1]}
    if min(map(len, classes.values())) < 2:
        raise ValueError('Each training class requires two distinct patients')
    rows = []
    for c in [1, 0]:
        group = classes[c]
        first = generator.integers(len(group), size=16)
        second = generator.integers(len(group) - 1, size=16)
        second += second >= first
        rows.extend(zip(group[first], group[second]))
    rows.extend(zip(generator.choice(classes[1], 32), generator.choice(classes[0], 32)))
    return np.asarray(rows, dtype=np.int64)


def similarity(a, b, geometry):
    if a.shape != b.shape or a.shape[-1] != 4:
        raise ValueError('Paired angles must have the same final dimension of four')
    if geometry == 'Q':
        return torch.cos((a - b) / 2).square().prod(dim=-1)
    if geometry == 'E':
        return torch.exp(-(a - b).square().sum(dim=-1) / 4)
    raise ValueError('Unknown geometry')


def pair_loss(encoder, features, labels, pairs, geometry):
    """Mean supervised pair residual, with gradients through both endpoints."""
    index = torch.as_tensor(pairs, dtype=torch.long)
    if index.shape != (64, 2) or index.min() < 0 or index.max() >= len(features):
        raise ValueError('Exactly 64 valid training pairs are required')
    if torch.any(index[:, 0] == index[:, 1]):
        raise ValueError('Diagonal pairs are forbidden')
    a = angles(encoder(features[index.reshape(-1)])).reshape(64, 2, 4)
    target = (labels[index[:, 0]] == labels[index[:, 1]]).to(features.dtype)
    return (similarity(a[:, 0], a[:, 1], geometry) - target).square().mean()


def _optimizer(model, encoder, head):
    for p in model.encoder.parameters():
        p.requires_grad_(encoder)
        p.grad = None
    for p in model.head.parameters():
        p.requires_grad_(head)
        p.grad = None
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=.01, weight_decay=.001,
                                 betas=(.9, .999), eps=1e-8, amsgrad=False,
                                 maximize=False)
    assert {id(p) for group in optimizer.param_groups for p in group['params']} == {id(p) for p in parameters}
    return optimizer


def component_gradients(model, x, y, fold, seed, geometry):
    """Measure loss components on fixed training-only inputs without updating state."""
    generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, fold, 2])))
    selected = generator.choice(len(x), min(32, len(x)), replace=False)
    pairs = pair_indices(y.numpy(), generator)
    bce = torch.nn.functional.binary_cross_entropy(model(x[selected]), y[selected])
    auxiliary = pair_loss(model.encoder, x, y, pairs, geometry)
    params = tuple(model.encoder.parameters())
    g_b = torch.cat([g.reshape(-1) for g in torch.autograd.grad(bce, params)])
    g_r = torch.cat([g.reshape(-1) for g in torch.autograd.grad(auxiliary, params)])
    nb, nr = float(g_b.norm()), float(g_r.norm())
    cosine = float(torch.dot(g_b, g_r) / (g_b.norm() * g_r.norm())) if min(nb, nr) > 1e-12 else None
    return dict(bce=float(bce.detach()), auxiliary=float(auxiliary.detach()),
                bce_encoder_gradient_norm=nb, auxiliary_encoder_gradient_norm=nr,
                ratio=nr / (nb + 1e-12), cosine=cosine)


def fit_pipeline(x, y, validation, target, fold, seed, arm, initial_state,
                 config, *, synthetic=False, technical_epochs=None):
    """Fit one approved arm, or an explicitly shortened synthetic technical fixture."""
    validate_config(config)
    if arm not in ARMS or seed not in CLOSED['seeds'] or fold not in CLOSED['folds']:
        raise ValueError('Unknown C0 arm, fold or seed')
    if technical_epochs is not None and (not synthetic or technical_epochs not in [1, 2]):
        raise ValueError('Short budgets are restricted to synthetic technical fixtures')
    if not synthetic and (np.shape(x) != (668, 32) or np.shape(validation) != (167, 32)):
        raise ValueError('Canonical development dimensions are required')
    epochs = technical_epochs if technical_epochs is not None else 50
    x, y, xv, yv = [torch.tensor(np.array(a, copy=True), dtype=torch.float64)
                    for a in [x, y, validation, target]]
    if x.ndim != 2 or xv.ndim != 2 or x.shape[1] != 32 or xv.shape[1] != 32:
        raise ValueError('PCA32 features are required')
    if y.shape != (len(x),) or yv.shape != (len(xv),):
        raise ValueError('Label and feature lengths disagree')
    if not all(torch.isfinite(a).all() for a in [x, y, xv, yv]):
        raise ValueError('Nonfinite inputs')
    if not all(torch.all((a == 0) | (a == 1)) for a in [y, yv]):
        raise ValueError('Binary labels required')
    model, patient_state = initialized_model(seed)
    if set(initial_state) != set(model.state_dict()) or any(
            not torch.equal(initial_state[k], v) for k, v in model.state_dict().items()):
        raise ValueError('Initial checkpoint differs from historical B generation')
    model.load_state_dict(initial_state)
    initial = copy.deepcopy(model.state_dict())
    geometry = arm[-1]
    two_stage = 'NQE' in arm
    pairs_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, fold, 1])))
    orders = torch.Generator().set_state(patient_state)
    pair_digest, order_digest = hashlib.sha256(), hashlib.sha256()
    curves, diagnostics, stages = [], [], []
    resource = dict(optimizer_steps=0, pair_draws=0, pair_endpoint_occurrences=0,
                    bce_patient_occurrences=0, training_qnode_calls=0,
                    training_sample_states=0, validation_qnode_calls=0,
                    validation_sample_states=0, diagnostic_qnode_calls=0,
                    diagnostic_sample_states=0)
    state_hash_before = tensor_hash(initial_state)
    cached_train = cached_validation = None
    stage1 = None
    def counted(kind, function):
        calls, samples = model.head.calls, model.head.samples
        value = function()
        resource[kind + '_qnode_calls'] += model.head.calls - calls
        resource[kind + '_sample_states'] += model.head.samples - samples
        return value
    if not two_stage:
        diagnostics.append(dict(epoch=0, **counted('diagnostic', lambda: component_gradients(
            model, x, y, fold, seed, geometry))))
    for stage in (['geometry', 'head'] if two_stage else ['joint']):
        optimizer = _optimizer(model, stage != 'head', stage != 'geometry')
        frozen = model.head if stage == 'geometry' else model.encoder if stage == 'head' else None
        frozen_hash = tensor_hash(frozen.state_dict()) if frozen is not None else None
        stage_started = time.perf_counter()
        if stage == 'head':
            with torch.no_grad():
                cached_train, cached_validation = model.encoder(x).detach(), model.encoder(xv).detach()
        stage_steps = 0
        for epoch in range(epochs):
            update_seconds = validation_seconds = 0.
            bce_sum = pair_sum = 0.
            norms = {'encoder': [], 'head': []}
            head_squares = []
            if stage == 'geometry':
                batches = [None] * math.ceil(len(x) / 32)
            else:
                order = torch.randperm(len(x), generator=orders)
                order_digest.update(order.numpy().tobytes())
                batches = list(order.split(32))
            for batch in batches:
                tick = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                if stage != 'head':
                    pairs = pair_indices(y.numpy(), pairs_rng)
                    pair_digest.update(pairs.tobytes())
                    aux = pair_loss(model.encoder, x, y, pairs, geometry)
                    resource['pair_draws'] += 64
                    resource['pair_endpoint_occurrences'] += 128
                    pair_sum += float(aux.detach())
                if stage != 'geometry':
                    def bce_call():
                        probability = model.head(cached_train[batch]) if stage == 'head' else model(x[batch])
                        return torch.nn.functional.binary_cross_entropy(probability.reshape(-1), y[batch])
                    bce = counted('training', bce_call)
                    bce_sum += float(bce.detach()) * len(batch)
                    resource['bce_patient_occurrences'] += len(batch)
                loss = aux if stage == 'geometry' else bce if stage == 'head' else bce + aux
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite C0 training loss')
                loss.backward()
                for name in norms:
                    gradients = [p.grad for p in getattr(model, name).parameters() if p.requires_grad]
                    if any(g is None or not torch.isfinite(g).all() for g in gradients):
                        raise FloatingPointError('Missing or nonfinite active gradient')
                    norms[name].append(sum(float(g.square().sum()) for g in gradients) ** .5)
                if stage != 'geometry':
                    head_squares.append(model.head.weights.grad.detach().square().clone())
                optimizer.step()
                if any(not torch.isfinite(p).all() for p in model.parameters()):
                    raise FloatingPointError('Nonfinite C0 parameters')
                stage_steps += 1
                resource['optimizer_steps'] += 1
                update_seconds += time.perf_counter() - tick
            val = None
            if stage != 'geometry':
                tick = time.perf_counter()
                with torch.no_grad():
                    val = counted('validation', lambda: torch.nn.functional.binary_cross_entropy(
                        (model.head(cached_validation) if stage == 'head' else model(xv)).reshape(-1), yv))
                val = float(val)
                if not np.isfinite(val):
                    raise FloatingPointError('Nonfinite observational validation loss')
                validation_seconds = time.perf_counter() - tick
            gradients = {}
            if head_squares:
                rms = torch.stack(head_squares).mean(dim=0).sqrt()
                gradients = {f'gradient_l{l}_q{q}_{gate}': float(rms[l, q, g])
                             for l in range(2) for q in range(4)
                             for g, gate in enumerate(['RY', 'RZ'])}
            curves.append(dict(stage=stage, epoch=epoch+1,
                online_bce=bce_sum / len(x) if stage != 'geometry' else None,
                online_auxiliary=pair_sum / len(batches) if stage != 'head' else None,
                validation_bce=val, update_seconds=update_seconds,
                validation_seconds=validation_seconds, stage_steps=stage_steps,
                **{name+'_gradient_norm': float(np.mean(v)) for name, v in norms.items()}, **gradients))
            if stage == 'joint' and epoch+1 in [25, 50]:
                diagnostics.append(dict(epoch=epoch+1, **counted('diagnostic', lambda: component_gradients(
                    model, x, y, fold, seed, geometry))))
        if frozen is not None and tensor_hash(frozen.state_dict()) != frozen_hash:
            raise AssertionError('A frozen parameter block changed')
        stages.append(dict(stage=stage, optimizer_steps=stage_steps,
                           seconds=time.perf_counter()-stage_started,
                           frozen_block_sha256=frozen_hash))
        if stage == 'geometry':
            stage1 = copy.deepcopy(model.state_dict())
    model.eval()
    with torch.no_grad():
        z = {'train': model.encoder(x).numpy(), 'validation': model.encoder(xv).numpy()}
        probabilities = {'train': model(x).numpy(), 'validation': model(xv).numpy()}
    resource.update(total_qnode_calls=model.head.calls, total_sample_states=model.head.samples,
                    final_prediction_qnode_calls=2, final_prediction_sample_states=len(x)+len(xv),
                    pair_stream_sha256=pair_digest.hexdigest(), patient_stream_sha256=order_digest.hexdigest())
    return dict(model=model, initial_state=initial, geometry_state=stage1,
                probabilities=probabilities, z=z, curves=curves, diagnostics=diagnostics,
                stages=stages, resources=resource, initial_sha256=state_hash_before,
                status='synthetic_technical' if synthetic else 'development_cv',
                scientific_epochs=50, executed_epochs_per_stage=epochs)


def synthetic_fixture(seed=1729):
    """Generate a small technical fixture with no patient-derived inputs."""
    generator = np.random.default_rng(seed)
    x, validation = generator.normal(size=(64, 32)), generator.normal(size=(16, 32))
    return x, (x[:, 0] > 0).astype(int), validation, (validation[:, 0] > 0).astype(int)
