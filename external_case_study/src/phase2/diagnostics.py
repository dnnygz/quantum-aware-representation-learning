"""Read-only circuit, angular and learning checks; never select hyperparameters."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .models import VQC, angles
from .training import seed_all


def angular_summary(z):
    """Rates pool patient × coordinate values; derivative refers to tanh(z/2)."""
    z = z.detach()
    t = torch.tanh(z / 2)
    derivative = .5 * (1 - t.square())
    return dict(angle_abs_mean=float((np.pi*t).abs().mean()),
                angle_abs_max=float((np.pi*t).abs().max()),
                angular_saturation_95=float((t.abs() > .95).double().mean()),
                angular_saturation_99=float((t.abs() > .99).double().mean()),
                tanh_derivative_mean=float(derivative.mean()),
                tanh_derivative_below_001=float((derivative < .01).double().mean()))


def circuit_sensitivity(seed=1729):
    seed_all(seed)
    model = VQC()
    z = torch.randn(19, 4, dtype=torch.float64)
    # Probe generic points away from the small initialization so a small local
    # derivative is not confused with an exactly commuting gate.
    weights = torch.randn_like(model.weights, requires_grad=True)
    x = angles(z)
    jacobian = torch.autograd.functional.jacobian(lambda w: model.circuit(x, w), weights)
    step = 1e-5
    finite_difference = torch.empty_like(jacobian)
    for i in range(16):
        delta = torch.zeros_like(weights)
        delta.reshape(-1)[i] = step
        finite_difference.reshape(19, 16)[:, i] = (
            (model.circuit(x, weights + delta) - model.circuit(x, weights - delta)) / (2*step)).detach()
    rms = jacobian.square().mean(dim=0).sqrt()
    return dict(seed=seed, observable='Z3', generic_points=19,
                jacobian_rms=rms.detach().tolist(),
                max_finite_difference_error=float((jacobian-finite_difference).abs().max()),
                locally_active=(rms > 1e-8).tolist(),
                structural_redundancy='All four RZ in layer 2 commute with the pulled-back Z0 Z1 Z2 Z3 observable')


def synthetic_learning(seed=1729, steps=100):
    """Fixed deterministic parity of two antipodal encodings, no biomedical data."""
    seed_all(seed)
    model = VQC()
    # Equal priors; only the fourth coordinate varies. Target=1 near angle 0,
    # target=0 near +/-pi. Nonzero other angles avoid degenerate derivatives.
    z = torch.tensor([[.2, -.3, .4, q] for q in [-5., -4., 0., .1, 4., 5.]], dtype=torch.float64)
    y = torch.tensor([0., 0., 1., 1., 0., 0.], dtype=torch.float64)
    with torch.no_grad():
        model.weights.copy_(torch.full_like(model.weights, .9))
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=.03)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = loss_fn(model(z), y)
        losses.append(float(loss.detach()))
        loss.backward()
        if not torch.isfinite(model.weights.grad).all():
            raise ValueError('Nonfinite synthetic gradient')
        optimizer.step()
    with torch.no_grad():
        p = model(z)
    return dict(seed=seed, steps=steps, initial_loss=losses[0],
                final_loss=float(loss_fn(p,y)), accuracy=float(((p>=.5)==y.bool()).double().mean()),
                loss_curve=losses, purpose='Circuit learnability sanity check only, not comparative evidence')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--development-angles', action='store_true')
    args = parser.parse_args()
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=False)
    results = dict(sensitivity=[circuit_sensitivity(s) for s in [1729,1730,1731]],
                   synthetic=synthetic_learning())
    if args.development_angles:
        from .data import load_development, Representation
        from .models import Model
        c, _, protocol, expression, _ = load_development()
        fold = next(f for f in protocol['folds'] if f['fold'] == 0)
        records = []
        for name, components in [('A',4),('B',32)]:
            representation = Representation(c['preprocessing'], components).fit(expression.loc[fold['train_case_ids']])
            for split, ids in [('train',fold['train_case_ids']),('validation',fold['validation_case_ids'])]:
                x = torch.tensor(representation.transform(expression.loc[ids]))
                for seed in [1729,1730,1731]:
                    seed_all(seed)
                    model = Model(components, encoder=name=='B')
                    with torch.no_grad():
                        records.append(dict(model=name,split=split,seed=seed, **angular_summary(model.encoder(x))))
        results['development_initial_angles'] = records
    (target/'diagnostics.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    print(json.dumps(results,indent=2))


if __name__ == '__main__':
    main()
