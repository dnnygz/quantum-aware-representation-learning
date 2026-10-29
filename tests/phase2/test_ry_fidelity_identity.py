"""Synthetic identities for the existing RY map and terminal VQC, without fitting."""

import json
import platform

import numpy as np
import pytest

torch = pytest.importorskip("torch")
qml = pytest.importorskip("pennylane")

from src.phase2.models import VQC, angles


def verification_summary():
    """Compare analytic overlaps with matrices extracted from the unchanged circuit."""
    errors = dict(fidelity=0., terminal_invariance=0., unitarity=0.,
                  input_gradient=0., finite_difference=0., terminal_gradient=0.,
                  observable_reconstruction=0.)
    different_head_effect = 0.
    local_head_response = 0.
    count = 0
    basis = torch.zeros(16, dtype=torch.complex128)
    basis[0] = 1
    z3 = torch.tensor([1., -1.] * 8, dtype=torch.float64)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1729)
        head = VQC()
    matrix = qml.matrix(head.circuit, wire_order=range(4))
    for seed in [1729, 1730, 1731]:
        generator = torch.Generator().manual_seed(seed)
        for case in range(16):
            z = 3 * torch.randn(2, 4, generator=generator, dtype=torch.float64)
            if case == 0:
                z = torch.tensor([[20., -20., 0., 2.],
                                  [-20., 20., 0., -2.]], dtype=torch.float64)
            a = angles(z).detach().requires_grad_()
            w = (.1 * torch.randn(2, 4, 2, generator=generator,
                                 dtype=torch.float64)).requires_grad_()
            unitary = matrix(a[0], w)
            first = unitary @ basis
            second = matrix(a[1], w) @ basis
            actual = torch.vdot(first, second).abs().square()
            delta = a[0] - a[1]
            factors = torch.cos(delta / 2).square()
            expected = factors.prod()
            errors['fidelity'] = max(errors['fidelity'], abs(float((actual - expected).detach())))
            errors['unitarity'] = max(errors['unitarity'], float(
                (unitary.conj().T @ unitary - torch.eye(16)).abs().max().detach()))
            altered = w.detach() + .7
            changed_first = matrix(a[0], altered) @ basis
            changed_second = matrix(a[1], altered) @ basis
            changed = torch.vdot(changed_first, changed_second).abs().square()
            errors['terminal_invariance'] = max(errors['terminal_invariance'],
                                                abs(float((actual - changed).detach())))
            different_head_effect = max(different_head_effect, abs(float((
                torch.vdot(first, changed_second).abs().square() - actual).detach())))
            grad_a, grad_w = torch.autograd.grad(actual, (a, w), retain_graph=True)
            analytic = torch.stack([-.5 * torch.sin(delta[j]) *
                                    torch.cat((factors[:j], factors[j+1:])).prod()
                                    for j in range(4)])
            errors['input_gradient'] = max(errors['input_gradient'], float(
                (grad_a - torch.stack((analytic, -analytic))).abs().max().detach()))
            errors['terminal_gradient'] = max(errors['terminal_gradient'],
                                               float(grad_w.abs().max()))
            for j in range(4):
                shifted = a.detach().clone()
                shifted[0, j] += 1e-5
                plus = torch.vdot(matrix(shifted[0], w) @ basis, second).abs().square()
                shifted[0, j] -= 2e-5
                minus = torch.vdot(matrix(shifted[0], w) @ basis, second).abs().square()
                difference = (plus - minus) / 2e-5
                errors['finite_difference'] = max(errors['finite_difference'],
                                                   abs(float((difference - analytic[j]).detach())))
            observable = (first.conj() * z3 * first).sum().real
            errors['observable_reconstruction'] = max(errors['observable_reconstruction'],
                abs(float((observable - head.circuit(a[0], w)).detach())))
            response = torch.autograd.grad(observable, w)[0]
            local_head_response = max(local_head_response, float(response.abs().max()))
            count += 1
    return dict(status='synthetic_identity_verification_only', cases=count,
                seeds=[1729, 1730, 1731], dtype='float64/complex128',
                max_absolute_errors=errors, different_head_effect=different_head_effect,
                nonzero_local_observable_head_response=local_head_response,
                python=platform.python_version(), numpy=np.__version__,
                torch=torch.__version__, pennylane=qml.__version__,
                training_executed=False, patient_data_accessed=False)


def test_ry_fidelity_and_terminal_invariance():
    result = verification_summary()
    for name, error in result['max_absolute_errors'].items():
        assert error < (1e-8 if name == 'finite_difference' else 1e-10), (name, error)
    assert result['different_head_effect'] > 1e-3
    assert result['nonzero_local_observable_head_response'] > 1e-3
    print(json.dumps(result, indent=2))


def test_ry_identical_orthogonal_and_periodic_limits():
    zero = torch.zeros(4, dtype=torch.float64)
    orthogonal = zero.clone()
    orthogonal[0] = np.pi
    periodic = zero.clone()
    periodic[0] = 2 * np.pi
    assert torch.cos((zero - zero) / 2).square().prod() == 1
    assert torch.cos((zero - orthogonal) / 2).square().prod() < 1e-28
    assert abs(float(torch.cos((zero - periodic) / 2).square().prod()) - 1) < 1e-14
    # Boundary angles are mathematical checks, not finite encoder outputs.


if __name__ == '__main__':
    print(json.dumps(verification_summary(), indent=2))
