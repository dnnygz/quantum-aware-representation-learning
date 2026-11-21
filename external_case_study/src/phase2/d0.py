"""Frozen-model D0 calculations: variation, functional response, and prediction changes."""
import numpy as np
import torch
from scipy.spatial import cKDTree
from .models import VQC
from .analyze_cv import latent_summary
from .evaluation import metrics

ALPHAS = (0.01, 0.05, 0.10)
DONOR_SEEDS = (271828, 271829, 271830)


def phi(z):
    return np.pi * np.tanh(np.asarray(z)/2)


def training_reference(z, y):
    """All perturbation scales, axes, bounds, and class descriptors use training only."""
    z = np.asarray(z, dtype=float)
    centered = z-z.mean(axis=0)
    # Direct SVD avoids covariance cancellation manufacturing variance in
    # near-constant directions. These scales also define the local PC shifts.
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    values, axes = singular**2/len(z), vt.T.copy()
    # Orient eigenvectors deterministically; coordinates are never matched across encoders.
    for j in range(4):
        if axes[np.argmax(np.abs(axes[:, j])), j] < 0:
            axes[:, j] *= -1
    return dict(mean=z.mean(axis=0), sd=z.std(axis=0), minimum=z.min(axis=0),
                maximum=z.max(axis=0), axes=axes, eigenvalues=values,
                class_difference=z[np.asarray(y)==1].mean(axis=0)-z[np.asarray(y)==0].mean(axis=0))


class FrozenHead:
    """A saved head with no optimizer, fitting API, or encoder forward pass."""
    def __init__(self, name, state=None, estimator=None):
        self.name = name
        self.quantum = name in ['A', 'B']
        if self.quantum:
            with torch.random.fork_rng(devices=[]):
                self.vqc = VQC()
            self.weights = state['head.weights'].detach().clone()
        elif name == 'B_classical':
            self.weight = state['head.0.weight'].detach().clone().reshape(4)
            self.bias = state['head.0.bias'].detach().clone().reshape(())
        else:
            self.weight = torch.tensor(estimator.coef_[0], dtype=torch.float64)
            self.bias = torch.tensor(estimator.intercept_[0], dtype=torch.float64)

    def expectation(self, theta):
        return self.vqc.circuit(theta, self.weights)

    def tensor_probability(self, z):
        if self.quantum:
            f = self.expectation(np.pi*torch.tanh(z/2))
            return ((1+f)/2).clamp(1e-7, 1-1e-7)
        return torch.sigmoid(z @ self.weight+self.bias)

    def predict(self, z):
        with torch.no_grad():
            return self.tensor_probability(torch.tensor(z, dtype=torch.float64)).numpy()

    def derivatives(self, z):
        """Batch rows are independent; differentiating their sum gives per-row derivatives."""
        x = torch.tensor(z, dtype=torch.float64, requires_grad=True)
        p = self.tensor_probability(x)
        gp = torch.autograd.grad(p.sum(), x)[0]
        result = dict(probability=p.detach().numpy(), probability_gradient=gp.detach().numpy())
        if self.quantum:
            theta = torch.tensor(phi(z), dtype=torch.float64, requires_grad=True)
            f = self.expectation(theta)
            ga = torch.autograd.grad(f.sum(), theta)[0]
            x2 = torch.tensor(z, dtype=torch.float64, requires_grad=True)
            fx = self.expectation(np.pi*torch.tanh(x2/2))
            gz = torch.autograd.grad(fx.sum(), x2)[0]
            derivative = np.pi/2*(1-np.tanh(np.asarray(z)/2)**2)
            chain = ga.detach().numpy()*derivative
            result.update(expectation=f.detach().numpy(), map_derivative=derivative,
                angle_gradient=ga.detach().numpy(), expectation_gradient=gz.detach().numpy(),
                chain_reconstructed=chain, chain_absolute_error=np.abs(gz.detach().numpy()-chain))
        return result

    def numerical_checks(self, z, count=19, step=1e-5):
        """Fixed canonical-order subset, chosen without inspecting errors or labels."""
        z = np.asarray(z[:count]).copy()
        result = self.derivatives(z)
        errors = []
        for j in range(4):
            delta = np.zeros_like(z); delta[:, j] = step
            fd = (self.predict(z+delta)-self.predict(z-delta))/(2*step)
            errors.append(np.max(np.abs(fd-result['probability_gradient'][:, j])))
        checks = dict(fd_probability_max_abs_error=float(max(errors)), fd_step=step, fd_patients=len(z))
        if self.quantum:
            with torch.no_grad():
                t = torch.tensor(phi(z), dtype=torch.float64)
                shifted = self.weights.clone(); shifted[1, :, 1] += .1
                redundant_error = (self.vqc.circuit(t, shifted)-self.expectation(t)).abs().max().item()
            checks.update(chain_max_abs_error=float(result['chain_absolute_error'].max()),
                final_rz_perturbation_max_abs_error=redundant_error)
        return checks


def interventions(z, training, reference, fold, seed, split):
    """Generate the entire prespecified design without labels or validation-derived scales."""
    z = np.asarray(z)
    for j in range(4):
        for family in ['mean', 'zero']:
            altered = z.copy(); altered[:, j] = reference['mean'][j] if family=='mean' else 0.
            yield dict(intervention=f'{family}_{j}', family=family, direction=f'coordinate_{j}',
                       alpha=0., sign=0, donor_seed=-1), altered
        for donor_seed in DONOR_SEEDS:
            rng = np.random.default_rng(np.random.SeedSequence([donor_seed, fold, seed, 0 if split=='train' else 1]))
            indices = rng.permutation(len(training))[:len(z)]
            altered = z.copy(); altered[:, j] = training[indices, j]
            yield dict(intervention=f'scramble_{j}_{donor_seed}', family='scramble',direction=f'coordinate_{j}',
                       alpha=0.,sign=0,donor_seed=donor_seed), altered
    for basis in ['coordinate', 'pc']:
        for j in range(4):
            direction = np.eye(4)[:, j] if basis=='coordinate' else reference['axes'][:, j]
            scale = reference['sd'][j] if basis=='coordinate' else np.sqrt(reference['eigenvalues'][j])
            for alpha in ALPHAS:
                for sign in [-1, 1]:
                    delta = sign*alpha*scale*direction
                    yield dict(intervention=f'{basis}_{j}_{alpha}_{sign}', family=f'local_{basis}',
                        direction=f'{basis}_{j}',alpha=alpha,sign=sign,donor_seed=-1), z+delta


def displacement(z, altered, training, reference, split):
    """Effective shifts and limited support diagnostics; bounds are not a manifold test."""
    dz = altered-z; da = phi(altered)-phi(z)
    rows = {}
    for prefix, delta in [('z', dz), ('angle', da)]:
        norm = np.linalg.norm(delta, axis=1)
        rows.update({f'{prefix}_l2_mean':float(norm.mean()),f'{prefix}_l2_median':float(np.median(norm)),
                     f'{prefix}_l2_q95':float(np.quantile(norm,.95)),f'{prefix}_l2_max':float(norm.max())})
    floor = 1e-12*max(1., float(reference['sd'].max()))
    rows['train_scaled_l2_mean'] = float(np.linalg.norm(dz/np.maximum(reference['sd'],floor),axis=1).mean())
    rows['scale_floor'] = floor
    tolerance = 1e-12*np.maximum(1., np.abs(reference['maximum']-reference['minimum']))
    def outside(a):
        return ((a < reference['minimum']-tolerance)|(a > reference['maximum']+tolerance)).any(axis=1)
    rows['outside_training_bounds_before'] = float(outside(z).mean())
    rows['outside_training_bounds_after'] = float(outside(altered).mean())
    tree = cKDTree(training)
    def nearest(a):
        if split=='validation':
            return tree.query(a, k=1)[0]
        distances, indices = tree.query(a, k=2)
        return np.where(indices[:,0]==np.arange(len(a)),distances[:,1],distances[:,0])
    rows['nearest_training_distance_before'] = float(nearest(z).mean())
    rows['nearest_training_distance_after'] = float(nearest(altered).mean())
    return rows


def predictive_change(y, baseline, altered):
    before, after = metrics(y, baseline), metrics(y, altered)
    change = altered-baseline
    return {**{f'delta_{m}':float(after[m]-before[m]) for m in before if m!='confusion_matrix'},
            'delta_brier':float(np.mean((altered-y)**2)-np.mean((baseline-y)**2)),
            'probability_change_signed_mean':float(change.mean()),
            'probability_change_abs_mean':float(np.abs(change).mean()),
            'probability_change_abs_q95':float(np.quantile(np.abs(change),.95)),
            'decision_flip_fraction':float(((altered>=.5)!=(baseline>=.5)).mean())}


def response_summary(gradient, reference):
    """Aggregate local functional sensitivity; no labels enter this calculation."""
    gram = gradient.T @ gradient/len(gradient)
    values = np.linalg.eigvalsh(gram)[::-1].clip(0)
    weights = values/values.sum() if values.sum() else np.zeros(4)
    positive = weights[weights>0]
    return dict(gradient_norm_mean=float(np.linalg.norm(gradient,axis=1).mean()),
        gradient_norm_rms=float(np.sqrt(np.mean(np.sum(gradient**2,axis=1)))),
        training_sd_scaled_gradient_norm_rms=float(np.sqrt(np.mean(np.sum((gradient*reference['sd'])**2,axis=1)))),
        sensitivity_entropy_rank=float(np.exp(-np.sum(positive*np.log(positive)))) if len(positive) else 0.,
        **{f'G_eigenvalue_{j}':float(values[j]) for j in range(4)})


def geometry(z):
    return latent_summary(np.asarray(z))
