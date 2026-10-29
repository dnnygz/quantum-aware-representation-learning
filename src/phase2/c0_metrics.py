"""C0 frozen-readout, geometry, and response diagnostics; no encoder fitting."""
import time
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from .evaluation import metrics
from .models import VQC, angles


EPS = 1e-7


def classification(y, p):
    p = np.asarray(p)
    result = metrics(y, p)
    decision = p >= .5
    result.update(brier=float(np.mean((p-np.asarray(y))**2)),
                  sensitivity_er_positive=float(decision[np.asarray(y)==1].mean()),
                  sensitivity_er_negative=float((~decision[np.asarray(y)==0]).mean()))
    return result


def kernel(a, b, geometry):
    difference = a[:, None, :] - b[None, :, :]
    if geometry == 'Q':
        return np.prod(np.cos(difference/2)**2, axis=-1)
    if geometry == 'E':
        return np.exp(-np.sum(difference**2, axis=-1)/4)
    raise ValueError('Unknown geometry')


def state_vectors(a):
    state = np.ones((len(a), 1))
    for j in range(4):
        local = np.stack([np.cos(a[:, j]/2), np.sin(a[:, j]/2)], axis=-1)
        state = (state[:, :, None]*local[:, None, :]).reshape(len(a), -1)
    return state


def spectrum(x):
    singular = np.linalg.svd(x-x.mean(axis=0), compute_uv=False)
    variance = singular**2
    total = variance.sum()
    if total == 0:
        return dict(singular_values=singular.tolist(), numerical_rank=0,
                    entropy_rank=None, participation_ratio=None)
    weight = variance/total
    positive = weight > 0
    return dict(singular_values=singular.tolist(), numerical_rank=int((singular > 1e-6*singular[0]).sum()),
                entropy_rank=float(np.exp(-np.sum(weight[positive]*np.log(weight[positive])))),
                participation_ratio=float(1/np.sum(weight**2)))


def reference_scores(train_angles, train_labels, queries, geometry, *, training=False):
    """Class means with explicit leave-one-out training references, without tuning."""
    values = kernel(queries, train_angles, geometry)
    means = []
    for c in [0, 1]:
        group = np.flatnonzero(train_labels == c)
        if len(group) < 2:
            raise ValueError('Class reference sets require two patients per class')
        sums = values[:, group].sum(axis=1)
        denominator = np.full(len(queries), len(group), dtype=float)
        if training:
            if not np.array_equal(queries, train_angles):
                raise ValueError('Training queries must equal the ordered reference features')
            sums[group] -= values[group, group]
            denominator[group] -= 1
        means.append(sums/denominator)
    return np.clip((1+means[1]-means[0])/2, EPS, 1-EPS)


def geometry(z, labels, head):
    """Keep pair risk, trace distance, and observable contrast as distinct outputs."""
    a = angles(torch.as_tensor(z)).numpy()
    pair_summaries = {}
    for kind in ['Q', 'E']:
        k = kernel(a, a, kind)
        groups = [np.flatnonzero(labels == c) for c in [1, 0]]
        plus, minus = groups
        positive = k[np.ix_(plus, plus)][np.triu_indices(len(plus), 1)]
        negative = k[np.ix_(minus, minus)][np.triu_indices(len(minus), 1)]
        cross = k[np.ix_(plus, minus)].reshape(-1)
        if min(len(positive), len(negative), len(cross)) == 0:
            raise ValueError('Both classes require non-diagonal evaluation pairs')
        risks = [np.mean((positive-1)**2), np.mean((negative-1)**2), np.mean(cross**2)]
        pair_summaries[kind] = dict(balanced_pair_risk=float(.25*risks[0]+.25*risks[1]+.5*risks[2]),
            balanced_separation=float(.5*(positive.mean()+negative.mean())-cross.mean()),
            strata={name: dict(count=len(values), mean=float(values.mean()), sd=float(values.std()),
                               quantiles=np.quantile(values,[.05,.5,.95]).tolist(),
                               fraction_below_001=float((values<.01).mean()),
                               fraction_above_099=float((values>.99).mean()), residual=float(risk))
                    for name, values, risk in zip(['++','--','+-'],[positive,negative,cross],risks)})
    states = state_vectors(a)
    rho = [states[labels==c].T @ states[labels==c]/sum(labels==c) for c in [0,1]]
    difference = rho[1]-rho[0]
    trace_distance = float(np.abs(np.linalg.eigvalsh(difference)).sum()/2)
    with torch.no_grad():
        raw_p = (1+head.circuit(torch.tensor(a), head.weights).numpy())/2
    raw_contrast = float(raw_p[labels==1].mean()-raw_p[labels==0].mean())
    clipped = np.clip(raw_p, EPS, 1-EPS)
    observable = dict(raw_class_mean_contrast=raw_contrast,
                      clipped_class_mean_contrast=float(clipped[labels==1].mean()-clipped[labels==0].mean()),
                      accessibility_ratio=abs(raw_contrast)/trace_distance if trace_distance>1e-12 else None,
                      ratio_is_complete_discriminative_capacity=False)
    return dict(pair_geometry=pair_summaries,
        class_states=dict(purity_er_negative=float(np.trace(rho[0]@rho[0])),
                          purity_er_positive=float(np.trace(rho[1]@rho[1])),
                          cross_overlap=float(np.trace(rho[0]@rho[1])),
                          hilbert_schmidt_squared=float(np.trace(difference@difference)),
                          trace_distance=trace_distance, purity_includes_diagonals=True),
        observable=observable, latent_spectrum=spectrum(z), angular_spectrum=spectrum(a),
        coordinate_sd=np.std(z,axis=0).tolist(),
        class_mean_difference=(z[labels==1].mean(axis=0)-z[labels==0].mean(axis=0)).tolist(),
        saturation_95=float((np.abs(np.tanh(z/2))>.95).mean()),
        saturation_99=float((np.abs(np.tanh(z/2))>.99).mean()))


def responses(z, head):
    """Differentiate fixed checkpoints; these sensitivities do not measure utility."""
    x = torch.tensor(z, dtype=torch.float64, requires_grad=True)
    a = angles(x)
    m = head.circuit(a, head.weights)
    angular = torch.autograd.grad(m.sum(), a, retain_graph=True)[0]
    direct = torch.autograd.grad(m.sum(), x, retain_graph=True)[0]
    probability = ((1+m)/2).clamp(EPS,1-EPS)
    probability_gradient = torch.autograd.grad(probability.sum(), x)[0]
    slope = (np.pi/2)*(1-torch.tanh(x/2).square())
    return dict(mean_map_derivative=slope.detach().mean(0).tolist(),
                rms_expectation_angular_response=angular.detach().square().mean(0).sqrt().tolist(),
                rms_probability_latent_response=probability_gradient.detach().square().mean(0).sqrt().tolist(),
                chain_rule_max_error=float((direct-angular*slope).detach().abs().max()))


def evaluate_encoder(z, labels, head, *, fit_probes=True):
    """Use training references and probes; validation never enters any fitting call."""
    a = {split: angles(torch.tensor(values)).numpy() for split, values in z.items()}
    output = dict(readouts={}, geometry={}, responses={}, calibration={}, timing={})
    estimators = {}
    for kind in ['Q', 'E']:
        tick=time.perf_counter()
        output['readouts']['class_reference_'+kind] = {
            split: reference_scores(a['train'], labels['train'], a[split], kind, training=split=='train')
            for split in z}
        output['timing']['reference_'+kind]=time.perf_counter()-tick
    if fit_probes:
        for name, values in [('LR_z',z),('LR_angle',a)]:
            estimator = LogisticRegression(C=1.,max_iter=2000,solver='lbfgs',tol=1e-4,
                fit_intercept=True,class_weight=None,warm_start=False,random_state=labels.get('seed',1729))
            tick=time.perf_counter()
            estimator.fit(values['train'],labels['train'])
            output['timing'][name+'_fit']=time.perf_counter()-tick
            output['readouts'][name] = {split: estimator.predict_proba(values[split])[:,1] for split in values}
            estimators[name]=estimator
    for split in z:
        tick=time.perf_counter()
        output['geometry'][split]=geometry(z[split],labels[split],head)
        output['timing'][split+'_geometry']=time.perf_counter()-tick
        tick=time.perf_counter()
        output['responses'][split]=responses(z[split],head)
        output['timing'][split+'_responses']=time.perf_counter()-tick
    output['direct_diagnostic_qnode_calls']=4
    output['direct_diagnostic_sample_states']=2*sum(len(a) for a in z.values())
    return output, estimators


def calibration(y, p):
    bins = np.minimum((np.asarray(p)*10).astype(int),9)
    return [dict(bin=i,count=int((bins==i).sum()),
                 mean_probability=float(np.asarray(p)[bins==i].mean()) if (bins==i).any() else None,
                 event_rate=float(np.asarray(y)[bins==i].mean()) if (bins==i).any() else None)
            for i in range(10)]
