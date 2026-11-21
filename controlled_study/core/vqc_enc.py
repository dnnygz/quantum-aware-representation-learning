"""VQC with configurable data encoding (H2): data re-uploading before selected layers and input scaling.
Same circuit, readout, loss and optimizer as vqc.VQC; with uploads=(True,False,...) and scale=1 it reproduces vqc.VQC.
Exact noise-free statevector; gradients of <Z> by the parameter-shift rule (each variational angle appears in one gate,
so the rule stays exact when the data are re-uploaded)."""
import numpy as np
from vqc import NQ, DIM, ZMAT, CNOT_CHAIN, NO_ENT, _ry, _rz, sigmoid

def upload(psi, Xs):
    """Apply RY(Xs[:,q]) on every qubit q to each sample's state psi (N,16)."""
    N = psi.shape[0]
    for q in range(NQ):
        c = np.cos(Xs[:, q] / 2)[:, None, None]; s = np.sin(Xs[:, q] / 2)[:, None, None]
        a = psi.reshape(N, 2 ** q, 2, 2 ** (NQ - 1 - q))
        a0, a1 = a[:, :, 0, :], a[:, :, 1, :]
        psi = np.stack([c * a0 - s * a1, s * a0 + c * a1], axis=2).reshape(N, DIM)
    return psi

def layer_unitary(p_l, entangle=True):
    U = np.array([[1.0]], dtype=complex)
    for q in range(NQ):
        U = np.kron(U, _rz(p_l[q, 1]) @ _ry(p_l[q, 0]))
    return (CNOT_CHAIN if entangle else NO_ENT) @ U

def run(params, Xs, uploads, entangle, start=0, psi=None):
    """Run layers start..L-1 (re-uploading before the layers flagged in `uploads`); returns final state."""
    if psi is None:
        psi = np.zeros((Xs.shape[0], DIM), dtype=complex); psi[:, 0] = 1.0
    for l in range(start, len(uploads)):
        if uploads[l]:
            psi = upload(psi, Xs)
        psi = psi @ layer_unitary(params[l], entangle).T
    return psi

def zexp(psi):
    return (psi.real ** 2 + psi.imag ** 2) @ ZMAT

def prefix_states(params, Xs, uploads, entangle):
    """State right before layer l's upload/gates, for l = 0..L-1."""
    psi = np.zeros((Xs.shape[0], DIM), dtype=complex); psi[:, 0] = 1.0
    out = []
    for l in range(len(uploads)):
        out.append(psi)
        if uploads[l]:
            psi = upload(psi, Xs)
        psi = psi @ layer_unitary(params[l], entangle).T
    return out

def grad_z(params, Xs, uploads, entangle):
    L = len(uploads); pre = prefix_states(params, Xs, uploads, entangle)
    G = np.empty((Xs.shape[0], NQ, L, NQ, 2))
    for l in range(L):
        for q in range(NQ):
            for k in range(2):
                pp = params.copy(); pp[l, q, k] += np.pi / 2
                pm = params.copy(); pm[l, q, k] -= np.pi / 2
                zp = zexp(run(pp, Xs, uploads, entangle, l, pre[l]))
                zm = zexp(run(pm, Xs, uploads, entangle, l, pre[l]))
                G[:, :, l, q, k] = 0.5 * (zp - zm)
    return G.reshape(Xs.shape[0], NQ, -1)

class VQCEnc:
    def __init__(self, n_layers=2, uploads=None, scale=1.0, entangle=True, lr=0.05, epochs=200,
                 init_std=0.1, seed=0):
        self.L, self.ent, self.lr, self.epochs, self.init_std, self.seed = n_layers, entangle, lr, epochs, init_std, seed
        self.uploads = tuple(uploads) if uploads is not None else tuple([True] + [False] * (n_layers - 1))
        assert len(self.uploads) == n_layers
        self.scale = scale

    def fit(self, X, y):
        rng = np.random.RandomState(self.seed)
        p = rng.normal(0, self.init_std, (self.L, NQ, 2)); w = rng.normal(0, self.init_std, NQ); b = 0.0
        Xs = X * self.scale
        theta = [p.reshape(-1).copy(), w.copy(), np.array([b])]
        m = [np.zeros_like(t) for t in theta]; v = [np.zeros_like(t) for t in theta]
        b1, b2, eps = 0.9, 0.999, 1e-8
        for t in range(1, self.epochs + 1):
            P = theta[0].reshape(self.L, NQ, 2)
            Z = zexp(run(P, Xs, self.uploads, self.ent))
            pr = sigmoid(Z @ theta[1] + theta[2][0])
            r = (pr - y) / len(y)
            dw = Z.T @ r; db = np.array([r.sum()])
            dZ = r[:, None] * theta[1][None, :]
            G = grad_z(P, Xs, self.uploads, self.ent)
            dp = np.einsum("nq,nqk->k", dZ, G)
            for i, g in enumerate([dp, dw, db]):
                m[i] = b1 * m[i] + (1 - b1) * g; v[i] = b2 * v[i] + (1 - b2) * g ** 2
                theta[i] = theta[i] - self.lr * (m[i] / (1 - b1 ** t)) / (np.sqrt(v[i] / (1 - b2 ** t)) + eps)
        self.p = theta[0].reshape(self.L, NQ, 2); self.w = theta[1]; self.b = theta[2][0]
        return self

    def decision_function(self, X):
        return zexp(run(self.p, X * self.scale, self.uploads, self.ent)) @ self.w + self.b

    def predict_proba1(self, X):
        return sigmoid(self.decision_function(X))
