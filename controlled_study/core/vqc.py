"""Exact statevector simulation of the 4-qubit VQC (noise-free), numpy only.
Circuit: RY(x_j) angle encoding -> L x [RY(theta) RZ(phi) per qubit, CNOT chain 0->1->2->3]
Readout: <Z_q> on all qubits -> linear layer -> sigmoid.
Gradients of <Z> w.r.t. circuit angles: parameter-shift rule (exact for RY/RZ)."""
import numpy as np

NQ = 4
DIM = 2 ** NQ
I2 = np.eye(2)

# basis index bits: q0 is the most significant bit
_bits = np.array([[(i >> (NQ - 1 - q)) & 1 for q in range(NQ)] for i in range(DIM)])
ZMAT = 1.0 - 2.0 * _bits                       # (16,4): eigenvalue of Z_q on each basis state

def _cnot_perm_matrix(chain=((0, 1), (1, 2), (2, 3))):
    P = np.eye(DIM)
    for c, t in chain:
        M = np.zeros((DIM, DIM))
        for i in range(DIM):
            b = _bits[i].copy()
            if b[c] == 1:
                b[t] ^= 1
            j = int("".join(map(str, b)), 2)
            M[j, i] = 1.0
        P = M @ P
    return P

CNOT_CHAIN = _cnot_perm_matrix()
NO_ENT = np.eye(DIM)

def _ry(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)

def _rz(p):
    return np.array([[np.exp(-1j * p / 2), 0], [0, np.exp(1j * p / 2)]])

def circuit_unitary(params, n_layers, entangle=True):
    """params shape (n_layers, 4, 2): [layer, qubit, (theta_RY, phi_RZ)]."""
    ent = CNOT_CHAIN if entangle else NO_ENT
    V = np.eye(DIM, dtype=complex)
    for l in range(n_layers):
        U = np.array([[1.0]], dtype=complex)
        for q in range(NQ):
            U = np.kron(U, _rz(params[l, q, 1]) @ _ry(params[l, q, 0]))
        V = ent @ U @ V
    return V

def encode(X):
    """X (N,4) angles -> product state (N,16) real amplitudes."""
    psi = np.ones((X.shape[0], 1))
    for q in range(NQ):
        a = np.stack([np.cos(X[:, q] / 2), np.sin(X[:, q] / 2)], 1)
        psi = (psi[:, :, None] * a[:, None, :]).reshape(X.shape[0], -1)
    return psi

def expz(params, psi0, n_layers, entangle=True):
    V = circuit_unitary(params, n_layers, entangle)
    psi = psi0 @ V.T
    return (psi.real ** 2 + psi.imag ** 2) @ ZMAT          # (N,4)

def expz_grad(params, psi0, n_layers, entangle=True):
    """Returns dZ with shape (N, 4, P) via parameter shift, P = n_layers*4*2."""
    flat = params.reshape(-1)
    out = np.empty((psi0.shape[0], NQ, flat.size))
    for k in range(flat.size):
        e = np.zeros_like(flat); e[k] = np.pi / 2
        zp = expz((flat + e).reshape(params.shape), psi0, n_layers, entangle)
        zm = expz((flat - e).reshape(params.shape), psi0, n_layers, entangle)
        out[:, :, k] = 0.5 * (zp - zm)
    return out

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

class VQC:
    def __init__(self, n_layers=2, entangle=True, head=True, lr=0.05, epochs=150,
                 init_std=0.1, seed=0, track=False, init_mode="normal", checkpoints=()):
        self.L, self.ent, self.head = n_layers, entangle, head
        self.lr, self.epochs, self.init_std, self.seed, self.track = lr, epochs, init_std, seed, track
        self.init_mode, self.checkpoints, self.snaps = init_mode, set(checkpoints), {}

    def n_params(self):
        return self.L * NQ * 2 + (NQ + 1 if self.head else 0)

    def _logit(self, Z):
        return Z @ self.w + self.b if self.head else Z.sum(1) * 1.0 + self.b

    def fit(self, X, y):
        rng = np.random.RandomState(self.seed)
        self.p = (rng.uniform(-np.pi, np.pi, (self.L, NQ, 2)) if self.init_mode == "uniform"
                  else rng.normal(0, self.init_std, (self.L, NQ, 2)))
        self.w = rng.normal(0, self.init_std, NQ) if self.head else np.ones(NQ)
        self.b = 0.0
        psi0 = encode(X)
        theta = [self.p.reshape(-1).copy(), self.w.copy(), np.array([self.b])]
        m = [np.zeros_like(t) for t in theta]; v = [np.zeros_like(t) for t in theta]
        b1, b2, eps = 0.9, 0.999, 1e-8
        self.hist = []
        for t in range(1, self.epochs + 1):
            P = theta[0].reshape(self.L, NQ, 2)
            Z = expz(P, psi0, self.L, self.ent)
            logit = Z @ theta[1] + theta[2][0] if self.head else Z.sum(1) + theta[2][0]
            pr = sigmoid(logit)
            r = (pr - y) / len(y)                          # dL/dlogit for mean BCE
            dw = Z.T @ r if self.head else np.zeros(NQ)
            db = np.array([r.sum()])
            dZ = r[:, None] * (theta[1] if self.head else np.ones(NQ))[None, :]  # (N,4)
            G = expz_grad(P, psi0, self.L, self.ent)       # (N,4,P)
            dp = np.einsum("nq,nqk->k", dZ, G)
            grads = [dp, dw, db]
            if not self.head:
                grads[1] = np.zeros(NQ)
            for i in range(3):
                m[i] = b1 * m[i] + (1 - b1) * grads[i]
                v[i] = b2 * v[i] + (1 - b2) * grads[i] ** 2
                mh = m[i] / (1 - b1 ** t); vh = v[i] / (1 - b2 ** t)
                if i == 1 and not self.head:
                    continue
                theta[i] = theta[i] - self.lr * mh / (np.sqrt(vh) + eps)
            if t in self.checkpoints:
                self.snaps[t] = (theta[0].reshape(self.L, NQ, 2).copy(), theta[1].copy(), float(theta[2][0]))
            if self.track:
                loss = -np.mean(y * np.log(pr + 1e-12) + (1 - y) * np.log(1 - pr + 1e-12))
                self.hist.append(loss)
        self.p = theta[0].reshape(self.L, NQ, 2); self.w = theta[1]; self.b = theta[2][0]
        return self

    def decision_function(self, X):
        return self._logit(expz(self.p, encode(X), self.L, self.ent))

    def predict_proba1(self, X):
        return sigmoid(self.decision_function(X))
