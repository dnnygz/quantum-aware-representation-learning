"""Nominal vs effective number of circuit parameters = numerical rank of the Jacobian of <Z> w.r.t. the angles."""
import numpy as np
from vqc import *
rng = np.random.RandomState(0)
out = []
for name, L, ent in [("L1", 1, True), ("L2 (base)", 2, True), ("L4", 4, True), ("L2 no entanglement", 2, False)]:
    X = rng.uniform(-np.pi, np.pi, (200, 4)); psi0 = encode(X)
    ranks = []
    for _ in range(5):
        P = rng.uniform(-np.pi, np.pi, (L, 4, 2))
        G = expz_grad(P, psi0, L, ent)                 # (N,4,P)
        M = G.reshape(-1, G.shape[2])
        s = np.linalg.svd(M, compute_uv=False); ranks.append(int((s > 1e-8 * s[0]).sum()))
    out.append((name, L * 8, min(ranks), max(ranks)))
    print(f"{name:22s} nominal circuit params={L*8:3d} effective rank min/max over 5 random points: {min(ranks)}/{max(ranks)}")
import csv; csv.writer(open("eff_params.csv", "w")).writerows([("variant", "nominal", "eff_min", "eff_max")] + out)
