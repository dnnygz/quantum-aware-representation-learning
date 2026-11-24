import numpy as np, time
from vqc import VQC
from vqc_enc import VQCEnc, grad_z, zexp, run
import vqc
rng = np.random.RandomState(1); X = rng.uniform(-np.pi, np.pi, (300, 4)); y = (np.sin(X[:,0]) + X[:,1] > 0).astype(float)
# gradient check vs finite differences for a re-uploading circuit
P = rng.normal(0, 1, (3, 4, 2)); up = (True, True, True)
G = grad_z(P, X, up, True); h = 1e-5; k = 5
pf = P.reshape(-1).copy(); pf[k] += h; pb = pf.copy(); pb[k] -= 2 * h
fd = (zexp(run(pf.reshape(P.shape), X, up, True)) - zexp(run(pb.reshape(P.shape), X, up, True))) / (2 * h)
print("grad max err (re-upload, 3 layers):", np.abs(G[:, :, k] - fd).max())
# base equivalence
a = VQC(lr=0.05, epochs=50, seed=3).fit(X, y); b = VQCEnc(lr=0.05, epochs=50, seed=3).fit(X, y)
print("base equivalence max |dlogit|:", np.abs(a.decision_function(X) - b.decision_function(X)).max())
# frequency check: output as function of one feature, single upload vs 2 uploads (Fourier content)
for ups in [(True, False), (True, True)]:
    m = VQCEnc(uploads=ups); m.p = rng.normal(0, 1.5, (2, 4, 2)); m.w = np.ones(4); m.b = 0
    xs = np.linspace(-np.pi, np.pi, 256, endpoint=False); Xg = np.zeros((256, 4)); Xg[:, 0] = xs
    f = m.decision_function(Xg); F = np.abs(np.fft.rfft(f)) / 256
    print(ups, "freqs with weight>1e-8:", [i for i, v in enumerate(F) if v > 1e-8])
# timing
for L, ups in [(2, (True, False)), (2, (True, True)), (4, (True, False, False, False)), (4, (True,) * 4)]:
    t = time.time(); VQCEnc(n_layers=L, uploads=ups, epochs=20).fit(np.tile(X, (2, 1))[:420], np.tile(y, 2)[:420]); print(L, ups, "s/epoch", round((time.time() - t) / 20, 3))
