"""H5 (optimization): VQC test/train AUC at several training lengths for several (lr, init) settings.
One long run per (dataset, seed, setting); Adam is deterministic, so checkpoints equal separate shorter runs.
The MLP gets the same lr and the same epochs (separate fits). Test AUC is reported as a function of the factor;
no configuration is selected with it."""
import sys, time
from multiprocessing import Pool
from common import *
CKPT = [25, 50, 100, 200, 400, 800]
SETTINGS = {"base(lr.05,init.1)": dict(lr=0.05, init_mode="normal", init_std=0.1),
            "lr.01": dict(lr=0.01, init_mode="normal", init_std=0.1),
            "lr.2": dict(lr=0.2, init_mode="normal", init_std=0.1),
            "init~N(0,1)": dict(lr=0.05, init_mode="normal", init_std=1.0),
            "init~U(-pi,pi)": dict(lr=0.05, init_mode="uniform", init_std=0.1)}
NSEEDS = NSEEDS_ENV

def bce(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

def task(args):
    name, sname = args
    X, y = load_datasets()[name]; kw = SETTINGS[sname]; rows = []
    for seed in range(NSEEDS):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
        A, B = preprocess(Xtr, Xte)
        m = VQC(epochs=max(CKPT), seed=seed, checkpoints=CKPT, **kw).fit(A, ytr)
        for ep in CKPT:
            m.p, m.w, m.b = m.snaps[ep]
            dtr, dte = m.decision_function(A), m.decision_function(B)
            ptr, pte = m.predict_proba1(A), m.predict_proba1(B)
            rows.append(dict(ds=name, setting=sname, model="VQC", epochs=ep, seed=seed,
                             auc_tr=auc(ytr, dtr), auc_te=auc(yte, dte), bce_tr=bce(ytr, ptr), brier_te=brier_score_loss(yte, pte)))
        if sname == "base(lr.05,init.1)":      # MLP reference, same lr and epochs as the base VQC setting
            for ep in CKPT:
                mm = MLPClassifier((3,), activation="tanh", solver="adam", batch_size=10**6, learning_rate_init=0.05,
                                   max_iter=ep, tol=0, n_iter_no_change=10**6, random_state=seed).fit(A, ytr)
                dtr = mm.predict_proba(A)[:, 1]; dte = mm.predict_proba(B)[:, 1]
                rows.append(dict(ds=name, setting="MLP(lr.05)", model="MLP", epochs=ep, seed=seed,
                                 auc_tr=auc(ytr, dtr), auc_te=auc(yte, dte), bce_tr=bce(ytr, mm.predict_proba(A)[:, 1]),
                                 brier_te=brier_score_loss(yte, mm.predict_proba(B)[:, 1])))
        print(name, sname, seed, flush=True)
    return rows

if __name__ == "__main__":
    names = ["WDBC", "Ionosphere", "Synthetic", "Transfusion"]
    jobs = [(n, s) for s in SETTINGS for n in names]
    with Pool(NJOBS) as p:
        out = p.map(task, jobs, chunksize=1)
    pd.DataFrame([r for o in out for r in o]).to_csv("h5_results.csv", index=False)
    print("DONE")
