"""H3 (capacity and readout): VQC variants + matched-size MLPs, same splits/seeds/preprocessing as E1.
Real labels: checkpoints at 200 and 400 epochs. Shuffled labels: 200 epochs (memorization control)."""
from multiprocessing import Pool
from common import *
NSEEDS = NSEEDS_ENV
VQC_VARIANTS = {"L1": dict(n_layers=1), "L2": dict(n_layers=2), "L4": dict(n_layers=4),
                "L2-noent": dict(n_layers=2, entangle=False), "L2-nohead": dict(n_layers=2, head=False)}
MLP_HIDDEN = {"MLP h2 (13p)": 2, "MLP h3 (19p)": 3, "MLP h6 (37p)": 6}
def bce(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

def task(args):
    name, kind = args
    X, y = load_datasets()[name]; rows = []
    for seed in range(NSEEDS):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
        A, B = preprocess(Xtr, Xte)
        ysh = np.random.RandomState(10_000 + seed).permutation(ytr)
        if kind != "MLP":
            kw = VQC_VARIANTS[kind]
            m = VQC(lr=0.05, epochs=400, seed=seed, checkpoints=[200, 400], **kw).fit(A, ytr)
            for ep in (200, 400):
                m.p, m.w, m.b = m.snaps[ep]
                rows.append(dict(ds=name, model=kind, cond="real", epochs=ep, seed=seed, auc_tr=auc(ytr, m.decision_function(A)),
                                 auc_te=auc(yte, m.decision_function(B)), bce_tr=bce(ytr, m.predict_proba1(A)), brier_te=brier_score_loss(yte, m.predict_proba1(B))))
            ms = VQC(lr=0.05, epochs=200, seed=seed, **kw).fit(A, ysh)
            rows.append(dict(ds=name, model=kind, cond="shuffled", epochs=200, seed=seed, auc_tr=auc(ysh, ms.decision_function(A)),
                             auc_te=auc(yte, ms.decision_function(B)), bce_tr=np.nan, brier_te=np.nan))
        else:
            for mn, h in MLP_HIDDEN.items():
                for cond, yy, eps in (("real", ytr, (200, 400)), ("shuffled", ysh, (200,))):
                    for ep in eps:
                        mm = MLPClassifier((h,), activation="tanh", solver="adam", batch_size=10**6, learning_rate_init=0.05,
                                           max_iter=ep, tol=0, n_iter_no_change=10**6, random_state=seed).fit(A, yy)
                        rows.append(dict(ds=name, model=mn, cond=cond, epochs=ep, seed=seed, auc_tr=auc(yy, mm.predict_proba(A)[:, 1]),
                                         auc_te=auc(yte, mm.predict_proba(B)[:, 1]), bce_tr=bce(yy, mm.predict_proba(A)[:, 1]) if cond == "real" else np.nan,
                                         brier_te=brier_score_loss(yte, mm.predict_proba(B)[:, 1]) if cond == "real" else np.nan))
        print(name, kind, seed, flush=True)
    return rows

if __name__ == "__main__":
    names = ["WDBC", "Ionosphere", "Synthetic", "Transfusion"]
    jobs = [(n, k) for k in ["L4", "L2", "L2-noent", "L2-nohead", "L1", "MLP"] for n in names]   # slowest first
    with Pool(NJOBS) as p:
        out = p.map(task, jobs, chunksize=1)
    pd.DataFrame([r for o in out for r in o]).to_csv("h3_results.csv", index=False); print("DONE")
