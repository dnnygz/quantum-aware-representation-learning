"""H6 (classical teacher). Arm 1: knowledge distillation -- students trained on soft targets alpha*p_teacher+(1-alpha)*y.
Arm 2: teacher representation -- a classical bottleneck encoder (all features -> 8 -> 4, tanh) trained on the training split,
then a VQC head vs a classical LR head on the same four learned features (encoder + head control).
Same splits, PCA-4 student inputs and budget (lr .05, 200 epochs) as E1. Hard-label baselines come from E1 (same seeds)."""
import sys, time
from multiprocessing import Pool
from common import *
NSEEDS = NSEEDS_ENV; ALPHAS = [0.5, 1.0]

def auc_rows(name, seed, model, ytr, yte, dtr, dte, ptr=None, pte=None):
    r = dict(ds=name, seed=seed, model=model, auc_tr=auc(ytr, dtr), auc_te=auc(yte, dte))
    if pte is not None: r["brier_te"] = brier_score_loss(yte, pte)
    return r

def task(args):
    name, seed = args
    X, y = load_datasets()[name]; rows = []
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    A, B = preprocess(Xtr, Xte)                                   # student inputs (PCA-4 angles)
    sc = StandardScaler().fit(Xtr); S, T = sc.transform(Xtr), sc.transform(Xte)   # teacher inputs (all features)
    # ---- arm 1: distillation from an RBF-SVM teacher fitted on all features
    teacher = SVC(probability=True, random_state=seed).fit(S, ytr)
    pt = teacher.predict_proba(S)[:, 1]
    rows.append(auc_rows(name, seed, "Teacher-SVM", ytr, yte, teacher.decision_function(S), teacher.decision_function(T)))
    for a in ALPHAS:
        soft = a * pt + (1 - a) * ytr
        v = VQC(lr=0.05, epochs=200, seed=seed).fit(A, soft)
        rows.append(auc_rows(name, seed, f"VQC-KD{a}", ytr, yte, v.decision_function(A), v.decision_function(B)))
        w = np.concatenate([soft, 1 - soft]); yy = np.concatenate([np.ones(len(ytr)), np.zeros(len(ytr))]); AA = np.vstack([A, A])
        m = make_models(seed, 0.05, 200)["MLP"]()
        m.fit(AA, yy, sample_weight=w)
        rows.append(auc_rows(name, seed, f"MLP-KD{a}", ytr, yte, scores(m, A)[0], scores(m, B)[0]))
    # ---- arm 2: learned classical representation (not for the 4-feature native dataset)
    if X.shape[1] > 4:
        enc = MLPClassifier((8, 4), activation="tanh", solver="adam", batch_size=10**6, learning_rate_init=0.05,
                            max_iter=200, tol=0, n_iter_no_change=10**6, random_state=seed).fit(S, ytr)
        rows.append(auc_rows(name, seed, "Teacher-MLPenc", ytr, yte, scores(enc, S)[0], scores(enc, T)[0]))
        def z(Z):
            h = np.tanh(Z @ enc.coefs_[0] + enc.intercepts_[0]); return np.tanh(h @ enc.coefs_[1] + enc.intercepts_[1])
        Ztr, Zte = z(S), z(T); lo, hi = Ztr.min(0), Ztr.max(0)
        g = lambda Q: np.clip(2 * np.pi * (Q - lo) / np.maximum(hi - lo, 1e-9) - np.pi, -np.pi, np.pi)
        Gtr, Gte = g(Ztr), g(Zte)
        v = VQC(lr=0.05, epochs=200, seed=seed).fit(Gtr, ytr)
        rows.append(auc_rows(name, seed, "VQC-on-enc", ytr, yte, v.decision_function(Gtr), v.decision_function(Gte)))
        v = VQC(lr=0.05, epochs=200, seed=seed).fit(Gtr / 2, ytr)   # sensitivity: range [-pi/2, pi/2] (the two extremes no longer map to the same state)
        rows.append(auc_rows(name, seed, "VQC-on-enc-half", ytr, yte, v.decision_function(Gtr / 2), v.decision_function(Gte / 2)))
        lr = LogisticRegression(max_iter=2000).fit(Gtr, ytr)
        rows.append(auc_rows(name, seed, "LR-on-enc", ytr, yte, lr.decision_function(Gtr), lr.decision_function(Gte)))
    print(name, seed, flush=True); return rows

if __name__ == "__main__":
    tasks = [(n, s) for s in range(NSEEDS) for n in ["WDBC", "Ionosphere", "Synthetic", "Transfusion"]]
    if len(sys.argv) > 1: tasks = tasks[:int(sys.argv[1])]
    t0 = time.time()
    with Pool(NJOBS) as p: out = [r for rows in p.imap_unordered(task, tasks, chunksize=1) for r in rows]
    pd.DataFrame(out).to_csv("h6_results.csv" if len(sys.argv) == 1 else "h6_smoke.csv", index=False)
    print("DONE", round(time.time() - t0), "s")
