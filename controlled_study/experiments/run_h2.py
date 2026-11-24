"""H2 (encoding). Part A: encoding variants of the VQC on the four datasets (same splits/preprocessing as E1).
Part B: synthetic 4-feature data with controlled frequency/interaction structure, a fresh sample per seed."""
import sys, time
from multiprocessing import Pool
from common import *
from vqc_enc import VQCEnc
NSEEDS = NSEEDS_ENV
VAR = {  # name: kwargs  (L layers; uploads flags; input scale)
    "base":   dict(n_layers=2, uploads=(True, False), scale=1.0),
    "reup2":  dict(n_layers=2, uploads=(True, True), scale=1.0),
    "scale2": dict(n_layers=2, uploads=(True, False), scale=2.0),
    "scale05": dict(n_layers=2, uploads=(True, False), scale=0.5),
    "L4":     dict(n_layers=4, uploads=(True, False, False, False), scale=1.0),
    "L4reup": dict(n_layers=4, uploads=(True, True, True, True), scale=1.0),
}
VAR_A = ["base", "reup2", "scale05", "scale2", "L4", "L4reup"]
VAR_B = ["base", "reup2", "scale2", "L4", "L4reup"]
FAMS = [("add", 1), ("add", 2), ("add", 3), ("prod", 1), ("prod", 2), ("prod", 3)]

def gen_B(fam, k, seed, n=600, noise=0.05):
    rng = np.random.RandomState(10000 + 97 * seed + 13 * k + (0 if fam == "add" else 5))
    X = rng.uniform(-np.pi, np.pi, (n, 4))
    s = np.sin(k * X[:, 0]) + np.sin(k * X[:, 1]) if fam == "add" else np.sin(k * X[:, 0]) * np.sin(k * X[:, 1])
    y = (s > 0).astype(int); flip = rng.rand(n) < noise; y[flip] = 1 - y[flip]
    return X, y

def taskA(args):
    name, seed = args
    X, y = load_datasets()[name]; rows = []
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    A, B = preprocess(Xtr, Xte)
    for vn in VAR_A:
        r = evaluate(lambda: VQCEnc(lr=0.05, epochs=200, seed=seed, **VAR[vn]), A, ytr, B, yte)
        rows.append(dict(part="A", ds=name, seed=seed, model="VQC-" + vn, **r))
    print("A", name, seed, flush=True); return rows

def taskB(args):
    (fam, k), seed = args
    X, y = gen_B(fam, k, seed); rows = []
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    for vn in VAR_B:
        r = evaluate(lambda: VQCEnc(lr=0.05, epochs=200, seed=seed, **VAR[vn]), Xtr, ytr, Xte, yte)
        rows.append(dict(part="B", ds=f"{fam}{k}", seed=seed, model="VQC-" + vn, **r))
    for mn, mk in make_models(seed, 0.05, 200).items():
        if mn == "VQC": continue
        r = evaluate(mk, Xtr, ytr, Xte, yte); rows.append(dict(part="B", ds=f"{fam}{k}", seed=seed, model=mn, **r))
    print("B", fam, k, seed, flush=True); return rows

if __name__ == "__main__":
    which = sys.argv[1]
    if which == "A":
        tasks = [(n, s) for s in range(NSEEDS) for n in ["WDBC", "Ionosphere", "Synthetic", "Transfusion"]]; fn = taskA
    else:
        tasks = [(f, s) for s in range(NSEEDS) for f in FAMS]; fn = taskB
    if len(sys.argv) > 2: tasks = tasks[:int(sys.argv[2])]
    out = []; t0 = time.time()
    with Pool(NJOBS) as p:
        for i, rows in enumerate(p.imap_unordered(fn, tasks, chunksize=1)):
            out += rows
            if i % 10 == 0: pd.DataFrame(out).to_csv(f"h2{which}_partial.csv", index=False)
    pd.DataFrame(out).to_csv(f"h2{which}_results.csv" if len(sys.argv) <= 2 else f"h2{which}_smoke.csv", index=False)
    print("DONE", which, round(time.time() - t0), "s")
