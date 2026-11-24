"""E1 (minimal): VQC vs classical models with the SAME 4 inputs, 30 seeds, same splits.
Also label-randomization control. Config frozen from the pilot: lr=0.05, epochs=200 (shared by VQC and MLP)."""
import sys
from multiprocessing import Pool
from common import *
LR, EPOCHS = 0.05, 200; NSEEDS = NSEEDS_ENV

def run_ds(name):
    X, y = load_datasets()[name]
    rows = []
    for seed in range(NSEEDS):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
        A, B = preprocess(Xtr, Xte)
        ysh = np.random.RandomState(10_000 + seed).permutation(ytr)
        for mn, mk in make_models(seed, LR, EPOCHS).items():
            for cond, yy in (("real", ytr), ("shuffled", ysh)):
                r = evaluate(mk, A, yy, B, yte)
                rows.append(dict(ds=name, seed=seed, model=mn, cond=cond, **r))
        print(name, seed, flush=True)
    return rows

if __name__ == "__main__":
    names = list(load_datasets())
    with Pool(NJOBS) as p:
        out = p.map(run_ds, names, chunksize=1)
    pd.DataFrame([r for o in out for r in o]).to_csv("e1_results.csv", index=False)
    print("DONE")
