"""H4 (training-set size): fractions of the training split, all models, same test split as E1, fixed budget (lr .05, 200 epochs)."""
from multiprocessing import Pool
from common import *
FR = [0.1, 0.25, 0.5, 0.75, 1.0]; NSEEDS = NSEEDS_ENV
def task(name):
    X, y = load_datasets()[name]; rows = []
    for seed in range(NSEEDS):
        Xtr0, Xte, ytr0, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
        for fr in FR:
            if fr < 1.0:
                Xtr, _, ytr, _ = train_test_split(Xtr0, ytr0, train_size=fr, stratify=ytr0, random_state=seed)
            else:
                Xtr, ytr = Xtr0, ytr0
            A, B = preprocess(Xtr, Xte)
            for mn, mk in make_models(seed, 0.05, 200).items():
                r = evaluate(mk, A, ytr, B, yte)
                rows.append(dict(ds=name, frac=fr, n_train=len(ytr), seed=seed, model=mn, **r))
        print(name, seed, flush=True)
    return rows
if __name__ == "__main__":
    with Pool(NJOBS) as p: out = p.map(task, ["WDBC", "Ionosphere", "Synthetic", "Transfusion"], chunksize=1)
    pd.DataFrame([r for o in out for r in o]).to_csv("h4_results.csv", index=False); print("DONE")
