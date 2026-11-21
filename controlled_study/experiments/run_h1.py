"""H1 (representation): the same models on (a) PCA-4, (b) ANOVA top-4 features, (c) all features (classical models only).
Datasets with more than 4 native features; Transfusion is the native 4-feature case already covered by E1."""
from multiprocessing import Pool
from common import *
NSEEDS = NSEEDS_ENV
def task(name):
    X, y = load_datasets()[name]; rows = []
    for seed in range(NSEEDS):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
        for kind in ("pca4", "anova4", "all"):
            A, B = preprocess_variant(Xtr, Xte, ytr, kind)
            for mn, mk in make_models(seed, 0.05, 200).items():
                if kind == "all" and mn == "VQC": continue
                r = evaluate(mk, A, ytr, B, yte)
                rows.append(dict(ds=name, input=kind, seed=seed, model=mn, **r))
        print(name, seed, flush=True)
    return rows
if __name__ == "__main__":
    with Pool(NJOBS) as p: out = p.map(task, ["WDBC", "Ionosphere", "Synthetic"], chunksize=1)
    pd.DataFrame([r for o in out for r in o]).to_csv("h1_results.csv", index=False); print("DONE")
