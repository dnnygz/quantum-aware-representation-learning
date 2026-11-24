"""Pilot: choose lr/epochs for VQC and MLP on a VALIDATION split carved from the training part,
using pilot seeds (1000-1004) disjoint from the evaluation seeds (0-29). Test split never touched."""
import itertools, sys, time
from common import *
data = load_datasets()
rows = []
t0 = time.time()
for name, (X, y) in data.items():
    for seed in range(1000, 1005):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
        Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.3, stratify=ytr, random_state=seed)
        A, V = preprocess(Xa, Xv)
        for lr, ep in itertools.product([0.02, 0.05, 0.1], [100, 200]):
            for mn in ["VQC", "MLP"]:
                r = evaluate(make_models(seed, lr, ep)[mn], A, ya, V, yv)
                rows.append(dict(ds=name, seed=seed, lr=lr, epochs=ep, model=mn, **r))
    print(name, round(time.time() - t0), "s", flush=True)
df = pd.DataFrame(rows); df.to_csv("pilot.csv", index=False)
g = df.groupby(["model", "lr", "epochs"]).agg(val_auc=("auc_te", "mean"), tr=("auc_tr", "mean"), val_brier=("brier_te", "mean")).round(4)
print(g.to_string())
print(df.groupby(["ds", "model", "lr", "epochs"]).auc_te.mean().unstack(["model"]).round(3).to_string())
