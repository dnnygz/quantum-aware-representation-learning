import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
df = pd.read_csv("h3_results.csv"); df["gap"] = df.auc_tr - df.auc_te
DS = ["WDBC", "Ionosphere", "Synthetic", "Transfusion"]
V = ["L1", "L2", "L4", "L2-noent", "L2-nohead"]; MLPS = ["MLP h2 (13p)", "MLP h3 (19p)", "MLP h6 (37p)"]
real = df[df.cond == "real"]; sh = df[df.cond == "shuffled"]
pd.set_option("display.width", 220)
for ep in (200, 400):
    S = real[real.epochs == ep].groupby(["ds", "model"]).agg(auc_te=("auc_te", "mean"), gap=("gap", "mean"), bce=("bce_tr", "mean"), brier=("brier_te", "mean")).round(3)
    S.to_csv(f"h3_summary_ep{ep}.csv")
    print(f"\n=== epochs {ep}: test AUC / gap")
    print(S[["auc_te", "gap"]].unstack("model").reindex(DS)[[("auc_te", m) for m in V + MLPS] + [("gap", m) for m in V + MLPS]].round(3).to_string())
S = sh.groupby(["ds", "model"]).agg(tr=("auc_tr", "mean"), te=("auc_te", "mean")).round(3); S.to_csv("h3_shuffled.csv")
print("\n=== shuffled labels, train AUC (200 ep)"); print(S["tr"].unstack("model").reindex(DS)[V + MLPS].to_string())
rng = np.random.RandomState(0)
def ci(d, B=10000):
    idx = rng.randint(0, len(d), (B, len(d))); return np.percentile(d[idx].mean(1), [2.5, 97.5])
rows = []
def add(ds, label, ep, m, d):
    lo, hi = ci(d); rows.append(dict(ds=ds, cmp=label, epochs=ep, metric=m, mean_diff=d.mean(), lo=lo, hi=hi, excl0=(lo > 0) or (hi < 0)))
pairs = [("L1", "L2"), ("L4", "L2"), ("L2-noent", "L2"), ("L2-nohead", "L2"), ("L1", "MLP h2 (13p)"), ("L2", "MLP h3 (19p)"), ("L4", "MLP h6 (37p)")]
for ds in DS:
    for ep in (200, 400):
        P = real[(real.ds == ds) & (real.epochs == ep)].pivot_table(index="seed", columns="model", values=["auc_te", "gap"])
        for a, b in pairs:
            for m in ("auc_te", "gap"): add(ds, f"{a} - {b}", ep, m, (P[m][a] - P[m][b]).values)
T = pd.DataFrame(rows).round(4); T.to_csv("h3_paired.csv", index=False)
print("\n=== PAIRED (epochs 200)"); print(T[T.epochs == 200].round(3).to_string())
print("\n=== PAIRED (epochs 400, only rows whose CI excludes 0 or matched-MLP comparisons)")
print(T[(T.epochs == 400) & (T.cmp.str.contains("MLP") | T.excl0)].round(3).to_string())
