import numpy as np, pandas as pd
d = pd.read_csv("h1_results.csv"); d["gap"] = d.auc_tr - d.auc_te
DS = ["WDBC", "Ionosphere", "Synthetic"]; M = ["VQC", "LR", "SVM", "RF", "XGB", "MLP"]
S = d.groupby(["ds", "model", "input"]).agg(auc_te=("auc_te", "mean"), sd_te=("auc_te", "std"), gap=("gap", "mean")).round(4); S.to_csv("h1_summary.csv")
rng = np.random.RandomState(0)
def ci(x, B=10000):
    idx = rng.randint(0, len(x), (B, len(x))); return np.percentile(x[idx].mean(1), [2.5, 97.5])
rows = []
def add(ds, label, m, x):
    lo, hi = ci(x); rows.append(dict(ds=ds, cmp=label, metric=m, mean_diff=x.mean(), lo=lo, hi=hi, excl0=(lo > 0) or (hi < 0)))
for ds in DS:
    P = d[d.ds == ds].pivot_table(index="seed", columns=["model", "input"], values=["auc_te", "gap"])
    for m in ("auc_te", "gap"):
        for mod in M[1:]:
            add(ds, f"{mod}: all - pca4", m, (P[m][(mod, "all")] - P[m][(mod, "pca4")]).values)
            add(ds, f"{mod}: anova4 - pca4", m, (P[m][(mod, "anova4")] - P[m][(mod, "pca4")]).values)
        add(ds, "VQC: anova4 - pca4", m, (P[m][("VQC", "anova4")] - P[m][("VQC", "pca4")]).values)
        for inp in ("pca4", "anova4"):
            for mod in M[1:]:
                add(ds, f"VQC - {mod} [{inp}]", m, (P[m][("VQC", inp)] - P[m][(mod, inp)]).values)
        for mod in ("LR", "SVM", "RF", "XGB", "MLP"):
            add(ds, f"VQC[pca4] - {mod}[all]", m, (P[m][("VQC", "pca4")] - P[m][(mod, "all")]).values)
T = pd.DataFrame(rows).round(4); T.to_csv("h1_paired.csv", index=False)
pd.set_option("display.width", 220)
print(T[T.metric == "auc_te"].pivot_table(index="cmp", columns="ds", values="mean_diff").round(3).to_string())
print(T[(T.metric == "auc_te") & T.cmp.str.contains("VQC")].query("excl0")[["ds", "cmp", "mean_diff", "lo", "hi"]].round(3).to_string())
print(T[(T.metric == "gap") & T.cmp.str.contains("MLP|anova4 - pca4")][["ds", "cmp", "mean_diff", "lo", "hi", "excl0"]].round(3).to_string())
