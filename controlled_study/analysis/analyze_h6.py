import numpy as np, pandas as pd
d = pd.read_csv("h6_results.csv"); d["gap"] = d.auc_tr - d.auc_te
E = pd.read_csv("e1_results.csv"); E = E[E.cond == "real"].copy(); E["gap"] = E.auc_tr - E.auc_te
E = E[E.model.isin(["VQC", "MLP", "LR"]) & E.ds.isin(d.ds.unique())]
E["model"] = E.model + "-hard"                                   # hard-label PCA-4 baselines from E1 (same seeds and splits)
allr = pd.concat([d[["ds", "seed", "model", "auc_tr", "auc_te", "gap"]], E[["ds", "seed", "model", "auc_tr", "auc_te", "gap"]]])
S = allr.groupby(["ds", "model"]).agg(auc_tr=("auc_tr", "mean"), auc_te=("auc_te", "mean"), gap=("gap", "mean"), n=("seed", "count")).round(4); S.to_csv("h6_summary.csv")
rng = np.random.RandomState(0)
def ci(x, B=10000):
    idx = rng.randint(0, len(x), (B, len(x))); return np.percentile(x[idx].mean(1), [2.5, 97.5])
CMP = [("VQC-KD0.5", "VQC-hard"), ("VQC-KD1.0", "VQC-hard"), ("MLP-KD0.5", "MLP-hard"), ("MLP-KD1.0", "MLP-hard"), ("VQC-KD1.0", "MLP-KD1.0"),
       ("VQC-KD1.0", "Teacher-SVM"), ("MLP-KD1.0", "Teacher-SVM"),
       ("VQC-on-enc", "VQC-hard"), ("VQC-on-enc-half", "VQC-hard"), ("LR-on-enc", "LR-hard"), ("VQC-on-enc", "LR-on-enc"), ("VQC-on-enc-half", "LR-on-enc"),
       ("LR-on-enc", "Teacher-MLPenc"), ("VQC-on-enc-half", "Teacher-MLPenc"), ("VQC-on-enc", "Teacher-MLPenc")]
rows = []
for ds in d.ds.unique():
    P = allr[allr.ds == ds].pivot_table(index="seed", columns="model", values=["auc_te", "gap", "auc_tr"])
    for a, b in CMP:
        if ("auc_te", a) not in P.columns or ("auc_te", b) not in P.columns: continue
        for m in ("auc_te", "gap"):
            x = (P[m][a] - P[m][b]).dropna().values; lo, hi = ci(x)
            rows.append(dict(ds=ds, metric=m, cmp=f"{a} - {b}", mean_diff=x.mean(), lo=lo, hi=hi, excl0=(lo > 0) or (hi < 0), n=len(x)))
T = pd.DataFrame(rows).round(4); T.to_csv("h6_paired.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
for m in ("auc_te", "gap", "auc_tr"):
    print(m); print(S.reset_index().pivot_table(index="ds", columns="model", values=m).round(3).to_string())
t = T.assign(s=T.apply(lambda r: f"{r.mean_diff:+.3f}{'*' if r.excl0 else ''} [{r.lo:+.3f},{r.hi:+.3f}]", axis=1))
for m in ("auc_te", "gap"):
    print(m); print(t[t.metric == m].pivot(index="cmp", columns="ds", values="s").to_string())
