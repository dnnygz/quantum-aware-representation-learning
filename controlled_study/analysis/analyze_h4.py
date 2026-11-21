import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
d = pd.read_csv("h4_results.csv"); d["gap"] = d.auc_tr - d.auc_te
DS = ["WDBC", "Ionosphere", "Synthetic", "Transfusion"]; M = ["VQC", "LR", "SVM", "RF", "XGB", "MLP"]; FR = [0.1, 0.25, 0.5, 0.75, 1.0]
S = d.groupby(["ds", "model", "frac"]).agg(auc_te=("auc_te", "mean"), gap=("gap", "mean"), n=("n_train", "mean")).round(4); S.to_csv("h4_summary.csv")
rng = np.random.RandomState(0)
def ci(x, B=10000):
    idx = rng.randint(0, len(x), (B, len(x))); return np.percentile(x[idx].mean(1), [2.5, 97.5])
rows = []
for ds in DS:
    P = d[d.ds == ds].pivot_table(index="seed", columns=["model", "frac"], values=["auc_te", "gap"])
    for fr in FR:
        for m in ("auc_te", "gap"):
            for other in ("MLP", "LR"):
                x = (P[m][("VQC", fr)] - P[m][(other, fr)]).values; lo, hi = ci(x)
                rows.append(dict(ds=ds, frac=fr, metric=m, cmp=f"VQC - {other}", mean_diff=x.mean(), lo=lo, hi=hi, excl0=(lo > 0) or (hi < 0)))
    for m in ("auc_te", "gap"):
        for mod in M:
            x = (P[m][(mod, 1.0)] - P[m][(mod, 0.1)]).values; lo, hi = ci(x)
            rows.append(dict(ds=ds, frac="100% - 10%", metric=m, cmp=mod, mean_diff=x.mean(), lo=lo, hi=hi, excl0=(lo > 0) or (hi < 0)))
T = pd.DataFrame(rows).round(4); T.to_csv("h4_paired.csv", index=False)
COL = {"VQC": "#2a78d6", "LR": "#eb6834", "SVM": "#1baf7a", "RF": "#eda100", "XGB": "#e87ba4", "MLP": "#52514e"}; MK = {"VQC": "o", "LR": "s", "SVM": "^", "RF": "D", "XGB": "v", "MLP": "x"}
fig, axs = plt.subplots(2, 4, figsize=(8.4, 4.3), sharex=True)
for j, ds in enumerate(DS):
    for i, (m, lab) in enumerate([("auc_te", "Test AUC"), ("gap", "Generalization gap")]):
        ax = axs[i, j]
        for mod in M:
            v = S.loc[(ds, mod)][m].reindex(FR)
            ax.plot([f * 100 for f in FR], v.values, color=COL[mod], marker=MK[mod], ms=3.5, lw=1.5 if mod == "VQC" else 1.1, ls="--" if mod == "MLP" else "-", label=mod, zorder=3 if mod == "VQC" else 2)
        ax.set_xscale("log"); ax.minorticks_off(); ax.set_xticks([f * 100 for f in FR]); ax.set_xticklabels([f"{int(f*100)}" for f in FR], fontsize=6)
        ax.tick_params(axis="y", labelsize=7); ax.grid(color="#e5e4e0", lw=0.6); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        if i == 0: ax.set_title(ds, fontsize=9)
        if j == 0: ax.set_ylabel(lab, fontsize=8)
        if i == 1: ax.set_xlabel("Training data used (%)", fontsize=8)
h, l = axs[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=6, fontsize=7, frameon=False)
fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig("fig_h4.png", dpi=300); fig.savefig("fig_h4.pdf")
pd.set_option("display.width", 200)
print(T[(T.cmp == "VQC - MLP")].pivot_table(index=["ds", "metric"], columns="frac", values="mean_diff").round(3).to_string())
print(T[(T.cmp == "VQC - MLP") & T.excl0][["ds", "frac", "metric", "mean_diff", "lo", "hi"]].round(3).to_string())
print(T[T.frac == "100% - 10%"].pivot_table(index=["ds", "metric"], columns="cmp", values="mean_diff").round(3)[M].to_string())
