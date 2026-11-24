import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
df = pd.read_csv("h5_results.csv"); df["gap"] = df.auc_tr - df.auc_te
DS = ["WDBC", "Ionosphere", "Synthetic", "Transfusion"]
SET = ["base(lr.05,init.1)", "lr.01", "lr.2", "init~N(0,1)", "init~U(-pi,pi)", "MLP(lr.05)"]
EP = sorted(df.epochs.unique())
S = df.groupby(["ds", "setting", "epochs"]).agg(auc_te=("auc_te", "mean"), gap=("gap", "mean"), bce_tr=("bce_tr", "mean"), sd_te=("auc_te", "std")).round(3)
S.to_csv("h5_summary.csv")
pd.set_option("display.width", 220)
print(S["auc_te"].unstack("epochs").reindex(pd.MultiIndex.from_product([DS, SET])).to_string())
print("\nGAP\n", S["gap"].unstack("epochs").reindex(pd.MultiIndex.from_product([DS, SET])).to_string())
print("\nTRAIN BCE\n", S["bce_tr"].unstack("epochs").reindex(pd.MultiIndex.from_product([DS, SET])).to_string())
rng = np.random.RandomState(0)
def ci(d, B=10000):
    idx = rng.randint(0, len(d), (B, len(d))); return np.percentile(d[idx].mean(1), [2.5, 97.5])
rows = []
for ds in DS:
    P = df[df.ds == ds].pivot_table(index="seed", columns=["setting", "epochs"], values=["auc_te", "gap"])
    for ep in [200, 800]:
        for sname in SET[:5]:
            for metric in ["auc_te", "gap"]:
                d = (P[metric][(sname, ep)] - P[metric][("MLP(lr.05)", ep)]).values
                lo, hi = ci(d); rows.append(dict(ds=ds, cmp=f"{sname} - MLP", epochs=ep, metric=metric, mean_diff=d.mean(), lo=lo, hi=hi, excl0=(lo > 0) or (hi < 0)))
    for metric in ["auc_te", "gap"]:
        d = (P[metric][("base(lr.05,init.1)", 800)] - P[metric][("base(lr.05,init.1)", 200)]).values
        lo, hi = ci(d); rows.append(dict(ds=ds, cmp="base 800ep - base 200ep", epochs=800, metric=metric, mean_diff=d.mean(), lo=lo, hi=hi, excl0=(lo > 0) or (hi < 0)))
T = pd.DataFrame(rows).round(4); T.to_csv("h5_paired.csv", index=False)
print("\nPAIRED\n", T.round(3).to_string())
COL = {"base(lr.05,init.1)": "#2a78d6", "lr.01": "#eb6834", "lr.2": "#1baf7a", "init~N(0,1)": "#eda100", "init~U(-pi,pi)": "#e87ba4", "MLP(lr.05)": "#52514e"}
MK = {"base(lr.05,init.1)": "o", "lr.01": "s", "lr.2": "^", "init~N(0,1)": "D", "init~U(-pi,pi)": "v", "MLP(lr.05)": "x"}
fig, axs = plt.subplots(2, 4, figsize=(8.4, 4.3), sharex=True)
for j, ds in enumerate(DS):
    for i, (m, lab) in enumerate([("auc_te", "Test AUC"), ("gap", "Generalization gap")]):
        ax = axs[i, j]
        for sname in SET:
            v = S.loc[(ds, sname), m].reindex(EP)
            ax.plot(EP, v.values, color=COL[sname], marker=MK[sname], ms=3.5, lw=1.4 if sname != "MLP(lr.05)" else 1.2,
                    ls="--" if sname == "MLP(lr.05)" else "-", label=sname)
        ax.set_xscale("log"); ax.minorticks_off(); ax.set_xticks(EP); ax.set_xticklabels([str(e) for e in EP], fontsize=6)
        ax.tick_params(axis="y", labelsize=7); ax.grid(color="#e5e4e0", lw=0.6); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        if i == 0: ax.set_title(ds, fontsize=9)
        if j == 0: ax.set_ylabel(lab, fontsize=8)
        if i == 1: ax.set_xlabel("Epochs", fontsize=8)
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=6, fontsize=6.5, frameon=False)
fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig("fig_h5.png", dpi=300); fig.savefig("fig_h5.pdf"); print("saved")
