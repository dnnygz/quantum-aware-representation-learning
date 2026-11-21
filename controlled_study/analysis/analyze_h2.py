import sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
src = sys.argv[1] if len(sys.argv) > 1 else ""   # "" = final files, "_partial" = partial files
A = pd.read_csv(f"h2A{'_partial' if src else '_results'}.csv"); B = pd.read_csv(f"h2B{'_partial' if src else '_results'}.csv")
for d in (A, B): d["gap"] = d.auc_tr - d.auc_te
E1 = pd.read_csv("e1_results.csv"); E1 = E1[E1.cond == "real"]; E1["gap"] = E1.auc_tr - E1.auc_te
rng = np.random.RandomState(0)
def ci(x, Bn=10000):
    idx = rng.randint(0, len(x), (Bn, len(x))); return np.percentile(x[idx].mean(1), [2.5, 97.5])
def paired(P, m, a, b):
    x = (P[m][a] - P[m][b]).dropna().values; lo, hi = ci(x); return dict(mean_diff=x.mean(), lo=lo, hi=hi, excl0=(lo > 0) or (hi < 0), n=len(x))
# sanity: VQC-base from the new simulator equals the E1 VQC (same seeds)
ev = E1[(E1.model == "VQC") & E1.ds.isin(A.ds.unique())].set_index(["ds", "seed"]).auc_te
nb = A[A.model == "VQC-base"].set_index(["ds", "seed"]).auc_te
print("max |VQC-base(new) - VQC(E1)| test AUC:", float((ev.reindex(nb.index) - nb).abs().max()))
# summaries
SA = A.groupby(["ds", "model"]).agg(auc_tr=("auc_tr", "mean"), auc_te=("auc_te", "mean"), gap=("gap", "mean"), n=("seed", "count")).round(4)
SB = B.groupby(["ds", "model"]).agg(auc_tr=("auc_tr", "mean"), auc_te=("auc_te", "mean"), gap=("gap", "mean"), n=("seed", "count")).round(4)
SA.to_csv("h2A_summary.csv"); SB.to_csv("h2B_summary.csv")
rows = []
VA = ["reup2", "scale05", "scale2", "L4reup"]
for ds in A.ds.unique():
    P = A[A.ds == ds].pivot_table(index="seed", columns="model", values=["auc_te", "gap"])
    mlp = E1[(E1.ds == ds) & (E1.model == "MLP")].set_index("seed")
    for m in ("auc_te", "gap"):
        for v in VA:
            rows.append(dict(part="A", ds=ds, metric=m, cmp=f"{v} - base", **paired(P, m, "VQC-" + v, "VQC-base")))
        rows.append(dict(part="A", ds=ds, metric=m, cmp="L4reup - L4", **paired(P, m, "VQC-L4reup", "VQC-L4")))
        for v in ["base", "reup2", "L4reup"]:
            Q = P.copy(); Q[(m, "MLP")] = mlp[m]
            rows.append(dict(part="A", ds=ds, metric=m, cmp=f"{v} - MLP", **paired(Q, m, "VQC-" + v, "MLP")))
for ds in B.ds.unique():
    P = B[B.ds == ds].pivot_table(index="seed", columns="model", values=["auc_te", "gap"])
    for m in ("auc_te", "gap"):
        for v in ["reup2", "scale2", "L4", "L4reup"]:
            rows.append(dict(part="B", ds=ds, metric=m, cmp=f"{v} - base", **paired(P, m, "VQC-" + v, "VQC-base")))
        rows.append(dict(part="B", ds=ds, metric=m, cmp="L4reup - L4", **paired(P, m, "VQC-L4reup", "VQC-L4")))
        for v in ["base", "reup2"]:
            for o in ["MLP", "SVM", "RF"]:
                rows.append(dict(part="B", ds=ds, metric=m, cmp=f"{v} - {o}", **paired(P, m, "VQC-" + v, o)))
T = pd.DataFrame(rows).round(4); T.to_csv("h2_paired.csv", index=False)
# figure (part B)
COL = {"VQC-base": "#2a78d6", "VQC-reup2": "#7b3fb5", "MLP": "#52514e", "SVM": "#1baf7a", "RF": "#eda100"}
MK = {"VQC-base": "o", "VQC-reup2": "s", "MLP": "x", "SVM": "^", "RF": "D"}
LAB = {"VQC-base": "VQC (one upload)", "VQC-reup2": "VQC (re-upload)", "MLP": "MLP", "SVM": "SVM", "RF": "RF"}
fig, axs = plt.subplots(2, 2, figsize=(5.2, 4.4), sharex=True)
for j, (fam, ft) in enumerate([("add", r"Additive: $\sin kx_1+\sin kx_2$"), ("prod", r"Product: $\sin kx_1\,\sin kx_2$")]):
    for i, (m, lab) in enumerate([("auc_te", "Test AUC"), ("gap", "Generalization gap")]):
        ax = axs[i, j]
        for mod in COL:
            v = [SB.loc[(f"{fam}{k}", mod)][m] if (f"{fam}{k}", mod) in SB.index else np.nan for k in (1, 2, 3)]
            ax.plot([1, 2, 3], v, color=COL[mod], marker=MK[mod], ms=4, lw=1.6 if mod.startswith("VQC") else 1.1, ls="--" if mod == "MLP" else "-", label=LAB[mod])
        ax.set_xticks([1, 2, 3]); ax.tick_params(labelsize=7); ax.grid(color="#e5e4e0", lw=0.6); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        if i == 0: ax.set_title(ft, fontsize=8)
        if j == 0: ax.set_ylabel(lab, fontsize=8)
        if i == 1: ax.set_xlabel("Frequency $k$", fontsize=8)
h, l = axs[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=3, fontsize=7, frameon=False)
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig("fig_h2.png", dpi=300); fig.savefig("fig_h2.pdf")
pd.set_option("display.width", 220)
print(SB.reset_index().pivot_table(index="ds", columns="model", values="auc_te").round(3).to_string())
print(SB.reset_index().pivot_table(index="ds", columns="model", values="gap").round(3).to_string())
print(SA.reset_index().pivot_table(index="ds", columns="model", values="auc_te").round(3).to_string())
print(SA.reset_index().pivot_table(index="ds", columns="model", values="gap").round(3).to_string())
