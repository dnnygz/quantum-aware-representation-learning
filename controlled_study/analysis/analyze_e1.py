import numpy as np, pandas as pd
from scipy.stats import wilcoxon
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
df = pd.read_csv("e1_results.csv")
KEEP = ["WDBC", "Ionosphere", "Synthetic", "Transfusion"]   # Sonar was run but is excluded from the final design (redundant with Ionosphere, n=208)
df = df[df.ds.isin(KEEP)]
df["gap"] = df.auc_tr - df.auc_te
real = df[df.cond == "real"]
MODELS = ["VQC", "LR", "SVM", "RF", "XGB", "MLP"]
DS = KEEP
# ---- summary table
S = real.groupby(["ds", "model"]).agg(auc_tr=("auc_tr", "mean"), auc_te=("auc_te", "mean"),
        sd_te=("auc_te", "std"), gap=("gap", "mean"), sd_gap=("gap", "std"), brier=("brier_te", "mean")).round(3)
S = S.reindex(pd.MultiIndex.from_product([DS, MODELS]))
S.to_csv("e1_summary.csv"); print(S.to_string())
# ---- label randomization: train AUC on shuffled labels (memorization) and test AUC
sh = df[df.cond == "shuffled"].groupby(["ds", "model"]).agg(tr=("auc_tr", "mean"), te=("auc_te", "mean")).round(3)
sh = sh.reindex(pd.MultiIndex.from_product([DS, MODELS])); sh.to_csv("e1_shuffled.csv"); print("\nSHUFFLED LABELS\n", sh.to_string())
# ---- paired comparisons VQC vs each classical (same seeds/splits)
rng = np.random.RandomState(0)
def boot_ci(d, B=10000):
    idx = rng.randint(0, len(d), (B, len(d))); m = d[idx].mean(1)
    return np.percentile(m, [2.5, 97.5])
rows = []
for ds in DS:
    P = real[real.ds == ds].pivot(index="seed", columns="model", values=["auc_te", "gap"])
    for mn in MODELS[1:]:
        for metric in ["auc_te", "gap"]:
            d = (P[metric]["VQC"] - P[metric][mn]).values
            lo, hi = boot_ci(d)
            try: p = wilcoxon(d).pvalue
            except ValueError: p = 1.0
            rows.append(dict(ds=ds, vs=mn, metric=metric, mean_diff=d.mean(), ci_lo=lo, ci_hi=hi, p_wilcoxon=p))
T = pd.DataFrame(rows)
# Holm correction within each metric across all dataset x model comparisons
T["p_holm"] = np.nan
for metric in ["auc_te", "gap"]:
    m = T.metric == metric; p = T.loc[m, "p_wilcoxon"].values; o = np.argsort(p); k = len(p)
    adj = np.empty(k); run = 0
    for r, i in enumerate(o):
        run = max(run, min(1, (k - r) * p[i])); adj[i] = run
    T.loc[m, "p_holm"] = adj
T["excl0"] = (T.ci_lo > 0) | (T.ci_hi < 0)
T.round(4).to_csv("e1_paired.csv", index=False)
pd.set_option("display.width", 200); print("\nPAIRED (VQC minus model)\n", T.round(3).to_string())
# ---- figure: test AUC (filled) and train AUC (hollow) per model, one panel per dataset
fig, axs = plt.subplots(1, len(DS), figsize=(7.2, 2.6), sharey=True)
for ax, ds in zip(axs, DS):
    s = S.loc[ds]
    y = np.arange(len(MODELS))[::-1]
    for yi, mn in zip(y, MODELS):
        ax.plot([s.loc[mn, "auc_te"], s.loc[mn, "auc_tr"]], [yi, yi], color="#8a8a86", lw=1.2, zorder=1)
        ax.errorbar(s.loc[mn, "auc_te"], yi, xerr=s.loc[mn, "sd_te"], fmt="none", ecolor="#2a78d6", elinewidth=1, capsize=0, zorder=2)
    ax.scatter(s.loc[MODELS, "auc_tr"], y, s=26, facecolor="white", edgecolor="#52514e", lw=1.2, zorder=3, label="Train AUC")
    ax.scatter(s.loc[MODELS, "auc_te"], y, s=26, color="#2a78d6", zorder=4, label="Test AUC")
    ax.set_title(ds, fontsize=9); ax.set_yticks(y); ax.set_yticklabels(MODELS, fontsize=8)
    ax.tick_params(axis="x", labelsize=7); ax.grid(axis="x", color="#e5e4e0", lw=0.6); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    lo = min(s.auc_te.min() - 0.05, 0.7); ax.set_xlim(max(0.6, lo), 1.01)
axs[0].get_yticklabels()[-1].set_fontweight("bold") if False else None
fig.supxlabel("AUC (mean over 30 seeds; bars = SD of test AUC)", fontsize=8, y=0.02)
h, l = axs[-1].get_legend_handles_labels(); fig.legend(h, l, loc="upper right", ncol=2, fontsize=7, frameon=False, bbox_to_anchor=(0.99, 1.0))
fig.tight_layout(rect=(0, 0.04, 1, 0.94)); fig.savefig("fig_e1.png", dpi=300); fig.savefig("fig_e1.pdf")
print("saved fig")
