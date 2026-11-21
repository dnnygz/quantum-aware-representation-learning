import pandas as pd, numpy as np
df = pd.read_csv("h5_results.csv"); df["gap"] = df.auc_tr - df.auc_te
DS = ["WDBC", "Ionosphere", "Synthetic", "Transfusion"]; EP = [25, 50, 100, 200, 400, 800]
rng = np.random.RandomState(1)
f3 = lambda x: f"{x:.3f}".replace("-", "$-$")
rows = []
for ds in DS:
    P = df[df.ds == ds].pivot_table(index="seed", columns=["setting", "epochs"], values=["auc_te", "gap"])
    for m, lab in [("auc_te", "$\\Delta$ test AUC"), ("gap", "$\\Delta$ gap")]:
        cells = []
        for ep in EP:
            d = (P[m][("base(lr.05,init.1)", ep)] - P[m][("MLP(lr.05)", ep)]).values
            idx = rng.randint(0, len(d), (10000, len(d))); lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
            cells.append(f3(d.mean()) + ("$^{\\dagger}$" if (lo > 0 or hi < 0) else ""))
        rows.append(f"{ds if m == 'auc_te' else ''} & {lab} & " + " & ".join(cells) + " \\\\")
tab = ["\\begin{table*}[t]", "\\centering",
       "\\caption{Paired difference between the VQC (base setting) and the MLP, trained for the same number of epochs (columns), mean over 30 seeds. A dagger marks a 95\\% bootstrap interval that excludes zero (no multiple-comparison correction).}",
       "\\label{tab:h5}", "\\footnotesize", "\\setlength{\\tabcolsep}{2.5pt}", "\\begin{tabular}{llcccccc}", "\\hline",
       "Dataset & Metric & " + " & ".join(str(e) for e in EP) + " \\\\", "\\hline"] + rows + ["\\hline", "\\end{tabular}", "\\end{table*}"]
open("tab_h5.tex", "w").write("\n".join(tab) + "\n"); print("\n".join(tab))
