import numpy as np, pandas as pd
f3 = lambda x: f"{x:.3f}".replace("-", "$-$")
# ================= H4 =================
d4 = pd.read_csv("h4_results.csv"); d4["gap"] = d4.auc_tr - d4.auc_te
DS4 = ["WDBC", "Ionosphere", "Synthetic", "Transfusion"]; FR = [0.1, 0.25, 0.5, 0.75, 1.0]
rng = np.random.RandomState(2); rows = []
for ds in DS4:
    P = d4[d4.ds == ds].pivot_table(index="seed", columns=["model", "frac"], values=["auc_te", "gap"])
    for m, lab in [("auc_te", "$\\Delta$ test AUC"), ("gap", "$\\Delta$ gap")]:
        cells = []
        for fr in FR:
            x = (P[m][("VQC", fr)] - P[m][("MLP", fr)]).values
            idx = rng.randint(0, len(x), (10000, len(x))); lo, hi = np.percentile(x[idx].mean(1), [2.5, 97.5])
            cells.append(f3(x.mean()) + ("$^{\\dagger}$" if (lo > 0 or hi < 0) else ""))
        rows.append(f"{ds if m == 'auc_te' else ''} & {lab} & " + " & ".join(cells) + " \\\\")
t4 = ["\\begin{table*}[t]", "\\centering",
      "\\caption{Paired difference between the VQC and the MLP when trained on a fraction of the training split (columns, percentage of the training data), mean over 30 seeds. At 10\\% the training sets had 24 to 52 samples, and at 100\\% they had 245 to 523. A dagger marks a 95\\% bootstrap interval that excludes zero (no multiple-comparison correction).}",
      "\\label{tab:h4}", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{llccccc}", "\\hline",
      "Dataset & Metric & 10\\% & 25\\% & 50\\% & 75\\% & 100\\% \\\\", "\\hline"] + rows + ["\\hline", "\\end{tabular}", "\\end{table*}"]
sec4 = r"""\subsubsection{Training-set size (H4)}

To test whether the gap shrinks as more data become available, we trained all models on 10\%, 25\%, 50\%, 75\%, and 100\% of the training split, drawn at random for each seed with stratification, and evaluated them on the same test split as before. The training budget stayed fixed (learning rate 0.05, 200 epochs), and the preprocessing was refitted on each subset. Fig.~\ref{fig:h4} shows the learning curves and Table~\ref{tab:h4} the paired difference between the VQC and the MLP.

\begin{figure*}[t]
\centering
\includegraphics[width=0.9\textwidth]{fig_h4.png}
\caption{Test AUC (top) and generalization gap (bottom) as a function of the percentage of the training split used, for the VQC and the classical models, averaged over 30 seeds.}
\label{fig:h4}
\end{figure*}

\textit{The gap shrinks with more data, and more for the VQC.} The gap of the VQC decreased from 0.070 to 0.007 on WDBC, from 0.234 to 0.054 on Ionosphere, from 0.397 to 0.052 on the synthetic data, and from 0.278 to 0.019 on Blood Transfusion between 10\% and 100\% of the data, and its test AUC increased by 0.056, 0.099, 0.213, and 0.109. The MLP showed the same trend with smaller changes (gap decreases of 0.016, 0.124, 0.283, and 0.156; test AUC increases of 0.013, 0.093, 0.175, and 0.076). Logistic regression had the smallest gap at every size on all four datasets, whereas the Random Forest and XGBoost still had gaps between 0.014 and 0.283 with the full training split.

\textit{At small sizes, the VQC generalized worse than the MLP.} With 10\% of the data, the gap of the VQC was larger than that of the MLP on all four datasets ($+0.049$ on WDBC, $+0.061$ on Ionosphere, $+0.071$ on the synthetic data, and $+0.118$ on Blood Transfusion, with intervals above zero), and its test AUC was lower ($-0.049$, $-0.039$, $-0.043$, and $-0.038$; the interval on Blood Transfusion ended at $-0.000$). The difference in gap shrank with more data, but not in the same way on every dataset (Table~\ref{tab:h4}). On WDBC and Blood Transfusion it decreased steadily (to $+0.003$ and $+0.015$ with the full split). On the synthetic data it was close to zero from 50\% of the data on ($0.000$ at 50\% and $+0.009$ with the full split, with intervals including zero), and on Ionosphere it was not monotonic ($+0.008$ at 50\%, $+0.030$ at 75\%, and $+0.006$ with the full split). With the full split, the test AUC of the VQC was still lower than that of the MLP on WDBC ($-0.006$) and on Ionosphere ($-0.033$), and similar on the other two datasets.

""" + "\n".join(t4) + "\n"
open("sec42_h4.tex", "w").write(sec4)
# ================= H1 =================
d1 = pd.read_csv("h1_results.csv"); d1["gap"] = d1.auc_tr - d1.auc_te
DS1 = ["WDBC", "Ionosphere", "Synthetic"]; M = ["VQC", "LR", "SVM", "RF", "XGB", "MLP"]
S = d1.groupby(["ds", "model", "input"])[["auc_te", "gap"]].mean()
NAME = {"XGB": "XGBoost"}
t1 = ["\\begin{table*}[t]", "\\centering",
      "\\caption{Representation (H1): test AUC and generalization gap (means over 30 seeds) with four inputs obtained by PCA, with the four features with the highest ANOVA F-value, and with all the features (classical models only; the VQC always receives four inputs).}",
      "\\label{tab:h1}", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}", "\\begin{tabular}{llcccccc}", "\\hline",
      " & & \\multicolumn{3}{c}{Test AUC} & \\multicolumn{3}{c}{Gap} \\\\",
      "Dataset & Model & PCA-4 & ANOVA-4 & All & PCA-4 & ANOVA-4 & All \\\\", "\\hline"]
for ds in DS1:
    for i, m in enumerate(M):
        def c(k, inp):
            return "--" if (m == "VQC" and inp == "all") else f3(S.loc[(ds, m, inp)][k])
        t1.append(f"{ds if i == 0 else ''} & {NAME.get(m, m)} & " + " & ".join(c("auc_te", n) for n in ("pca4", "anova4", "all")) + " & " + " & ".join(c("gap", n) for n in ("pca4", "anova4", "all")) + " \\\\")
    if ds != DS1[-1]: t1.append("\\hline")
t1 += ["\\hline", "\\end{tabular}", "\\end{table*}"]
sec1 = r"""\subsubsection{Input representation (H1)}

To test how much of the difference between the VQC and the classical models comes from reducing the input to four variables, we ran the same models on three inputs for the datasets with more than four native features: four principal components (the base setting), the four features with the highest ANOVA F-value on the training part, and, for the classical models only, all the features (Table~\ref{tab:h1}). Blood Transfusion, which has four native variables, was already covered by the base experiment. Because the three inputs were not tuned for the VQC, these results show the sensitivity of the VQC to the representation rather than its best possible performance.

\textit{Reducing to four components costs the classical models, depending on the dataset.} On WDBC, the classical models lost at most 0.005 of test AUC with respect to all the features. On Ionosphere, the loss was 0.066 for the SVM, 0.053 for XGBoost, 0.048 for the Random Forest, 0.025 for the MLP, and 0.019 for logistic regression. On the synthetic data, it was 0.186 for XGBoost and 0.174 for the Random Forest, and small for the SVM ($0.001$) and logistic regression ($0.044$); the MLP, in contrast, did better with four components than with all the features ($-0.066$), because its gap with 30 inputs was 0.225.

""" + "\n".join(t1) + r"""

\textit{Part of the deficit of the VQC comes from the reduction.} On Ionosphere, the VQC with four components was 0.119 below the SVM with all the features (95\% interval $-0.134$ to $-0.106$); with the same four inputs, the difference with the SVM was $-0.053$ ($-0.067$ to $-0.040$), so the rest corresponded to the reduction. On the synthetic data, the VQC was 0.171 below the Random Forest with all the features ($-0.181$ to $-0.162$), whereas with the same four inputs the difference was $+0.003$, with an interval that included zero. On WDBC, the difference with logistic regression with all the features was $-0.010$, of which $-0.006$ remained with the same inputs.

\textit{The choice of the reduction changes the VQC.} Replacing the principal components by the four features with the highest F-value raised the test AUC of the VQC on Ionosphere by 0.040 (interval 0.026 to 0.054), lowered it on the synthetic data by 0.047 ($-0.065$ to $-0.028$), and changed it by $-0.002$ on WDBC. On Ionosphere, the VQC also reduced its gap from 0.054 to 0.024 ($-0.031$, interval $-0.045$ to $-0.016$), and its difference with the MLP changed from $-0.033$ with principal components to $+0.010$ with the selected features, with an interval that included zero. On the synthetic data, the opposite occurred: the difference with the MLP went from $-0.006$ (interval including zero) to $-0.060$ ($-0.077$ to $-0.043$), probably because a univariate criterion cannot select features whose class structure is nonlinear. The sign and the size of the difference between the VQC and the classical models therefore depended on the representation.
"""
open("sec42_h1.tex", "w").write(sec1); print("ok")
# print numbers quoted, from raw data, for verification
pr = d1.pivot_table(index="seed", columns=["ds", "model", "input"], values=["auc_te", "gap"])
def dd(m, ds, a, b): 
    x = (pr[m][(ds,) + a] - pr[m][(ds,) + b]).values; return round(x.mean(), 3)
print("VQC anova-pca ion/syn/wdbc:", dd("auc_te","Ionosphere",("VQC","anova4"),("VQC","pca4")), dd("auc_te","Synthetic",("VQC","anova4"),("VQC","pca4")), dd("auc_te","WDBC",("VQC","anova4"),("VQC","pca4")))
print("gap VQC anova-pca ion:", dd("gap","Ionosphere",("VQC","anova4"),("VQC","pca4")))
print("VQC-MLP ion pca/anova:", dd("auc_te","Ionosphere",("VQC","pca4"),("MLP","pca4")), dd("auc_te","Ionosphere",("VQC","anova4"),("MLP","anova4")))
print("VQC-MLP syn pca/anova:", dd("auc_te","Synthetic",("VQC","pca4"),("MLP","pca4")), dd("auc_te","Synthetic",("VQC","anova4"),("MLP","anova4")))
print("VQC[pca4]-SVM[all] ion; VQC-SVM pca4 ion:", dd("auc_te","Ionosphere",("VQC","pca4"),("SVM","all")), dd("auc_te","Ionosphere",("VQC","pca4"),("SVM","pca4")))
print("VQC[pca4]-RF[all] syn; VQC-RF pca4 syn:", dd("auc_te","Synthetic",("VQC","pca4"),("RF","all")), dd("auc_te","Synthetic",("VQC","pca4"),("RF","pca4")))
print("VQC[pca4]-LR[all] wdbc; VQC-LR pca4 wdbc:", dd("auc_te","WDBC",("VQC","pca4"),("LR","all")), dd("auc_te","WDBC",("VQC","pca4"),("LR","pca4")))
print("MLP gap all syn:", round(d1[(d1.ds=="Synthetic")&(d1.model=="MLP")&(d1.input=="all")].gap.mean(),3))
print("MLP gap all ion:", round(d1[(d1.ds=="Ionosphere")&(d1.model=="MLP")&(d1.input=="all")].gap.mean(),3))
