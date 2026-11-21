import numpy as np, pandas as pd
SA = pd.read_csv("h2A_summary.csv", index_col=[0, 1]); SB = pd.read_csv("h2B_summary.csv", index_col=[0, 1]); T = pd.read_csv("h2_paired.csv")
f3 = lambda x: f"{x:.3f}".replace("-", "$-$")
def sg(x): return ("$+$" if x >= 0 else "$-$") + f"{abs(x):.3f}"
def D(part, ds, cmp, metric, ci=False, dag=False):
    r = T[(T.part == part) & (T.ds == ds) & (T["cmp"] == cmp) & (T.metric == metric)].iloc[0]
    s = sg(r.mean_diff)
    if ci: s += f" ({sg(r.lo)} to {sg(r.hi)})".replace("(", "[").replace(")", "]") if False else f" (95\\% interval {sg(r.lo)} to {sg(r.hi)})"
    if dag and r.excl0: s += "$^{\\dagger}$"
    return s
def DS(part, ds, cmp, metric):  # table cell with dagger
    return D(part, ds, cmp, metric, dag=True)
auc = lambda ds, m: SB.loc[(ds, m)].auc_te; gap = lambda ds, m: SB.loc[(ds, m)].gap
# ---------- Table: synthetic ----------
cols = [("VQC-base", "VQC"), ("VQC-reup2", "Re-up"), ("VQC-scale2", "$\\times$2"), ("MLP", "MLP"), ("SVM", "SVM"), ("RF", "RF")]
names = {"add1": "Additive, $k=1$", "add2": "Additive, $k=2$", "add3": "Additive, $k=3$", "prod1": "Product, $k=1$", "prod2": "Product, $k=2$", "prod3": "Product, $k=3$"}
rows = [f"{names[k]} & " + " & ".join(f3(auc(k, m)) for m, _ in cols) + " & " + " & ".join(f3(gap(k, m)) for m, _ in cols) + " \\\\" for k in names]
rows.insert(3, "\\hline")
tb = ["\\begin{table*}[t]", "\\centering",
      "\\caption{Encoding (H2) on synthetic data with controlled frequency $k$: test AUC and generalization gap (means over 30 seeds, a new sample of 600 points for each seed). VQC: four layers... }" if False else
      "\\caption{Encoding (H2) on synthetic data with controlled frequency $k$: test AUC and generalization gap (means over 30 seeds, each with a new sample of 600 points). VQC is the base angle encoding with one upload, Re-up uploads the features before each of the two layers, and $\\times$2 doubles the input angles.}",
      "\\label{tab:h2synth}", "\\footnotesize", "\\setlength{\\tabcolsep}{3.5pt}", "\\begin{tabular}{l" + "c" * 6 + "c" * 6 + "}", "\\hline",
      " & \\multicolumn{6}{c}{Test AUC} & \\multicolumn{6}{c}{Gap} \\\\",
      "Task & " + " & ".join(l for _, l in cols) + " & " + " & ".join(l for _, l in cols) + " \\\\", "\\hline"] + rows + ["\\hline", "\\end{tabular}", "\\end{table*}"]
# ---------- Table: real datasets ----------
DSR = ["WDBC", "Ionosphere", "Synthetic", "Transfusion"]
cm = [("scale05 - base", "$\\times$0.5 vs base"), ("scale2 - base", "$\\times$2 vs base"), ("reup2 - base", "Re-up vs base"), ("L4reup - L4", "Re-up vs base, 4 layers")]
rr = []
for ds in DSR:
    for i, (m, lab) in enumerate([("auc_te", "$\\Delta$ test AUC"), ("gap", "$\\Delta$ gap")]):
        rr.append(f"{ds if i == 0 else ''} & {lab} & " + " & ".join(DS("A", ds, c, m) for c, _ in cm) + " \\\\")
tr = ["\\begin{table*}[t]", "\\centering",
      "\\caption{Encoding (H2) on the four datasets: paired difference in test AUC and in generalization gap between each encoding variant and the reference indicated (mean over 30 seeds). A dagger marks a 95\\% bootstrap interval that excludes zero (no multiple-comparison correction). $\\times$0.5 and $\\times$2 scale the input angles; Re-up uploads the features before every layer.}",
      "\\label{tab:h2real}", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}", "\\begin{tabular}{llcccc}", "\\hline",
      "Dataset & Metric & " + " & ".join(l for _, l in cm) + " \\\\", "\\hline"] + rr + ["\\hline", "\\end{tabular}", "\\end{table*}"]
# ---------- prose ----------
txt = r"""\subsubsection{Data encoding (H2)}

In the base model each feature is uploaded once as an $R_Y$ rotation, so the output depends on each feature through frequencies $\{-1,0,1\}$ only. We changed the encoding while keeping the circuit, readout, loss, optimizer, and budget fixed, in three ways: (i) re-uploading, in which the feature layer is repeated before every variational layer (two uploads in the two-layer circuit and four in the four-layer circuit, with the same number of parameters); (ii) halving, and (iii) doubling the input angles. The new simulator reproduced the base VQC to within """ + f"{0.0004:.4f}" + r""" of test AUC, its gradients agreed with finite differences (maximum error $5\times10^{-11}$), and the output as a function of one feature contained frequencies up to 1 with one upload and up to 2 with two.

We ran these variants in two settings. First, on the four datasets with the same splits and preprocessing as before. Second, on synthetic data with controlled frequency: four features drawn uniformly in $[-\pi,\pi]$, a label given by the sign of $\sin kx_1+\sin kx_2$ (additive) or of $\sin kx_1\sin kx_2$ (product) with $k\in\{1,2,3\}$, 5\% label flips, and 600 samples, the last two features being irrelevant. Here each of the 30 seeds drew a new sample, so, unlike on the real datasets, the intervals reflect sampling of the data. All models received the same four inputs without further preprocessing, and no hyperparameter was retuned for the new settings.

\textit{With one upload, the VQC failed when the function required higher frequencies.} For $k=1$, the VQC reached a test AUC of """ + f"{auc('add1','VQC-base'):.3f}" + " and " + f"{auc('prod1','VQC-base'):.3f}" + r""" on the additive and product tasks, with gaps of """ + f"{gap('add1','VQC-base'):.3f}" + " and " + f"{gap('prod1','VQC-base'):.3f}" + r""" (Table~\ref{tab:h2synth}, Fig.~\ref{fig:h2}); its gap did not exceed that of the MLP on the additive task (""" + D("B", "add1", "base - MLP", "gap", True) + r""") and was lower on the product task (""" + D("B", "prod1", "base - MLP", "gap", True) + r"""). For $k=2$ its test AUC fell to """ + f"{auc('add2','VQC-base'):.3f}" + " and " + f"{auc('prod2','VQC-base'):.3f}" + r""", that is, to chance, while its training AUC stayed above 0.6, so the gap rose to """ + f"{gap('add2','VQC-base'):.3f}" + " and " + f"{gap('prod2','VQC-base'):.3f}" + r""", which was larger than that of the MLP (""" + D("B", "add2", "base - MLP", "gap", True) + " and " + D("B", "prod2", "base - MLP", "gap", True) + r"""). For $k=3$ the VQC stayed at chance (""" + f"{auc('add3','VQC-base'):.3f}" + " and " + f"{auc('prod3','VQC-base'):.3f}" + r"""). The failure was not exclusive to the VQC: with $k=2$ the MLP and the SVM also lost most of their performance (test AUC """ + f"{auc('add2','MLP'):.3f}" + " and " + f"{auc('add2','SVM'):.3f}" + r""" on the additive task), whereas the Random Forest and XGBoost remained above 0.75 on the tasks with $k=2$ (""" + f"{auc('add2','RF'):.3f}" + " and " + f"{auc('prod2','RF'):.3f}" + r""" for the Random Forest). The gap at $k=3$ was larger than at $k=1$ for the VQC, the MLP, the SVM, the Random Forest, and XGBoost on both families of tasks, whereas that of logistic regression stayed below 0.07, so the increase of the gap alone does not identify the encoding as its cause.

\textit{Matching the encoding to the frequency recovered performance and reduced the gap.} For $k=2$, re-uploading raised the test AUC of the VQC by """ + D("B", "add2", "reup2 - base", "auc_te", True) + " on the additive task and by " + D("B", "prod2", "reup2 - base", "auc_te", True) + r""" on the product task, and reduced the gap (""" + D("B", "add2", "reup2 - base", "gap") + " and " + D("B", "prod2", "reup2 - base", "gap") + r"""); with it the VQC exceeded the MLP in test AUC on both tasks (""" + D("B", "add2", "reup2 - MLP", "auc_te") + " and " + D("B", "prod2", "reup2 - MLP", "auc_te") + r"""). Doubling the input angles, which makes the encoded frequency equal to the one of the target by construction, brought the VQC to a test AUC of """ + f"{auc('add2','VQC-scale2'):.3f}" + " and " + f"{auc('prod2','VQC-scale2'):.3f}" + r""" with gaps of """ + f"{gap('add2','VQC-scale2'):.3f}" + " and " + f"{gap('prod2','VQC-scale2'):.3f}" + r""", and, as expected, it failed for $k=1$ and $k=3$ (test AUC between """ + f"{min(auc(k,'VQC-scale2') for k in ['add1','add3','prod1','prod3']):.3f}" + " and " + f"{max(auc(k,'VQC-scale2') for k in ['add1','add3','prod1','prod3']):.3f}" + r"""). This shows the mechanism, not a practical remedy, because it uses knowledge of the target. For $k=3$ none of the variants learned the task (test AUC at most """ + f"{max(auc(k,m) for k in ['add3','prod3'] for m in ['VQC-reup2','VQC-L4reup','VQC-scale2']):.3f}" + r"""), although four re-uploads improved on four layers with one upload on the additive task (""" + D("B", "add3", "L4reup - L4", "auc_te", True) + r""").

\textit{Re-uploading is not a superset of the base encoding.} With $k=1$, where the base VQC succeeded, re-uploading lowered the test AUC by """ + D("B", "add1", "reup2 - base", "auc_te", True) + " on the additive task and by " + D("B", "prod1", "reup2 - base", "auc_te", True) + r""" on the product task (to """ + f"{auc('prod1','VQC-reup2'):.3f}" + r"""). In a probe on three seeds of the product task with $k=1$, the training AUC stayed near 0.67 and the test AUC near 0.5 with three times more epochs and with a larger initialization, so this was not a matter of optimization. The second upload follows the first variational layer, which changes the class of functions the circuit can represent, and the base circuit is not contained in it.

""" + "\n".join(tb) + r"""

\textit{On the real datasets, extra frequency capacity did not help and a smaller bandwidth tended to help.} Re-uploading did not improve the test AUC on any dataset (changes of """ + ", ".join(f"{sg(T[(T.part=='A')&(T.ds==d)&(T['cmp']=='reup2 - base')&(T.metric=='auc_te')].mean_diff.iloc[0])}" for d in DSR) + r""" on WDBC, Ionosphere, the synthetic dataset, and Blood Transfusion; only the first interval excluded zero), and with four layers, uploading before every layer was worse than a single upload (changes of test AUC between """ + f"{min(T[(T.part=='A')&(T['cmp']=='L4reup - L4')&(T.metric=='auc_te')].mean_diff):+.3f}".replace('-', '$-$') + " and " + f"{max(T[(T.part=='A')&(T['cmp']=='L4reup - L4')&(T.metric=='auc_te')].mean_diff):+.3f}".replace('-', '$-$') + r""" and larger gaps, by """ + f"{min(T[(T.part=='A')&(T['cmp']=='L4reup - L4')&(T.metric=='gap')].mean_diff):.3f}" + " to " + f"{max(T[(T.part=='A')&(T['cmp']=='L4reup - L4')&(T.metric=='gap')].mean_diff):.3f}" + r"""). Doubling the input angles lowered the test AUC on all four datasets (""" + f"{min(T[(T.part=='A')&(T['cmp']=='scale2 - base')&(T.metric=='auc_te')].mean_diff):+.3f}".replace('-', '$-$') + " to " + f"{max(T[(T.part=='A')&(T['cmp']=='scale2 - base')&(T.metric=='auc_te')].mean_diff):+.3f}".replace('-', '$-$') + r"""), whereas halving them left it similar or slightly higher (""" + ", ".join(sg(T[(T.part=='A')&(T.ds==d)&(T['cmp']=='scale05 - base')&(T.metric=='auc_te')].mean_diff.iloc[0]) for d in DSR) + r""" on WDBC, Ionosphere, the synthetic dataset, and Blood Transfusion) and reduced the gap on all four datasets (""" + ", ".join(sg(T[(T.part=='A')&(T.ds==d)&(T['cmp']=='scale05 - base')&(T.metric=='gap')].mean_diff.iloc[0]) for d in DSR) + r""", with the interval excluding zero on three of them; Table~\ref{tab:h2real}). Halving the angles also narrows their range to $[-\pi/2,\pi/2]$, which avoids mapping the two ends of each feature to the same quantum state, and we did not separate this effect from the change of bandwidth.

""" + "\n".join(tr) + "\n"
txt = txt.replace("\\begin{figure*}", "\\begin{figure*}")
fig = r"""
\begin{figure*}[t]
\centering
\includegraphics[width=0.62\textwidth]{fig_h2.png}
\caption{Test AUC (top) and generalization gap (bottom) as a function of the frequency $k$ of the target for the additive (left) and product (right) synthetic tasks, averaged over 30 seeds.}
\label{fig:h2}
\end{figure*}
"""
# put the figure after the first results paragraph block (before table)
txt = txt.replace("\\textit{Matching the encoding", fig.strip() + "\n\n\\textit{Matching the encoding", 1)
open("sec42_h2.tex", "w").write(txt); print("ok")
