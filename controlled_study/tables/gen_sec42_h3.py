import pandas as pd
S = pd.read_csv("h3_summary_ep200.csv", index_col=[0, 1]); H = pd.read_csv("h3_shuffled.csv", index_col=[0, 1])
DS = ["WDBC", "Ionosphere", "Synthetic", "Transfusion"]
ROWS = [("L1", "VQC, 1 layer", "13/9"), ("MLP h2 (13p)", "MLP, 2 hidden units", "13/13"),
        ("L2", "VQC, 2 layers (base)", "21/17"), ("MLP h3 (19p)", "MLP, 3 hidden units", "19/19"),
        ("L4", "VQC, 4 layers", "37/33"), ("MLP h6 (37p)", "MLP, 6 hidden units", "37/37"),
        ("L2-noent", "VQC, 2 layers, no entanglement", "21/13"), ("L2-nohead", "VQC, 2 layers, no linear head", "17/13")]
t1 = ["\\begin{table*}[t]", "\\centering",
      "\\caption{Capacity and readout (H3), after 200 epochs: nominal/effective number of trainable parameters, test AUC, and generalization gap (means over 30 seeds). The effective count of the circuit is the numerical rank of the Jacobian of the four expectation values with respect to the rotation angles, plus the readout parameters.}",
      "\\label{tab:h3}", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}", "\\begin{tabular}{llcccccccc}", "\\hline",
      " & & \\multicolumn{4}{c}{Test AUC} & \\multicolumn{4}{c}{Gap} \\\\",
      "Model & Params & " + " & ".join(DS) + " & " + " & ".join(DS) + " \\\\", "\\hline"]
for k, name, p in ROWS:
    t1.append(f"{name} & {p} & " + " & ".join(f"{S.loc[(d, k)]['auc_te']:.3f}" for d in DS) + " & " + " & ".join(f"{S.loc[(d, k)]['gap']:.3f}" for d in DS) + " \\\\")
    if k in ("MLP h2 (13p)", "MLP h3 (19p)", "MLP h6 (37p)"): t1.append("\\hline" if k == "MLP h6 (37p)" else "")
t1 = [l for l in t1 if l != ""] + ["\\hline", "\\end{tabular}", "\\end{table*}"]
t2 = ["\\begin{table}[t]", "\\centering",
      "\\caption{Training AUC when the training labels are randomly permuted (200 epochs, mean over 30 seeds), by model capacity.}",
      "\\label{tab:h3shuffled}", "\\footnotesize", "\\setlength{\\tabcolsep}{2.5pt}", "\\begin{tabular}{lcccc}", "\\hline", "Model & " + " & ".join(DS) + " \\\\", "\\hline"]
for k, name, p in ROWS:
    short = name.replace("VQC, 2 layers, no entanglement", "VQC, 2 layers, no ent.").replace("VQC, 2 layers, no linear head", "VQC, 2 layers, no head")
    t2.append(f"{short} & " + " & ".join(f"{H.loc[(d, k)]['tr']:.3f}" for d in DS) + " \\\\")
t2 += ["\\hline", "\\end{tabular}", "\\end{table}"]
sec = r"""\subsubsection{Capacity and readout (H3)}

To test whether the gap depends on capacity and on the readout, we varied one component of the VQC at a time: the number of layers (1, 2, and 4), the entangling gates (removed), and the linear output layer (replaced by a fixed sum of the four expectation values). We compared the one-, two-, and four-layer VQCs with MLPs of a similar number of parameters (13, 19, and 37). All models were trained for 200 epochs. The main conclusions below also held at 400 epochs, except that the small deficit of the four-layer VQC on Blood Transfusion was no longer clear. % TODO: report the 400-epoch table in the repository

\textit{Effective parameters.} Not all circuit parameters change the output. The rank of the Jacobian of the four expectation values, computed numerically at random inputs and angles, was 4 of 8 for one layer, 12 of 16 for two layers, 28 of 32 for four layers, and 8 of 16 for two layers without entanglement. Adding the readout parameters, the base model had 21 nominal and 17 effective parameters (Table~\ref{tab:h3}).

""" + "\n".join(t1) + r"""

\textit{Depth.} Increasing the circuit from two to four layers raised the test AUC on Ionosphere by 0.049 (95\% interval 0.036 to 0.063) and on WDBC by 0.004 (0.003 to 0.006), and had no clear effect on the synthetic data and on Blood Transfusion ($-0.002$ in both, with intervals including zero). A single layer lowered the test AUC on WDBC ($-0.016$), on Ionosphere ($-0.059$), and on Blood Transfusion ($-0.037$). The gap increased with depth on the synthetic data ($+0.034$ from two to four layers, interval 0.024 to 0.043) and on Blood Transfusion ($+0.014$, 0.009 to 0.020), and did not change clearly on WDBC and Ionosphere.

\textit{Comparison with MLPs of the same size.} With four layers (37 nominal parameters), the VQC showed no test deficit with respect to the MLP with the same number of parameters on WDBC ($-0.001$), Ionosphere ($+0.004$), and the synthetic data ($-0.007$), with intervals including zero in all three, and a small deficit on Blood Transfusion ($-0.006$, interval $-0.011$ to $-0.002$). The Ionosphere deficit of the base model ($-0.033$ against the three-unit MLP) therefore disappeared when the capacity was increased. In contrast, with one layer (13 nominal parameters), the VQC was clearly below the MLP of the same size on WDBC ($-0.022$) and on Ionosphere ($-0.084$). The gap of the four-layer VQC was smaller than that of the six-unit MLP on Ionosphere ($-0.015$, interval $-0.024$ to $-0.005$), larger on Blood Transfusion ($+0.017$, 0.011 to 0.022), and similar on WDBC and the synthetic data.

\textit{Entanglement and readout.} Removing the entangling gates, which also reduced the effective parameters from 12 to 8, did not change the test AUC clearly on Ionosphere ($+0.018$, interval $-0.000$ to 0.036), the synthetic data ($-0.000$), and Blood Transfusion ($-0.004$), and lowered it slightly on WDBC ($-0.003$). However, it reduced the gap on Ionosphere ($-0.033$), the synthetic data ($-0.025$), and Blood Transfusion ($-0.018$), with intervals excluding zero. Because the circuit without entanglement has fewer effective parameters, this reduction cannot be attributed to entanglement alone. Replacing the linear output layer by a fixed sum lowered the test AUC on WDBC ($-0.008$) and on Ionosphere ($-0.048$), and reduced the gap on all four datasets (from $-0.004$ on WDBC to $-0.032$ on the synthetic data).

\textit{Memorization.} Table~\ref{tab:h3shuffled} shows that the training AUC on randomly permuted labels increased with capacity: from 0.581--0.620 with one layer to 0.605--0.671 with two layers and 0.628--0.730 with four layers. The MLPs behaved similarly, from 0.519--0.606 with two hidden units to 0.576--0.758 with six. The circuit without entanglement had the lowest values (0.565--0.604). No variant approached the near-perfect fit of the Random Forest and XGBoost (Table~\ref{tab:shuffled}).

""" + "\n".join(t2) + "\n"
open("sec42_h3.tex", "w").write(sec); print(len(sec.split()), "words")
