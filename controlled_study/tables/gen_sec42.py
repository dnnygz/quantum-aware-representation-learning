import pandas as pd, numpy as np
S = pd.read_csv("e1_summary.csv", index_col=[0, 1]); H = pd.read_csv("e1_shuffled.csv", index_col=[0, 1]); T = pd.read_csv("e1_paired.csv")
DS = ["WDBC", "Ionosphere", "Synthetic", "Transfusion"]; M = ["VQC", "LR", "SVM", "RF", "XGB", "MLP"]
NAME = {"LR": "LR", "SVM": "SVM", "RF": "RF", "XGB": "XGBoost", "MLP": "MLP", "VQC": "VQC"}
f3 = lambda x: f"{x:.3f}".replace("-", "$-$")
# ---- Table 1 (table*): test AUC (SD) and gap
t1 = ["\\begin{table*}[t]", "\\centering",
      "\\caption{Test AUC (mean $\\pm$ SD over 30 seeds) and generalization gap (training AUC minus test AUC, mean) for the VQC and the classical comparators, all receiving the same four inputs. Best test AUC per dataset in bold.}",
      "\\label{tab:baseline}", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}", "\\begin{tabular}{l" + "cc" * len(DS) + "}", "\\hline",
      " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{d}}}" for d in DS) + " \\\\",
      " & " + " & ".join("Test AUC & Gap" for _ in DS) + " \\\\", "\\hline"]
best = {d: S.loc[d]["auc_te"].round(3).max() for d in DS}
for m in M:
    cells = []
    for d in DS:
        a, sd, g = S.loc[(d, m)]["auc_te"], S.loc[(d, m)]["sd_te"], S.loc[(d, m)]["gap"]
        s = f"{a:.3f}$\\pm${sd:.3f}"
        if round(a, 3) == best[d]: s = f"\\textbf{{{s}}}"
        cells += [s, f3(g)]
    t1.append(NAME[m] + " & " + " & ".join(cells) + " \\\\")
t1 += ["\\hline", "\\end{tabular}", "\\end{table*}"]
# ---- Table 2: paired VQC - MLP
def cells2(ds, metric):
    r = T[(T.ds == ds) & (T.vs == "MLP") & (T.metric == metric)].iloc[0]
    star = "$^{*}$" if r.p_holm < 0.05 else ""
    return f"{f3(r.mean_diff)}{star}", f"[{f3(r.ci_lo)}, {f3(r.ci_hi)}]"
t2 = ["\\begin{table}[t]", "\\centering",
      "\\caption{Paired differences between the VQC and the MLP over the same 30 seeds: mean difference and 95\\% bootstrap confidence interval. An asterisk marks differences that remain significant after Holm correction ($p<0.05$, Wilcoxon signed-rank test, 20 comparisons per metric).}",
      "\\label{tab:paired}", "\\footnotesize", "\\setlength{\\tabcolsep}{2pt}", "\\begin{tabular}{lcccc}", "\\hline",
      " & \\multicolumn{2}{c}{$\\Delta$ test AUC} & \\multicolumn{2}{c}{$\\Delta$ gap} \\\\",
      "Dataset & Mean & 95\\% CI & Mean & 95\\% CI \\\\", "\\hline"]
for d in DS:
    a, b = cells2(d, "auc_te"); c, e = cells2(d, "gap")
    t2.append(f"{d} & {a} & {b} & {c} & {e} \\\\")
t2 += ["\\hline", "\\end{tabular}", "\\end{table}"]
# ---- Table 3: label randomization (training AUC)
t3 = ["\\begin{table}[t]", "\\centering",
      "\\caption{Training AUC when the training labels are randomly permuted (mean over 30 seeds). A value of 0.5 means no memorization.}",
      "\\label{tab:shuffled}", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{l" + "c" * len(DS) + "}", "\\hline",
      "Model & " + " & ".join(DS) + " \\\\", "\\hline"]
for m in M: t3.append(NAME[m] + " & " + " & ".join(f3(H.loc[(d, m)]["tr"]) for d in DS) + " \\\\")
t3 += ["\\hline", "\\end{tabular}", "\\end{table}"]
open("tables_sec42.tex", "w").write("\n".join(t1 + [""] + t2 + [""] + t3) + "\n")
print("\n".join(t1 + [""] + t2 + [""] + t3))
