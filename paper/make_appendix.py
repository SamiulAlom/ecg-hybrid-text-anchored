"""Generate the appendix tables (diagnosis descriptions, implication rules) straight from src/ so the paper
always matches the code that produced the results."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from data import IMPLIES  # noqa: E402
from labels import EQUIVALENT, LABELS, SCORED_26  # noqa: E402


def tex(s):
    return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


rows = []
for code, (abbr, name, desc) in LABELS.items():
    scored = r"\checkmark" if code in SCORED_26 else ""
    rows.append(f"{abbr} & {code} & {scored} & {tex(name[0].upper() + name[1:])}: {tex(desc)} \\\\")

out = [r"\begingroup\footnotesize",
       r"\begin{longtable}{@{}l l c p{0.62\textwidth}@{}}",
       r"\caption{Diagnosis vocabulary and frozen text descriptions used as label anchors. "
       r"Scored = one of the 26 PhysioNet/CinC 2021 scored classes.}\label{tab:descriptions}\\",
       r"\toprule Abbr. & SNOMED-CT & Scored & Text anchor \\ \midrule \endfirsthead",
       r"\toprule Abbr. & SNOMED-CT & Scored & Text anchor \\ \midrule \endhead",
       *rows,
       r"\bottomrule \end{longtable}", r"\endgroup", ""]
with open(os.path.join(HERE, "tab_descriptions.tex"), "w", encoding="utf-8") as f:
    f.write("\n".join(out))

imp = [f"{LABELS[c][0]} $\\rightarrow$ {', '.join(LABELS[p][0] for p in ps)} \\\\" for c, ps in IMPLIES.items()]
eq = [f"{c} $\\rightarrow$ {LABELS[t][0]} ({t}) \\\\" for c, t in EQUIVALENT.items()]
with open(os.path.join(HERE, "tab_rules.tex"), "w", encoding="utf-8") as f:
    f.write("\n".join([
        r"\begin{table}[h]\centering\footnotesize",
        r"\caption{Label rules. Left: clinical implication rules used for hierarchy completion (child $\rightarrow$ "
        r"implied finding). Right: official PhysioNet/CinC 2021 code equivalences applied to all methods.}"
        r"\label{tab:rules}",
        r"\begin{tabular}{@{}l@{}}\toprule Implication \\ \midrule", *imp, r"\bottomrule\end{tabular}\qquad",
        r"\begin{tabular}{@{}l@{}}\toprule Merged code $\rightarrow$ class \\ \midrule", *eq,
        r"\bottomrule\end{tabular}", r"\end{table}", ""]))
print("wrote tab_descriptions.tex, tab_rules.tex")
