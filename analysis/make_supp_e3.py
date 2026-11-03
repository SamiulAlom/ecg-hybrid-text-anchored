"""Supplementary Table S5: every E3 (held-out database, diagnosis) pair -> paper/generated/supp_e3.tex"""
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "generated")
TT = {"CPSC": "CPSC", "PTBXL": "PTB-XL", "GEORGIA": "Georgia", "SHAONING": "S--N"}
rows = list(csv.DictReader(open(os.path.join(OUT, "e3_pairs.csv"))))
order = {t: i for i, t in enumerate(TT)}
rows.sort(key=lambda r: (order[r["target"]], int(r["group"]), r["abbr"]))
f = lambda v: f"{float(v) * 100:.1f}" if v not in ("", "nan") else "--"
lines = [r"\begin{longtable}{@{}llcrrrrrl@{}}",
         r"\caption{Systematic zero-shot evaluation (E3), all 120 (held-out database, diagnosis) pairs. Group: the "
         r"withheld group (drawn once at random, seed 0). Zero-shot: AUROC ($\times100$) of the text head when the "
         r"diagnosis is withheld, mean$\pm$std over three seeds. Supervised: AUROC of the proposed method when the "
         r"diagnosis is trained (E1, mean over seeds). MERL: zero-shot with its own prompts / our descriptions. "
         r"Nearest: trained diagnosis with the most similar description anchor (cosine).}\label{tab:e3pairs}\\",
         r"\toprule DB & Diagnosis & Group & Positives & Zero-shot & Supervised & MERL & MERL (desc.) & Nearest \\ "
         r"\midrule \endfirsthead",
         r"\toprule DB & Diagnosis & Group & Positives & Zero-shot & Supervised & MERL & MERL (desc.) & Nearest \\ "
         r"\midrule \endhead", r"\bottomrule \endfoot"]
for r in rows:
    zs = f"{float(r['zs']) * 100:.1f}$\\pm${float(r['zs_std']) * 100:.1f}"
    lines.append(" & ".join([TT[r["target"]], r["abbr"], str(int(r["group"]) + 1), r["n_pos"], zs, f(r["sup"]),
                             f(r["merl"]), f(r["merl_desc"]),
                             f"{r['nearest_abbr']} ({float(r['nearest']):.2f})"]) + r" \\")
lines.append(r"\end{longtable}")
open(os.path.join(OUT, "supp_e3.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
print(len(rows), "rows")
