"""Supplementary tables S2-S4 -> paper/generated/supp_*.tex

S2 supp_counts.tex   positive recordings per vocabulary diagnosis and database (from prep meta.csv, after the official
                     code equivalences, i.e. exactly the label matrix train.py uses)
S3 supp_perlabel.tex per-diagnosis AUROC of the proposed method and the strongest baseline per held-out database
                     (seed 0, DeLong p Holm-adjusted; from paper/generated/stats.csv), plus unseen diagnoses
S4 supp_e2.tex       removed diagnoses of the annotation-removal experiment with per-method AUROC
"""
import csv
import glob
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from data import MAIN_SOURCES, build_vocab, label_matrices  # noqa: E402
from labels import LABELS  # noqa: E402

OUT = os.path.join(ROOT, "paper", "generated")
PREP = os.path.join(ROOT, "results", "kaggle", "prep", "prep")
ORDER = ["CPSC", "PTBXL", "GEORGIA", "SHAONING"]
TT = {"CPSC": "CPSC", "PTBXL": "PTB-XL", "GEORGIA": "Georgia", "SHAONING": "S--N"}
NAME = {"naive": "Naive", "mask": "Masked", "li2024": "Li-style", "coral": "CORAL", "dann": "DANN",
        "groupdro": "GroupDRO", "mixstyle": "MixStyle"}
ABBR = {c: v[0] for c, v in LABELS.items()}


def tex(s):
    return s.replace("_", r"\_").replace("&", r"\&")


def counts():
    meta = pd.read_csv(os.path.join(PREP, "meta.csv"), dtype={"dx": str})
    meta["dx"] = meta["dx"].fillna("")
    meta = meta[meta.source.isin(MAIN_SOURCES)].reset_index(drop=True)
    vocab, annotated, _ = build_vocab(meta, min_source=10)
    Y, A = label_matrices(meta, vocab, annotated)
    rows = []
    for j, c in enumerate(vocab):
        n = {s: int(Y[meta.source.values == s, j].sum()) for s in ORDER}
        rows.append((ABBR.get(c, c), c, n, sum(n.values())))
    rows.sort(key=lambda r: -r[3])
    lines = [r"\begin{longtable}{@{}llrrrrr@{}}",
             r"\caption{Positive recordings per diagnosis and database after the official code equivalences, i.e.\ "
             r"the label matrix used for training and evaluation. Bold: the database annotates the diagnosis "
             r"($\geq10$ positives).}\label{tab:counts}\\",
             r"\toprule Abbr. & SNOMED-CT & CPSC & PTB-XL & Georgia & S--N & Total \\ \midrule \endfirsthead",
             r"\toprule Abbr. & SNOMED-CT & CPSC & PTB-XL & Georgia & S--N & Total \\ \midrule \endhead",
             r"\bottomrule \endfoot"]
    for a, c, n, tot in rows:
        cells = [rf"\textbf{{{n[s]:,}}}" if n[s] >= 10 else f"{n[s]:,}" for s in ORDER]
        lines.append(" & ".join([tex(a), c] + cells + [f"{tot:,}"]).replace(",", "{,}") + r" \\")
    lines.append(r"\end{longtable}")
    open(os.path.join(OUT, "supp_counts.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    return len(rows)


def perlabel():
    R = {}
    for f in glob.glob(os.path.join(ROOT, "results", "final", "loso-*-s0.json")):
        d = json.load(open(f))
        R[d["name"]] = d
    st = [r for r in csv.DictReader(open(os.path.join(OUT, "stats.csv"))) if r["test"] == "delong_per_label"]
    lines = [r"\begin{longtable}{@{}llrrrrl@{}}",
             r"\caption{Per-diagnosis AUROC ($\times100$) under leave-one-database-out (seed 0): proposed method "
             r"against the strongest baseline of each held-out database, with DeLong $p$ Holm-adjusted over all "
             r"per-diagnosis tests. Unseen diagnoses ($\dagger$) are listed with the best baseline by AUROC and no "
             r"test.}\label{tab:perlabel}\\",
             r"\toprule Target & Diagnosis & Positives & Ours & Baseline & $p_{\mathrm{Holm}}$ & Baseline \\ "
             r"\midrule \endfirsthead",
             r"\toprule Target & Diagnosis & Positives & Ours & Baseline & $p_{\mathrm{Holm}}$ & Baseline \\ "
             r"\midrule \endhead", r"\bottomrule \endfoot"]
    for t in ORDER:
        for r in [r for r in st if r["target"] == t]:
            p = float(r["p_holm"])
            ps = r"$<$0.001" if p < 0.001 else f"{p:.3f}"
            if p < 0.05:
                ps = rf"\textbf{{{ps}}}"
            lines.append(" & ".join([TT[t], tex(r["label"]), r["n_pos"], f"{float(r['auroc_ours']) * 100:.1f}",
                                     f"{float(r['auroc_baseline']) * 100:.1f}", ps, NAME.get(r["baseline"], r["baseline"])])
                         + r" \\")
        ours = R[f"loso-{t}-ours-s0"]
        for c in ours["unseen"]:
            a = ABBR[c]
            if a not in ours["per_label"]:
                continue
            best = max(NAME, key=lambda m: R[f"loso-{t}-{m}-s0"]["per_label"][a]["auroc"])
            lines.append(" & ".join([TT[t], tex(a) + r"$^\dagger$", str(ours["per_label"][a]["n_pos"]),
                                     f"{ours['per_label'][a]['auroc'] * 100:.1f}",
                                     f"{R[f'loso-{t}-{best}-s0']['per_label'][a]['auroc'] * 100:.1f}", "--",
                                     NAME[best]]) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = lines[-1].replace(r"\midrule", "")
    lines.append(r"\end{longtable}")
    open(os.path.join(OUT, "supp_perlabel.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")


def e2():
    R = {}
    for f in glob.glob(os.path.join(ROOT, "results", "final", "e4-*.json")):
        d = json.load(open(f))
        R[d["name"]] = d
    lines = [r"\begin{table}[h]\centering\footnotesize",
             r"\caption{Diagnoses removed from PTB-XL training supervision in the annotation-removal experiment and "
             r"their test AUROC ($\times100$, single seed). Positives: number of positive test recordings.}"
             r"\label{tab:e2diag}",
             r"\begin{tabular}{llrrrr}", r"\toprule $r$ & Diagnosis & Positives & Naive & Li-style & Ours \\ \midrule"]
    for r in ["0.25", "0.5"]:
        d = R[f"e4-PTBXL-oursr{r}-s0"]
        drop = [ABBR[c] for s_, c in d["dropped"] if s_ == "PTBXL"]
        for a in drop:
            if a not in d["per_label"]:
                continue
            v = [R[f"e4-PTBXL-{m}r{r}-s0"]["per_label"][a]["auroc"] * 100 for m in ["naive", "li2024", "ours"]]
            lines.append(" & ".join([r, tex(a), str(d["per_label"][a]["n_pos"])] + [f"{x:.1f}" for x in v]) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]
    open(os.path.join(OUT, "supp_e2.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    print("vocabulary rows:", counts())
    perlabel()
    e2()
    print("written to", OUT)
