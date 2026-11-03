"""Calibration of the final runs (results/final) on seen diagnoses of the held-out database.

For every run, entries (record, diagnosis) with a seen diagnosis that the held-out database annotates are pooled.
Reported per method (mean +- std over the 12 fold x seed runs):
  ECE   expected calibration error, 15 equal-width bins over pooled entries
  Brier mean squared error of the probability
  O/E   ratio of mean predicted probability to observed prevalence (1 = calibrated in the large)
Writes paper/generated/calibration.csv, tab_calibration.tex and fig_calibration.pdf (reliability diagram, pooled
over folds, seed 0).
"""
import glob
import json
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, OUT = os.path.join(ROOT, "results", "final"), os.path.join(ROOT, "paper", "generated")
METHODS = ["naive", "mask", "li2024", "coral", "dann", "groupdro", "mixstyle", "ours"]
NAME = {"naive": "Naive pooling", "mask": "Masked loss", "li2024": "Li-style", "coral": "CORAL", "dann": "DANN",
        "groupdro": "GroupDRO", "mixstyle": "MixStyle", "ours": "Ours (hybrid)"}
TARGETS = ["CPSC", "PTBXL", "GEORGIA", "SHAONING"]
NB = 15


def entries(m, t, s):
    d = json.load(open(os.path.join(RES, f"loso-{t}-{m}-s{s}.json")))
    z = np.load(os.path.join(RES, f"loso-{t}-{m}-s{s}_test.npz"))
    cols = [d["vocab"].index(c) for c in d["seen"]]
    P, Y, A = z["P"].astype(np.float64)[:, cols], z["Y"][:, cols], z["A"][:, cols]
    k = A > 0
    return P[k], Y[k].astype(np.float64)


def ece(p, y):
    b = np.minimum((p * NB).astype(int), NB - 1)
    e = 0.0
    for i in range(NB):
        m = b == i
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e


def main():
    rows = {}
    for m in METHODS:
        E, B, R = [], [], []
        for t in TARGETS:
            for s in range(3):
                p, y = entries(m, t, s)
                E.append(ece(p, y)); B.append(np.mean((p - y) ** 2)); R.append(p.mean() / y.mean())
        rows[m] = (E, B, R)
        print(f"{m:9s} ECE {st.mean(E):.4f}+-{st.stdev(E):.4f}  Brier {st.mean(B):.4f}  O/E {st.mean(R):.2f}")
    with open(os.path.join(OUT, "calibration.csv"), "w") as f:
        f.write("method,ece_mean,ece_std,brier_mean,brier_std,oe_mean,oe_std\n")
        for m, (E, B, R) in rows.items():
            f.write(f"{m},{st.mean(E):.5f},{st.stdev(E):.5f},{st.mean(B):.5f},{st.stdev(B):.5f},"
                    f"{st.mean(R):.4f},{st.stdev(R):.4f}\n")
    best = min(st.mean(rows[m][0]) for m in METHODS)
    lines = [r"\begin{table}[!htbp]", r"\centering\footnotesize",
             r"\caption{Calibration on seen diagnoses of the held-out database, pooled over annotated "
             r"record--diagnosis entries: expected calibration error (ECE, 15 bins, $\times100$), Brier score "
             r"($\times100$) and the ratio of mean predicted probability to observed prevalence (O/E; 1 is "
             r"calibrated in the large), mean$_{\pm\mathrm{std}}$ over four folds and three seeds. No method was "
             r"recalibrated.}\label{tab:calibration}",
             r"\begin{tabular}{lccc}", r"\toprule", r"Method & ECE & Brier & O/E \\", r"\midrule"]
    for m in METHODS:
        E, B, R = rows[m]
        e = f"{st.mean(E) * 100:.2f}$_{{\\pm {st.stdev(E) * 100:.2f}}}$"
        if np.isclose(st.mean(E), best):
            e = r"\textbf{" + e.split("$")[0] + "}$" + "$".join(e.split("$")[1:])
        if m == "ours":
            lines.append(r"\midrule")
        lines.append(f"{NAME[m]} & {e} & {st.mean(B) * 100:.2f}$_{{\\pm {st.stdev(B) * 100:.2f}}}$ & "
                     f"{st.mean(R):.2f}$_{{\\pm {st.stdev(R):.2f}}}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(OUT, "tab_calibration.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

    plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(8.5 / 2.54, 8 / 2.54))
    cols = {"naive": "#E69F00", "mask": "#56B4E9", "li2024": "#0072B2", "ours": "#D55E00"}
    edges = np.linspace(0, 1, NB + 1)
    for m, c in cols.items():
        ps, ys = zip(*[entries(m, t, 0) for t in TARGETS])
        p, y = np.concatenate(ps), np.concatenate(ys)
        b = np.minimum((p * NB).astype(int), NB - 1)
        xs = [p[b == i].mean() for i in range(NB) if (b == i).sum() >= 50]
        ysb = [y[b == i].mean() for i in range(NB) if (b == i).sum() >= 50]
        ax.plot(xs, ysb, marker="o", ms=2.5, lw=1, color=c, label=NAME[m])
    ax.plot([0, 1], [0, 1], ls=":", color="k", lw=0.7)
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed frequency")
    ax.legend(frameon=False, loc="upper left"); ax.grid(lw=0.3, alpha=0.5)
    fig.savefig(os.path.join(OUT, "fig_calibration.pdf"), bbox_inches="tight")
    print("written")


if __name__ == "__main__":
    main()
