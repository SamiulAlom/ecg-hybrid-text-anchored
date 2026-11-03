"""Figures for the final results (results/final, plus the development runs for the appendix) -> paper/generated.

fig_seen_seeds.pdf  seen-diagnosis macro AUROC per fold, every seed of every method
fig_unseen.pdf      per-diagnosis AUROC on unseen diagnoses (CPSC, Shaoxing-Ningbo), mean over seeds
fig_e2.pdf          annotation removal on PTB-XL: retained / removed macro AUROC vs removal rate, and per diagnosis
fig_curves.pdf      validation AUROC per epoch: earlier text-only design (v1) vs Li-style; hybrid vs Li-style
fig_ema.pdf         effect of weight averaging on each baseline (seed 0, mean over folds)
"""
import glob
import json
import os
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from labels import LABELS  # noqa: E402

OUT = os.path.join(ROOT, "paper", "generated")
OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
CM = 1 / 2.54
TARGETS = ["CPSC", "PTBXL", "GEORGIA", "SHAONING"]
TT = {"CPSC": "CPSC", "PTBXL": "PTB-XL", "GEORGIA": "Georgia", "SHAONING": "Shaoxing–Ningbo"}
BASE = ["naive", "mask", "li2024", "coral", "dann", "groupdro", "mixstyle"]
NAME = {"naive": "Naive", "mask": "Masked", "li2024": "Li-style", "coral": "CORAL", "dann": "DANN",
        "groupdro": "GroupDRO", "mixstyle": "MixStyle", "ours": "Ours (hybrid)"}
ABBR = {c: v[0] for c, v in LABELS.items()}
GREY = "#9a9a9a"


def setup():
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 6.5,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight"})


def load():
    R = {}
    for f in glob.glob(os.path.join(ROOT, "results", "final", "*.json")):
        d = json.load(open(f))
        R[d["name"]] = d
    return R


def run(R, m, t, s):
    return R[("abl" if m.startswith("ours-") else "loso") + f"-{t}-{m}-s{s}"]


def fig_seen_seeds(R):
    meths = BASE + ["ours"]
    fig, axs = plt.subplots(1, 4, figsize=(17 * CM, 5.2 * CM), sharey=False)
    for ax, t in zip(axs, TARGETS):
        for i, m in enumerate(meths):
            v = [run(R, m, t, s)["test_seen"]["auroc"] * 100 for s in range(3)]
            col = OKABE[3] if m == "ours" else GREY
            ax.scatter([i] * 3, v, s=9, color=col, alpha=0.75, lw=0, zorder=3)
            ax.plot([i - 0.28, i + 0.28], [np.mean(v)] * 2, color=col, lw=1.4, zorder=4)
        best = max(np.mean([run(R, m, t, s)["test_seen"]["auroc"] * 100 for s in range(3)]) for m in BASE)
        ax.axhline(best, color=GREY, lw=0.5, ls="--", zorder=1)
        ax.set_xticks(range(len(meths)), [NAME[m].replace(" (hybrid)", "") for m in meths], rotation=60, ha="right")
        ax.set_title(TT[t])
        ax.grid(axis="y", lw=0.3, alpha=0.5)
    axs[0].set_ylabel("Macro AUROC, seen diagnoses (×100)")
    fig.tight_layout(w_pad=0.6)
    fig.savefig(os.path.join(OUT, "fig_seen_seeds.pdf"))
    plt.close(fig)


def fig_unseen(R):
    items = [("CPSC", c) for c in run(R, "ours", "CPSC", 0)["unseen"]] + \
            [("SHAONING", c) for c in run(R, "ours", "SHAONING", 0)["unseen"]]
    fig, ax = plt.subplots(figsize=(12 * CM, 5.5 * CM))
    for k, (t, c) in enumerate(items):
        a = ABBR[c]
        for j, m in enumerate(BASE):
            v = st.mean(run(R, m, t, s)["per_label"][a]["auroc"] for s in range(3)) * 100
            ax.scatter(k - 0.18 + 0.36 * j / (len(BASE) - 1), v, s=10, color=GREY, lw=0, zorder=3,
                       label="Baselines (linear head)" if k == 0 and j == 0 else None)
        v = [run(R, "ours", t, s)["per_label"][a]["auroc"] * 100 for s in range(3)]
        ax.bar(k, np.mean(v), 0.55, color=OKABE[3], alpha=0.85, zorder=2, label="Ours (text head)" if k == 0 else None)
        ax.errorbar(k, np.mean(v), yerr=np.std(v, ddof=1), color="k", lw=0.7, capsize=2, zorder=4)
        for var, mk, fc, lab in (("merl", "^", "k", "MERL zero-shot (own prompts)"),
                                 ("merl_desc", "^", "none", "MERL zero-shot (our descriptions)")):
            f = os.path.join(ROOT, "results", "merl", "results", f"{var}-{t}.json")
            if os.path.exists(f):
                mv = json.load(open(f))["per_label"].get(a, {}).get("auroc")
                if mv is not None:
                    ax.scatter(k + 0.36, mv * 100, marker=mk, s=16, facecolors=fc, edgecolors="k", lw=0.7, zorder=5,
                               label=lab if k == 0 else None)
        npos = run(R, "ours", t, 0)["per_label"][a]["n_pos"]
        ax.text(k, 3, f"n={npos}", ha="center", fontsize=5.5, color="white" if np.mean(v) > 10 else "k")
    ax.axhline(50, color="k", lw=0.6, ls=":")
    ax.set_xticks(range(len(items)), [f"{ABBR[c]}\n({'CPSC' if t == 'CPSC' else 'S–N'})" for t, c in items])
    ax.set_ylim(0, 100)
    ax.set_ylabel("AUROC (×100)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6)
    ax.grid(axis="y", lw=0.3, alpha=0.5)
    fig.savefig(os.path.join(OUT, "fig_unseen.pdf"))
    plt.close(fig)


def fig_e2(R):
    meths = [("naive", "Naive", OKABE[1], "s"), ("li2024", "Li-style", OKABE[0], "o"), ("ours", "Ours", OKABE[3], "D")]
    rates = ["0.0", "0.25", "0.5"]
    fig = plt.figure(figsize=(17 * CM, 5.6 * CM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 2.3], wspace=0.35)
    ax1, ax2, ax3 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    for m, lab, col, mk in meths:
        d = [R[f"e4-PTBXL-{m}r{r}-s0"] for r in rates]
        ax1.plot([0, 25, 50], [x["test_seen"]["auroc"] * 100 for x in d], marker=mk, ms=3.5, color=col, label=lab)
        ax2.plot([25, 50], [x["test_unseen"]["auroc"] * 100 for x in d[1:]], marker=mk, ms=3.5, color=col, label=lab)
    ax1.set_title("Retained diagnoses"); ax2.set_title("Removed diagnoses")
    for ax in (ax1, ax2):
        ax.set_xlabel("Annotations removed, $r$ (%)"); ax.grid(axis="y", lw=0.3, alpha=0.5)
    ax1.set_ylabel("Macro AUROC (×100)")
    ax2.axhline(50, color="k", lw=0.6, ls=":")
    ax2.set_xticks([25, 50]); ax2.set_xlim(15, 60); ax1.set_xticks([0, 25, 50])
    ax1.legend(frameon=False)
    d = R["e4-PTBXL-oursr0.5-s0"]
    drop = [ABBR[c] for s_, c in d["dropped"] if s_ == "PTBXL"]
    drop = [a for a in drop if a in d["per_label"]]
    x = np.arange(len(drop))
    for j, (m, lab, col, mk) in enumerate(meths):
        v = [R[f"e4-PTBXL-{m}r0.5-s0"]["per_label"][a]["auroc"] * 100 for a in drop]
        ax3.bar(x + (j - 1) * 0.27, v, 0.27, color=col, label=lab)
    ax3.axhline(50, color="k", lw=0.6, ls=":")
    ax3.set_xticks(x, drop, rotation=60, ha="right")
    ax3.set_title("Removed diagnoses at $r$ = 50%, per diagnosis")
    ax3.set_ylim(0, 100); ax3.grid(axis="y", lw=0.3, alpha=0.5)
    fig.savefig(os.path.join(OUT, "fig_e2.pdf"))
    plt.close(fig)


def fig_curves():
    def hist(pattern):
        f = glob.glob(os.path.join(ROOT, "results", "**", pattern), recursive=True)
        f = [x for x in f if os.sep + "final" + os.sep not in x]
        return [h["val_auroc"] * 100 for h in json.load(open(f[0]))["history"]] if f else None
    fig, axs = plt.subplots(1, 2, figsize=(17 * CM, 5.2 * CM), sharey=True)
    pairs = [(axs[0], [("loso-GEORGIA-ours-s0.json", "Text-only design v1 (EMA-teacher pseudo-labels)", OKABE[4]),
                       ("loso-GEORGIA-listyle-s0.json", "Li-style (no weight averaging)", OKABE[0])],
              "(a) Development: v1 collapse"),
             (axs[1], [("loso-GEORGIA-hyb-avg-s1.json", "Ours (hybrid, averaged weights)", OKABE[3]),
                       ("loso-GEORGIA-listyleema-s1.json", "Li-style (averaged weights)", OKABE[0]),
                       ("loso-GEORGIA-listyle-s1.json", "Li-style (no averaging)", GREY)],
              "(b) Final configuration")]
    for ax, curves, title in pairs:
        for pat, lab, col in curves:
            h = hist(pat)
            if h:
                ax.plot(range(1, len(h) + 1), h, color=col, lw=1.1, label=lab)
        ax.set_title(title, loc="left"); ax.set_xlabel("Epoch"); ax.grid(lw=0.3, alpha=0.5)
        ax.legend(frameon=False, loc="lower right")
    axs[0].set_ylabel("Validation macro AUROC (×100)\n(Georgia held out)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_curves.pdf"))
    plt.close(fig)


def fig_ema():
    """seed-0 runs with and without weight averaging (development runs: loso-<T>-<m>-s0 vs loso-<T>-<m>ema-s0)."""
    dev = {}
    for f in glob.glob(os.path.join(ROOT, "results", "kaggle", "**", "loso-*-s0.json"), recursive=True):
        d = json.load(open(f))
        dev[d["name"]] = d["test_seen"]["auroc"] * 100
    raw = {"naive": "naive", "mask": "mask", "li2024": "listyle", "coral": "coral", "dann": "dann",
           "groupdro": "groupdro", "mixstyle": "mixstyle"}
    rows = []
    for m, tag in raw.items():
        a = [dev.get(f"loso-{t}-{tag}-s0") for t in TARGETS]
        b = [dev.get(f"loso-{t}-{tag}ema-s0") for t in TARGETS]
        if None not in a and None not in b:
            rows.append((NAME[m], np.mean(a), np.mean(b)))
    if not rows:
        return rows
    fig, ax = plt.subplots(figsize=(10 * CM, 5 * CM))
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [r[1] for r in rows], 0.4, color=GREY, label="without averaging")
    ax.bar(x + 0.2, [r[2] for r in rows], 0.4, color=OKABE[0], label="with averaging")
    ax.set_xticks(x, [r[0] for r in rows], rotation=30, ha="right")
    ax.set_ylim(min(r[1] for r in rows) - 1, max(r[2] for r in rows) + 0.6)
    ax.set_ylabel("Mean macro AUROC, seen (×100)")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.grid(axis="y", lw=0.3, alpha=0.5)
    fig.savefig(os.path.join(OUT, "fig_ema.pdf"))
    plt.close(fig)
    return rows


if __name__ == "__main__":
    setup()
    os.makedirs(OUT, exist_ok=True)
    R = load()
    fig_seen_seeds(R)
    fig_unseen(R)
    fig_e2(R)
    fig_curves()
    for r in fig_ema():
        print(f"EMA {r[0]}: {r[1]:.2f} -> {r[2]:.2f}")
    print("figures written to", OUT)
