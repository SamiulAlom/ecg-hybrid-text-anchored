"""Data-description figures/tables computed from the preprocessing outputs (meta.csv, code_counts.csv).

Usage: python paper/make_data_figures.py --prep results/kaggle/prep/prep
Writes into paper/figures/ and paper/generated_data/ (tables as .tex, numbers as data_stats.json).
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from data import MAIN_SOURCES, build_vocab, label_matrices  # noqa: E402
from labels import LABELS, SCORED_26  # noqa: E402

NAMES = {"CPSC": "CPSC", "PTBXL": "PTB-XL", "GEORGIA": "Georgia", "SHAONING": "Shaoxing--Ningbo"}
COUNTRY = {"CPSC": "China", "PTBXL": "Germany", "GEORGIA": "USA", "SHAONING": "China"}
plt.rcParams.update({"font.size": 7, "font.family": "sans-serif", "axes.linewidth": 0.5})
W = 17 / 2.54


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    a = ap.parse_args()
    meta = pd.read_csv(os.path.join(a.prep, "meta.csv"), dtype={"dx": str})
    meta["dx"] = meta["dx"].fillna("")
    meta = meta[meta.source.isin(MAIN_SOURCES)].reset_index(drop=True)
    figdir, gdir = os.path.join(HERE, "figures"), os.path.join(HERE, "generated_data")
    os.makedirs(figdir, exist_ok=True); os.makedirs(gdir, exist_ok=True)

    vocab, annotated, cnt = build_vocab(meta)
    Y, A = label_matrices(meta, vocab, annotated)
    src = meta.source.values
    stats = {"n_records": int(len(meta)), "vocab_size": len(vocab), "per_source": {}}

    # ---- Table: databases
    rows = []
    for s in MAIN_SOURCES:
        m = meta[meta.source == s]
        dur = (m.n_orig / m.fs_orig)
        age = pd.to_numeric(m.age, errors="coerce")
        age = age[(age >= 0) & (age <= 110)]  # PTB-XL codes ages > 89 as 300; negative = missing
        female = m.sex.str.lower().str.startswith("f").mean() * 100
        k = len(annotated[s])
        stats["per_source"][s] = dict(records=int(len(m)), fs=sorted(m.fs_orig.unique().tolist()),
                                      dur_min=float(dur.min()), dur_median=float(dur.median()), dur_max=float(dur.max()),
                                      age_mean=float(age.mean()), age_sd=float(age.std()), female_pct=float(female),
                                      annotated=k, labels_per_record=float(Y[src == s].sum(1).mean()))
        fs = "/".join(f"{int(f)}" for f in sorted(m.fs_orig.unique()))
        rows.append(f"{NAMES[s]} & {COUNTRY[s]} & {len(m):,} & {dur.median():.0f} ({dur.min():.0f}--{dur.max():.0f}) & "
                    f"{age.mean():.1f} $\\pm$ {age.std():.1f} & {female:.1f} & {k} \\\\")
    with open(os.path.join(gdir, "tab_datasets.tex"), "w") as f:
        f.write("\n".join([
            r"\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{4pt}",
            r"\caption{Databases used in this study after excluding recordings shorter than 5\,s, flat recordings and duplicates (two recordings in total). All recordings were sampled at 500\,Hz. Duration in seconds, "
            r"median (range). Age excludes missing values and the PTB-XL anonymization code (300) used for patients older than 89 years. Annotated = number of vocabulary diagnoses with at least ten positive recordings in the database.}"
            r"\label{tab:datasets}",
            r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrcccr@{}}\toprule",
            r"Database & Country & Records & Duration (s) & Age (years) & Female (\%) & Annotated \\ \midrule",
            *rows,
            rf"\midrule Total & & {len(meta):,} & & & & {len(vocab)} \\",
            r"\bottomrule\end{tabular*}\end{table}", ""]))

    # ---- Heterogeneity statistics for the introduction
    scored = [c for c in SCORED_26 if c in vocab]
    all4 = [c for c in scored if all(c in annotated[s] for s in MAIN_SOURCES)]
    j_missing = [vocab.index(c) for c in vocab if not all(c in annotated[s] for s in MAIN_SOURCES)]
    pos = Y.sum()
    stats.update(n_vocab_annotated_by_all=sum(all(c in annotated[s] for s in MAIN_SOURCES) for c in vocab),
                 n_scored_in_vocab=len(scored), n_scored_annotated_by_all=len(all4),
                 pct_pos_in_partially_annotated=float(Y[:, j_missing].sum() / pos * 100),
                 pct_unknown_entries=float((1 - A).mean() * 100))

    # ---- Figure: annotation matrix (prevalence where annotated, hatched where not)
    labs = [c for c in SCORED_26 if c in vocab] + [c for c in vocab if c not in SCORED_26]
    P = np.full((len(MAIN_SOURCES), len(labs)), np.nan)
    for i, s in enumerate(MAIN_SOURCES):
        for j, c in enumerate(labs):
            if c in annotated[s]:
                P[i, j] = cnt[s][c] / (src == s).sum() * 100
    fig, ax = plt.subplots(figsize=(W, 0.32 * W))
    ax.set_facecolor("#dddddd")
    im = ax.imshow(np.log10(P), cmap="viridis", aspect="auto", vmin=-1.5, vmax=1.8)
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            if np.isnan(P[i, j]):
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, hatch="////", lw=0, color="#888888"))
    ax.set_xticks(range(len(labs)), [LABELS[c][0] for c in labs], rotation=90)
    ax.set_yticks(range(len(MAIN_SOURCES)), [NAMES[s].replace("--", "-") for s in MAIN_SOURCES])
    ax.axvline(len(scored) - .5, color="w", lw=1)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    ticks = [-1, 0, 1]
    cb.set_ticks(ticks); cb.set_ticklabels([f"{10**t:g}%" for t in ticks]); cb.set_label("prevalence")
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig_annotation_matrix.pdf"))
    fig.savefig(os.path.join(figdir, "fig_annotation_matrix.png"), dpi=300)

    # ---- Figure: label co-occurrence of annotation sets (how many databases annotate each diagnosis)
    nann = [sum(c in annotated[s] for s in MAIN_SOURCES) for c in labs]
    stats["n_databases_annotating"] = {LABELS[c][0]: n for c, n in zip(labs, nann)}

    # ---- Figure: age distributions
    fig, axs = plt.subplots(1, 2, figsize=(W, 0.3 * W))
    for s in MAIN_SOURCES:
        age = pd.to_numeric(meta.age[meta.source == s], errors="coerce").dropna()
        axs[0].hist(age[(age > 0) & (age < 110)], bins=40, histtype="step", density=True, label=NAMES[s].replace("--", "-"))
        k = Y[src == s][:, [vocab.index(c) for c in annotated[s]]].sum(1)
        axs[1].hist(k, bins=np.arange(0, 9) - .5, histtype="step", density=True)
    axs[0].set_xlabel("age (years)"); axs[0].set_ylabel("density"); axs[0].legend(frameon=False)
    axs[1].set_xlabel("annotated diagnoses per record"); axs[1].set_ylabel("fraction of records")
    for x in axs:
        x.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig_demographics.pdf"))

    with open(os.path.join(gdir, "data_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps({k: v for k, v in stats.items() if k != "n_databases_annotating"}, indent=1))


if __name__ == "__main__":
    main()
