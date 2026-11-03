"""E3 systematic zero-shot analysis and annotation-threshold sensitivity.

Inputs: e3-<T>-ours-g<k>-s<seed>.json and thr<5|20>-<T>-<ours|li2024>-s0.json (any folder under results/),
results/final (supervised reference) and, if present, MERL outputs merl*-<T>.json (results/merl).
Outputs in paper/generated:
  e3_pairs.csv          one row per (held-out database, diagnosis): zero-shot AUROC (mean over seeds), supervised
                        AUROC of the same model when the diagnosis is trained, MERL AUROC (3 prompt variants),
                        nearest-anchor cosine to a trained diagnosis, positives
  tab_e3.tex            summary table
  fig_e3.pdf            (a) zero-shot vs supervised AUROC per pair, (b) zero-shot AUROC vs nearest-anchor cosine
  tab_threshold.tex     annotation-threshold sensitivity
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
from scipy.stats import spearmanr, wilcoxon  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from labels import LABELS  # noqa: E402

OUT = os.path.join(ROOT, "paper", "generated")
TARGETS = ["CPSC", "PTBXL", "GEORGIA", "SHAONING"]
TT = {"CPSC": "CPSC", "PTBXL": "PTB-XL", "GEORGIA": "Georgia", "SHAONING": "S--N"}
ABBR = {c: v[0] for c, v in LABELS.items()}


def load(pattern):
    R = {}
    for f in glob.glob(os.path.join(ROOT, "results", "**", pattern), recursive=True):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if isinstance(d, dict) and "name" in d:
            R.setdefault(d["name"], d)
    return R


def main():
    e3 = load("e3-*.json")
    fin = load("loso-*-ours-s*.json")
    fin = {k: v for k, v in fin.items() if "final" in json.dumps(v.get("config", {})) or True}
    final = {}
    for f in glob.glob(os.path.join(ROOT, "results", "final", "loso-*-ours-s*.json")):
        d = json.load(open(f))
        final[d["name"]] = d
    merl = {}
    for f in glob.glob(os.path.join(ROOT, "results", "merl", "**", "merl*.json"), recursive=True):
        d = json.load(open(f))
        if "per_label" in d:
            merl[os.path.basename(f)[:-5]] = d
    z = np.load(os.path.join(ROOT, "src", "text_emb_biolord_full.npz"))
    emb = {c: v / np.linalg.norm(v) for c, v in zip(z["codes"].tolist(), z["emb"])}
    groups = json.load(open(os.path.join(ROOT, "kaggle", "zeroshot_groups.json")))

    rows = []
    for t in TARGETS:
        for gi, g in enumerate(groups[t]):
            runs = [e3.get(f"e3-{t}-ours-g{gi}-s{s}") for s in range(3)]
            runs = [r for r in runs if r]
            if not runs:
                continue
            trained = [c for c in runs[0]["seen"]]
            for c in g:
                a = ABBR[c]
                zs = [r["per_label"][a]["auroc"] for r in runs if a in r["per_label"]]
                if not zs:
                    continue
                sup = [final[f"loso-{t}-ours-s{s}"]["per_label"][a]["auroc"] for s in range(3)
                       if a in final.get(f"loso-{t}-ours-s{s}", {}).get("per_label", {})]
                sims = [float(emb[c] @ emb[o]) for o in trained if o in emb and o != c]
                nearest = max(sims) if sims else float("nan")
                nn = trained[int(np.argmax(sims))] if sims else ""
                mv = {}
                for var in ["merl", "merl_desc", "merl_name"]:
                    d = merl.get(f"{var}-{t}")
                    if d and a in d["per_label"]:
                        mv[var] = d["per_label"][a]["auroc"]
                rows.append(dict(target=t, code=c, abbr=a, group=gi, n_seeds=len(zs), zs=st.mean(zs),
                                 zs_std=st.stdev(zs) if len(zs) > 1 else float("nan"),
                                 sup=st.mean(sup) if sup else float("nan"), nearest=nearest,
                                 nearest_abbr=ABBR.get(nn, ""), n_pos=runs[0]["per_label"][a]["n_pos"], **mv))
    if not rows:
        print("no E3 runs found")
        return
    keys = ["target", "abbr", "code", "group", "n_seeds", "n_pos", "zs", "zs_std", "sup", "nearest", "nearest_abbr",
            "merl", "merl_desc", "merl_name"]
    with open(os.path.join(OUT, "e3_pairs.csv"), "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
    zs = np.array([r["zs"] for r in rows])
    sup = np.array([r["sup"] for r in rows])
    near = np.array([r["nearest"] for r in rows])
    print(f"pairs {len(rows)}  zero-shot macro AUROC {zs.mean():.4f}  median {np.median(zs):.4f}  "
          f">0.5: {np.mean(zs > 0.5):.2f}  >0.7: {np.mean(zs > 0.7):.2f}  >0.8: {np.mean(zs > 0.8):.2f}")
    ok = ~np.isnan(sup)
    print(f"supervised macro {sup[ok].mean():.4f}  retained fraction of above-chance AUROC "
          f"{np.mean((zs[ok] - 0.5) / (sup[ok] - 0.5)):.3f}")
    rho, p_rho = spearmanr(near, zs)
    print(f"Spearman(nearest-anchor cosine, zero-shot AUROC) rho={rho:.3f} p={p_rho:.2g}")
    p_chance = wilcoxon(zs - 0.5).pvalue
    print(f"Wilcoxon zero-shot vs 0.5: p={p_chance:.2g}")
    summ = dict(n_pairs=len(rows), zs_mean=zs.mean(), zs_median=float(np.median(zs)), frac50=np.mean(zs > 0.5),
                frac70=np.mean(zs > 0.7), frac80=np.mean(zs > 0.8), sup_mean=sup[ok].mean(), rho=rho, p_rho=p_rho,
                p_chance=p_chance)
    for var in ["merl", "merl_desc", "merl_name"]:
        mv = np.array([r.get(var, np.nan) for r in rows], dtype=float)
        k = ~np.isnan(mv)
        if k.sum():
            pw = wilcoxon(zs[k], mv[k]).pvalue
            summ[var] = (mv[k].mean(), int(k.sum()), zs[k].mean(), pw, int(np.sum(zs[k] > mv[k])))
            print(f"{var}: n={k.sum()} MERL {mv[k].mean():.4f} vs ours {zs[k].mean():.4f} "
                  f"(ours better in {np.sum(zs[k] > mv[k])}/{k.sum()}), Wilcoxon p={pw:.2g}")
    json.dump({k: (list(v) if isinstance(v, tuple) else float(v)) for k, v in summ.items()},
              open(os.path.join(OUT, "e3_summary.json"), "w"), indent=1)
    # per-target summary table
    lines = [r"\begin{table}[!htbp]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{4pt}",
             r"\caption{Systematic zero-shot evaluation (E3). In every held-out database, each diagnosis that would "
             r"otherwise be seen was withheld from training in one of 4--6 groups and scored by the text head; "
             r"AUROC ($\times100$) is averaged over three seeds and then over diagnoses. Supervised: the same "
             r"diagnoses scored by the proposed method when they are trained. Pairs: number of (database, "
             r"diagnosis) pairs.}\label{tab:e3}",
             r"\begin{tabular}{lcccccc}", r"\toprule",
             r"Held-out database & Pairs & Zero-shot & Supervised & $>$50 & $>$70 & MERL \\", r"\midrule"]
    for t in TARGETS + ["All"]:
        sel = [r for r in rows if t == "All" or r["target"] == t]
        if not sel:
            continue
        z_ = np.array([r["zs"] for r in sel]); s_ = np.array([r["sup"] for r in sel])
        m_ = [r.get("merl") for r in sel if r.get("merl") is not None]
        if t == "All":
            lines.append(r"\midrule")
        lines.append(" & ".join([TT.get(t, "All"), str(len(sel)), f"{z_.mean() * 100:.1f}",
                                 f"{np.nanmean(s_) * 100:.1f}", f"{np.mean(z_ > 0.5) * 100:.0f}\\%",
                                 f"{np.mean(z_ > 0.7) * 100:.0f}\\%",
                                 f"{np.mean(m_) * 100:.1f}" if m_ else "--"]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(OUT, "tab_e3.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    # figure
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axs = plt.subplots(1, 2, figsize=(17 / 2.54, 6.2 / 2.54))
    colors = {"CPSC": "#0072B2", "PTBXL": "#E69F00", "GEORGIA": "#009E73", "SHAONING": "#D55E00"}
    for t in TARGETS:
        sel = [r for r in rows if r["target"] == t]
        axs[0].scatter([r["sup"] * 100 for r in sel], [r["zs"] * 100 for r in sel], s=9, color=colors[t],
                       label=TT[t].replace("--", "–"), lw=0, alpha=0.85)
        axs[1].scatter([r["nearest"] for r in sel], [r["zs"] * 100 for r in sel], s=9, color=colors[t], lw=0,
                       alpha=0.85)
    axs[0].plot([40, 100], [40, 100], ls=":", color="k", lw=0.6)
    axs[0].axhline(50, color="grey", lw=0.5, ls="--")
    axs[0].set_xlabel("Supervised AUROC (×100)"); axs[0].set_ylabel("Zero-shot AUROC (×100)")
    axs[0].set_title("(a) Zero-shot vs supervised", loc="left"); axs[0].legend(frameon=False, fontsize=6)
    axs[1].axhline(50, color="grey", lw=0.5, ls="--")
    axs[1].set_xlabel("Cosine to nearest trained diagnosis anchor")
    axs[1].set_title(f"(b) Spearman ρ = {rho:.2f}", loc="left")
    for ax in axs:
        ax.grid(lw=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_e3.pdf"))
    threshold()


def threshold():
    thr = load("thr*.json")
    if not thr:
        print("no threshold runs")
        return
    fin = {}
    for f in glob.glob(os.path.join(ROOT, "results", "final", "loso-*-s0.json")):
        d = json.load(open(f))
        fin[d["name"]] = d
    lines = [r"\begin{table}[!htbp]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{4pt}",
             r"\caption{Sensitivity to the annotation threshold (minimum positives for a database to count as "
             r"annotating a diagnosis): seen-diagnosis macro AUROC ($\times100$) per held-out database and mean, "
             r"seed 0, both methods with weight averaging; $C$ is the vocabulary size.}\label{tab:threshold}",
             r"\begin{tabular}{llcccccc}", r"\toprule",
             r"Threshold & Method & $C$ & CPSC & PTB-XL & Georgia & S--N & Mean \\", r"\midrule"]
    for ms in (5, 10, 20):
        for m, nm in (("li2024", "Li-style"), ("ours", "Ours")):
            vals, C = [], ""
            for t in TARGETS:
                d = fin.get(f"loso-{t}-{m}-s0") if ms == 10 else thr.get(f"thr{ms}-{t}-{m}-s0")
                vals.append(d["test_seen"]["auroc"] * 100 if d else None)
                if d:
                    C = len(d["vocab"])
            cells = [f"{v:.1f}" if v is not None else "--" for v in vals]
            mean = f"{np.mean(vals):.2f}" if None not in vals else "--"
            lines.append(" & ".join([str(ms) + (" (default)" if ms == 10 else ""), nm, str(C)] + cells + [mean])
                         + r" \\")
            print("thr", ms, m, C, cells, mean)
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]
    open(os.path.join(OUT, "tab_threshold.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
