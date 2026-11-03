"""Seed-level and supplementary statistics for the final runs (results/final), written to paper/generated.

* seed_stats.csv: ours vs each baseline over the 12 (target, seed) pairs of macro AUROC/AUPRC on seen labels:
  mean difference, wins, exact two-sided Wilcoxon signed-rank p, Holm-corrected over baselines (per metric).
* unseen.csv: macro AUROC on labels never annotated in training (Shaoxing-Ningbo target), mean/std over seeds.
* e2.csv: annotation-removal experiment on PTB-XL (seed 0): AUROC on retained (seen) and removed labels.
"""
import glob
import json
import os
import re
import statistics as st
from collections import defaultdict

from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, OUT = os.path.join(ROOT, "results", "final"), os.path.join(ROOT, "paper", "generated")
BASE = ["naive", "mask", "li2024", "coral", "dann", "groupdro", "mixstyle"]


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def main():
    runs = defaultdict(dict)
    e2 = []
    for f in glob.glob(os.path.join(RES, "*.json")):
        d = json.load(open(f))
        m = re.match(r"^(loso|abl|e4)-(\w+?)-(.+)-s(\d)$", d["name"])
        if m[1] == "e4":
            e2.append(d)
        else:
            runs[m[3]][(m[2], m[4])] = d
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "seed_stats.csv"), "w") as fo:
        fo.write("metric,baseline,n_pairs,mean_ours,mean_baseline,mean_diff,wins,p_wilcoxon,p_holm\n")
        for met in ["auroc", "auprc"]:
            rows = []
            for b in BASE:
                ks = sorted(k for k in runs["ours"] if k in runs[b])
                o = [runs["ours"][k]["test_seen"][met] for k in ks]
                x = [runs[b][k]["test_seen"][met] for k in ks]
                dif = [a - c for a, c in zip(o, x)]
                p = wilcoxon(o, x, method="exact").pvalue
                rows.append([met, b, len(ks), st.mean(o), st.mean(x), st.mean(dif), sum(v > 0 for v in dif), p])
            for r, a in zip(rows, holm([r[-1] for r in rows])):
                fo.write(",".join(str(v) if not isinstance(v, float) else f"{v:.6g}" for v in r + [a]) + "\n")
                print(*[f"{v:.4f}" if isinstance(v, float) else v for v in r + [a]])
    with open(os.path.join(OUT, "unseen.csv"), "w") as fo:
        fo.write("method,target,n_seeds,auroc_unseen_mean,auroc_unseen_std,n_labels\n")
        for meth in sorted(runs):
            for tgt in ["CPSC", "SHAONING"]:
                ds = [d for (t, s), d in runs[meth].items()
                      if t == tgt and d.get("test_unseen") and d["test_unseen"].get("auroc") is not None]
                if ds:
                    v = [d["test_unseen"]["auroc"] for d in ds]
                    fo.write(f"{meth},{tgt},{len(v)},{st.mean(v):.4f},"
                             f"{st.stdev(v) if len(v) > 1 else float('nan'):.4f},{ds[0]['test_unseen']['n_labels']}\n")
                    print("unseen", meth, tgt, len(v), round(st.mean(v), 4))
    with open(os.path.join(OUT, "e2.csv"), "w") as fo:
        fo.write("method,drop_rate,auroc_seen,auroc_removed,auroc_all\n")
        for d in sorted(e2, key=lambda d: d["name"]):
            m = re.match(r"^e4-\w+?-(.+?)r([\d.]+)-s\d$", d["name"])
            u = (d.get("test_unseen") or {}).get("auroc")
            fo.write(f"{m[1]},{m[2]},{d['test_seen']['auroc']:.4f},{'' if u is None else f'{u:.4f}'},"
                     f"{d['test_all']['auroc']:.4f}\n")


if __name__ == "__main__":
    main()
