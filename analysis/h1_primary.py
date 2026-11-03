"""Pre-specified variant (H1, linear scoring) as the primary seen-diagnosis comparison, with the fused model (H2) as a
secondary analysis, plus database-level effects (seed means per held-out database) -> paper/generated/h1_stats.csv,
paper/generated/db_effects.csv. Seeds share the same four test databases, so the 12 seed-level pairs are not
independent; the database-level table reports the effect per held-out database (n=4) without a p-value."""
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
DBS = ["CPSC", "PTBXL", "GEORGIA", "SHAONING"]
OURS = {"H1": "ours-hyblinear", "H2": "ours"}


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


runs = defaultdict(dict)
for f in glob.glob(os.path.join(RES, "*.json")):
    d = json.load(open(f))
    m = re.match(r"^(loso|abl)-(\w+?)-(.+)-s(\d)$", d["name"])
    if m:
        runs[m[3]][(m[2], m[4])] = d

with open(os.path.join(OUT, "h1_stats.csv"), "w") as fo:
    fo.write("variant,metric,baseline,n_pairs,mean_ours,mean_baseline,mean_diff,wins,p_wilcoxon,p_holm\n")
    for var, key in OURS.items():
        for met in ["auroc", "auprc"]:
            rows = []
            for b in BASE:
                ks = sorted(k for k in runs[key] if k in runs[b])
                o = [runs[key][k]["test_seen"][met] for k in ks]
                x = [runs[b][k]["test_seen"][met] for k in ks]
                dif = [a - c for a, c in zip(o, x)]
                p = wilcoxon(o, x, method="exact").pvalue
                rows.append([var, met, b, len(ks), st.mean(o), st.mean(x), st.mean(dif), sum(v > 0 for v in dif), p])
            for r, a in zip(rows, holm([r[-1] for r in rows])):
                fo.write(",".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in r + [a]) + "\n")
                print(*[f"{v:.4f}" if isinstance(v, float) else v for v in r + [a]])

with open(os.path.join(OUT, "db_effects.csv"), "w") as fo:
    fo.write("method,metric," + ",".join(DBS) + ",mean\n")
    for meth in ["H1", "H2"] + BASE:
        key = OURS.get(meth, meth)
        for met in ["auroc", "auprc"]:
            v = [st.mean(runs[key][(t, s)]["test_seen"][met] for s in "012" if (t, s) in runs[key]) for t in DBS]
            fo.write(f"{meth},{met}," + ",".join(f"{x:.4f}" for x in v) + f",{st.mean(v):.4f}\n")
            print(meth, met, [round(x * 100, 2) for x in v], round(st.mean(v) * 100, 2))
    for meth in ["H1", "H2"]:
        u = [runs[OURS[meth]][("SHAONING", s)]["test_unseen"]["auroc"] for s in "012"]
        print("unseen S-N", meth, round(st.mean(u) * 100, 2), round(st.stdev(u) * 100, 2))
