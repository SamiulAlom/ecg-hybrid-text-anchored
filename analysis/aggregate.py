"""Aggregate train.py runs into the paper's tables, statistics and figures.

Every number written here is read from (or recomputed from) a run file; nothing is imputed.
Missing runs are skipped with a warning and shown as "--".

Usage:
  .venv/Scripts/python analysis/aggregate.py --results results/kaggle --out paper/generated

Run classification (tag = json "name", which is the --tag given to train.py):
  <prefix>-<TARGET>-<method>-s<seed>, TARGET in {CPSC, PTBXL, GEORGIA, SHAONING}
  * main LOSO table : prefix "loso", config.drop_rate == 0, no holdout, method not an ablation variant
  * ablation        : prefix "abl", or prefix "loso" with method "ours-<variant>" / "ours_<variant>"
  * E4 (drop rate)  : config.drop_rate > 0 (any prefix), or prefix "e4"
  * open-vocabulary : config.holdout non-empty (used only for the unseen-label table)
  * external zero-shot baseline: merl-<TARGET>_test.npz (+ merl-<TARGET>.json with "vocab")
"""
import argparse
import json
import math
import os
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import norm, rankdata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
try:
    sys.path.insert(0, str(SRC))
    from labels import LABELS  # noqa: E402
except Exception:  # labels.py is only used for abbreviations
    LABELS = {}
finally:
    if sys.path and sys.path[0] == str(SRC):
        sys.path.pop(0)

TARGETS = ["CPSC", "PTBXL", "GEORGIA", "SHAONING"]
TARGET_TEX = {"CPSC": "CPSC", "PTBXL": "PTB-XL", "GEORGIA": "Georgia", "SHAONING": "Shaoxing-Ningbo"}
METHOD_ORDER = ["naive", "mask", "li2024", "coral", "dann", "groupdro", "mixstyle", "ours"]
METHOD_TEX = {"naive": "Naive (unknown = negative)", "mask": "Masked loss", "li2024": "Hierarchical mask (Li et al.)",
              "coral": "CORAL", "dann": "DANN", "groupdro": "GroupDRO", "mixstyle": "MixStyle",
              "ours": "Ours", "merl": "MERL (zero-shot)"}
OURS = "ours"
TAG_RE = re.compile(r"^(?P<prefix>[A-Za-z0-9]+)-(?P<target>" + "|".join(TARGETS) + r")-(?P<method>.+)-s(?P<seed>\d+)$")
MIN_POS = 5
# Okabe-Ito colour-blind-safe palette
OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
CM = 1 / 2.54


_WARNED = []


def warn(msg):
    if msg not in _WARNED:  # each distinct warning once; all are also listed in summary.md
        _WARNED.append(msg)
        print(f"WARNING: {msg}", file=sys.stderr)


def abbr(code):
    return LABELS[code][0] if code in LABELS else str(code)


def tex_escape(s):
    rep = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
           "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(rep.get(c, c) for c in str(s))


def method_tex(m):
    return METHOD_TEX.get(m, tex_escape(m))


# ----------------------------------------------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------------------------------------------
class Run:
    def __init__(self, path, d):
        self.json_path = Path(path)
        self.npz_path = self.json_path.with_name(self.json_path.stem + "_test.npz")
        self.d = d
        self.cfg = d.get("config", {})
        self.name = d.get("name", self.json_path.stem)
        m = TAG_RE.match(self.name)
        self.prefix, self.method, self.seed = (m["prefix"], m["method"], int(m["seed"])) if m else (None, None, None)
        self.target = self.cfg.get("target") or (m["target"] if m else None)
        if m and m["target"] != self.cfg.get("target", m["target"]):
            warn(f"{self.json_path.name}: tag target {m['target']} != config.target {self.cfg.get('target')}")
        if m and self.cfg.get("seed") is not None and int(self.cfg["seed"]) != self.seed:
            warn(f"{self.json_path.name}: tag seed {self.seed} != config.seed {self.cfg['seed']}; using config")
            self.seed = int(self.cfg["seed"])
        self.vocab = list(d.get("vocab", []))
        self._npz = None

    @property
    def file(self):
        return self.json_path.name

    @property
    def drop_rate(self):
        return float(self.cfg.get("drop_rate") or 0.0)

    @property
    def kind(self):
        if self.prefix is None:
            return "other"
        if self.prefix.lower() == "e4" or self.drop_rate > 0:  # E2 in the paper (indist mode on PTB-XL)
            return "e4"
        if self.cfg.get("mode", "loso") != "loso":
            return "other"
        if self.cfg.get("holdout"):
            return "openvocab"
        if self.prefix.lower() == "abl" or re.match(r"^(ours[-_].+|abl[-_].+)", self.method):
            return "ablation"
        if self.prefix.lower() == "loso":
            return "main"
        return "other"

    def npz(self):
        if self._npz is None:
            if not self.npz_path.exists():
                return None
            z = np.load(self.npz_path)
            self._npz = dict(P=z["P"].astype(np.float64), Y=z["Y"].astype(np.uint8), A=z["A"].astype(np.uint8),
                             idx=z["idx"] if "idx" in z else None)
        return self._npz

    def metric(self, split, key):
        v = (self.d.get(split) or {}).get(key)
        return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)

    def signature(self):
        """Config minus things that vary within an experiment (used to pair E4 runs with drop-rate-0 runs)."""
        skip = {"seed", "tag", "out", "prep", "text_emb", "drop_rate", "target", "split_seed", "max_train"}
        return tuple(sorted((k, json.dumps(v, sort_keys=True)) for k, v in self.cfg.items() if k not in skip))


class External:
    """Zero-shot baseline from merl-<TARGET>_test.npz (+ merl-<TARGET>.json with vocab)."""

    def __init__(self, npz_path, target, vocab):
        self.npz_path, self.target, self.vocab, self.method = Path(npz_path), target, vocab, "merl"
        self.file = self.npz_path.name
        z = np.load(npz_path)
        self._npz = dict(P=z["P"].astype(np.float64), Y=z["Y"].astype(np.uint8), A=z["A"].astype(np.uint8),
                         idx=z["idx"] if "idx" in z else None)

    def npz(self):
        return self._npz


def load_runs(folder):
    folder = Path(folder)
    runs, externals = [], {}
    if not folder.exists():
        warn(f"results folder {folder} does not exist")
        return runs, externals
    for jp in sorted(folder.glob("*.json")):
        if jp.stem.startswith("merl-"):
            continue
        try:
            d = json.load(open(jp))
        except Exception as e:
            warn(f"cannot read {jp.name}: {e}; skipped")
            continue
        if "config" not in d or "vocab" not in d:
            warn(f"{jp.name} is not a train.py result (no config/vocab); skipped")
            continue
        r = Run(jp, d)
        if r.prefix is None:
            warn(f"{jp.name}: tag '{r.name}' does not match <prefix>-<TARGET>-<method>-s<seed>; skipped")
            continue
        if not r.npz_path.exists():
            warn(f"{jp.name}: missing {r.npz_path.name} (json metrics still used; npz-based analyses skip it)")
        runs.append(r)
    for t in TARGETS:
        zp, jp = folder / f"merl-{t}_test.npz", folder / f"merl-{t}.json"
        if zp.exists():
            if not jp.exists():
                warn(f"{zp.name} present but {jp.name} (vocab) missing; skipped")
                continue
            externals[t] = External(zp, t, list(json.load(open(jp))["vocab"]))
    # duplicates (same tag twice is impossible in one folder, but same kind/target/method/seed with other prefix is)
    seen = {}
    for r in runs:
        key = (r.kind, r.target, r.method, r.seed, r.drop_rate)
        if key in seen:
            warn(f"duplicate run {key}: {seen[key]} and {r.file}; both kept for E4/other, main uses the first")
        seen.setdefault(key, r.file)
    return runs, externals


# ----------------------------------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------------------------------
def auroc(y, p):
    """Mann-Whitney AUROC with midranks (identical to sklearn.roc_auc_score)."""
    n1 = int(y.sum()); n0 = len(y) - n1
    r = rankdata(p)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def eligible(Y, A, j):
    m = A[:, j] > 0
    npos = int(Y[m, j].sum())
    return m.sum() > 0 and npos >= MIN_POS and npos < m.sum()


def weighted_auc_boot(p, y, W):
    """AUROC of (p, y) for every bootstrap weight row of W [B, n] (multiplicities). NaN if a class is absent."""
    o = np.argsort(p, kind="mergesort")
    ps, ys = p[o], y[o].astype(bool)
    starts = np.r_[0, np.flatnonzero(np.diff(ps)) + 1]  # tie groups
    Ws = W[:, o]
    gp = np.add.reduceat(Ws * ys, starts, axis=1)
    gn = np.add.reduceat(Ws * ~ys, starts, axis=1)
    cn = np.cumsum(gn, axis=1) - gn
    num = (gp * (cn + 0.5 * gn)).sum(1)
    den = gp.sum(1) * gn.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.maximum(den, 1), np.nan)


def bootstrap_weights(n, B, seed, chunk=100):
    rng = np.random.default_rng(seed)
    for s in range(0, B, chunk):
        b = min(chunk, B - s)
        yield np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(b)]).astype(np.float64)


def fast_delong(p1, p2, y):
    """Paired DeLong test, fast O(n log n) algorithm of Sun & Xu (2014). Returns auc1, auc2, z, p."""
    y = y.astype(bool)
    X = np.vstack([p1, p2])
    pos, neg = X[:, y], X[:, ~y]
    m, n = pos.shape[1], neg.shape[1]
    tx = np.vstack([rankdata(r) for r in pos])
    ty = np.vstack([rankdata(r) for r in neg])
    tz = np.vstack([rankdata(r) for r in np.hstack([pos, neg])])
    aucs = tz[:, :m].sum(1) / m / n - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    S = np.cov(v01) / m + np.cov(v10) / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    d = aucs[0] - aucs[1]
    if var <= 1e-15:
        return aucs[0], aucs[1], 0.0, 1.0
    z = d / math.sqrt(var)
    return aucs[0], aucs[1], z, float(2 * norm.sf(abs(z)))


def holm(pvals):
    p = np.asarray(pvals, float)
    if p.size == 0:
        return p
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for k, i in enumerate(o):
        run = max(run, (len(p) - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def mean_std(v):
    v = [x for x in v if x is not None]
    if not v:
        return None, None, 0
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else None), len(v)


# ----------------------------------------------------------------------------------------------------------------------
# Traceability log
# ----------------------------------------------------------------------------------------------------------------------
class Trace:
    def __init__(self):
        self.sections = defaultdict(list)  # section -> rows (item, value, sources)

    def add(self, section, item, value, sources):
        if isinstance(value, float):
            value = f"{value:.4f}"
        self.sections[section].append((item, value, ", ".join(sorted(set(sources))) if sources else "-"))

    def write(self, path, header):
        with open(path, "w", encoding="utf-8") as f:
            f.write(header)
            for sec, rows in self.sections.items():
                f.write(f"\n## {sec}\n\n| item | value | source run file(s) |\n|---|---|---|\n")
                for it, v, s in rows:
                    f.write(f"| {it} | {v} | {s} |\n")


# ----------------------------------------------------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------------------------------------------------
def write_tex(path, lines):
    """Write a table. In table* floats the tabular is scaled down only if wider than the text block (graphicx
    resizebox). Plain `table` floats are left alone: sn-jnl wraps them in threeparttable, which breaks resizebox."""
    text = "\n".join(lines)
    if r"\begin{table*}" in text:
        text = text.replace(r"\begin{tabular}",
                        r"\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%" + "\n" + r"\begin{tabular}", 1)
        text = text.replace(r"\end{tabular}", r"\end{tabular}}", 1)
    Path(path).write_text(text, encoding="utf-8")


def fmt_cell(mean, std, bold, scale=100, nd=1):
    if mean is None:
        return "--"
    s = f"{mean * scale:.{nd}f}"
    s = rf"\textbf{{{s}}}" if bold else s
    if std is not None:
        s += rf"$_{{\pm {std * scale:.{nd}f}}}$"
    return s


def ordered_methods(methods):
    known = [m for m in METHOD_ORDER if m in methods and m != OURS]
    other = sorted(m for m in methods if m not in METHOD_ORDER)
    return known + other + ([OURS] if OURS in methods else [])


def main_table(runs, metrics, metric_tex, fname, caption, label, out, trace, targets):
    """rows = methods; cols = targets x metrics + mean over targets per metric."""
    main = [r for r in runs if r.kind == "main"]
    by = defaultdict(dict)  # (method) -> {(target, seed): run}
    for r in main:
        by[r.method].setdefault((r.target, r.seed), r)
    methods = ordered_methods(by)
    if not methods:
        warn(f"{fname}: no main LOSO runs; table not written")
        return {}
    cells = {}  # (method, col) -> (mean, std, n)
    cols = [(t, k) for t in targets for k in metrics] + [("MEAN", k) for k in metrics]
    sec = f"{fname} (test_seen macro; values x100 in the .tex)"
    for m in methods:
        runs_m = by[m]
        for t in targets:
            for k in metrics:
                rs = [r for (tt, s), r in sorted(runs_m.items()) if tt == t and r.metric("test_seen", k) is not None]
                vals = [r.metric("test_seen", k) for r in rs]
                mu, sd, n = mean_std(vals)
                cells[(m, (t, k))] = (mu, sd, n)
                for r in rs:
                    trace.add(sec, f"{m} / {t} / {k} / seed {r.seed}", r.metric("test_seen", k), [r.file])
                if mu is not None:
                    trace.add(sec, f"{m} / {t} / {k} / mean over {n} seed(s)", mu, [r.file for r in rs])
                else:
                    warn(f"{fname}: no run for method={m} target={t}")
        seeds = sorted({s for (_, s) in runs_m})
        for k in metrics:
            per_seed, files = [], []
            for s in seeds:
                rs = [runs_m.get((t, s)) for t in targets]
                if all(r is not None and r.metric("test_seen", k) is not None for r in rs):
                    per_seed.append(float(np.mean([r.metric("test_seen", k) for r in rs])))
                    files += [r.file for r in rs]
            mu, sd, n = mean_std(per_seed)
            cells[(m, ("MEAN", k))] = (mu, sd, n)
            if mu is not None:
                trace.add(sec, f"{m} / mean over targets / {k} ({n} seed(s) with all targets)", mu, files)
            else:
                warn(f"{fname}: method={m} has no seed with all {len(targets)} targets; mean column '--'")
    best = {}
    for c in cols:
        vals = [cells[(m, c)][0] for m in methods if cells[(m, c)][0] is not None]
        best[c] = max(vals) if vals else None
    ncol = len(cols)
    lines = [r"\begin{table*}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\begin{tabular}{l" + "c" * ncol + "}", r"\toprule"]
    head1 = " & ".join([""] + [rf"\multicolumn{{{len(metrics)}}}{{c}}{{{TARGET_TEX.get(t, t)}}}" for t in targets]
                       + [rf"\multicolumn{{{len(metrics)}}}{{c}}{{Mean}}"]) + r" \\"
    rules = " ".join(rf"\cmidrule(lr){{{2 + i * len(metrics)}-{1 + (i + 1) * len(metrics)}}}"
                     for i in range(len(targets) + 1))
    head2 = " & ".join(["Method"] + [metric_tex[k] for _ in range(len(targets) + 1) for k in metrics]) + r" \\"
    lines += [head1, rules, head2, r"\midrule"]
    for m in methods:
        if m == OURS and len(methods) > 1:
            lines.append(r"\midrule")
        row = [method_tex(m)]
        for c in cols:
            mu, sd, _ = cells[(m, c)]
            row.append(fmt_cell(mu, sd, mu is not None and best[c] is not None and np.isclose(mu, best[c])))
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    write_tex(out / fname, lines)
    print(f"wrote {out / fname}")
    return cells


def unseen_table(runs, externals, out, trace, targets):
    """Macro AUROC on unseen labels for every main/ablation run (+ external zero-shot baseline). Linear-head runs
    have an untrained output for unseen labels and are listed as a chance-level reference."""
    sec = "tab_unseen.tex (test_unseen macro AUROC; values x100 in the .tex)"
    rows = defaultdict(lambda: defaultdict(list))  # rowname -> target -> [(auroc, n_labels, file)]
    for r in runs:
        if r.kind not in ("main", "openvocab", "ablation"):
            continue
        v, n = r.metric("test_unseen", "auroc"), (r.d.get("test_unseen") or {}).get("n_labels")
        if v is None:
            warn(f"tab_unseen: {r.file} has no evaluable unseen label (test_unseen empty); skipped")
            continue
        row = r.method + (" (held-out codes)" if r.kind == "openvocab" else "")
        rows[row][r.target].append((v, int(n), r.file))
    # external baseline: evaluated on the unseen codes of the reference run (ours seed 0, main; else any text run)
    for t, ext in externals.items():
        cand = [r for r in runs if r.target == t and r.cfg.get("head") in ("text", "query") and r.d.get("unseen")]
        cand.sort(key=lambda r: (r.method != OURS, r.kind != "main", r.seed))
        if not cand:
            warn(f"tab_unseen: no text/query run for {t} defines an unseen set; {ext.file} skipped")
            continue
        ref = cand[0]
        z = ext.npz()
        cols = [ext.vocab.index(c) for c in ref.d["unseen"] if c in ext.vocab]
        missing = [c for c in ref.d["unseen"] if c not in ext.vocab]
        if missing:
            warn(f"tab_unseen: {ext.file} lacks unseen codes {missing}")
        vals = [auroc(z["Y"][z["A"][:, j] > 0, j], z["P"][z["A"][:, j] > 0, j]) for j in cols
                if eligible(z["Y"], z["A"], j)]
        if not vals:
            warn(f"tab_unseen: {ext.file} has no evaluable unseen label; skipped")
            continue
        rows["merl"][t].append((float(np.mean(vals)), len(vals), f"{ext.file} (unseen set of {ref.file})"))
    if not rows:
        warn("tab_unseen: nothing to report; table not written")
        return
    order = {m: i for i, m in enumerate(METHOD_ORDER)}
    names = sorted(rows, key=lambda s: (s.startswith(OURS), s != "merl", order.get(s, 99), s))
    cells = {}
    for nm in names:
        for t in targets:
            ent = rows[nm].get(t, [])
            mu, sd, n = mean_std([e[0] for e in ent])
            nl = sorted({e[1] for e in ent})
            cells[(nm, t)] = (mu, sd, nl)
            for e in ent:
                trace.add(sec, f"{nm} / {t} (n_labels={e[1]})", e[0], [e[2]])
    best = {t: max([cells[(nm, t)][0] for nm in names if cells[(nm, t)][0] is not None], default=None) for t in targets}
    lines = [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             r"\caption{Macro AUROC ($\times 100$) on labels never supervised in the training sources (unseen), "
             r"mean$_{\pm\mathrm{std}}$ over seeds; $n$ = number of evaluable unseen labels.}",
             r"\label{tab:unseen}", r"\begin{tabular}{l" + "cc" * len(targets) + "}", r"\toprule",
             " & ".join([""] + [rf"\multicolumn{{2}}{{c}}{{{TARGET_TEX.get(t, t)}}}" for t in targets]) + r" \\",
             " ".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(targets))),
             " & ".join(["Method"] + ["AUROC", "$n$"] * len(targets)) + r" \\", r"\midrule"]
    for nm in names:
        base, _, suffix = nm.partition(" (")
        row = [method_tex(base) + (f" ({tex_escape(suffix)}" if suffix else "")]
        for t in targets:
            mu, sd, nl = cells[(nm, t)]
            row += [fmt_cell(mu, sd, mu is not None and np.isclose(mu, best[t] or -1)),
                    "/".join(map(str, nl)) if nl else "--"]
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write_tex(out / "tab_unseen.tex", lines)
    print(f"wrote {out / 'tab_unseen.tex'}")


# ----------------------------------------------------------------------------------------------------------------------
# Significance
# ----------------------------------------------------------------------------------------------------------------------
def aligned(run, ref, codes):
    """Columns of `codes` in run's npz; asserts records and labels are identical to the reference run."""
    z, zr = run.npz(), ref.npz()
    assert z["idx"] is not None and zr["idx"] is not None, f"{run.file}/{ref.file}: npz without idx"
    assert np.array_equal(z["idx"], zr["idx"]), \
        f"test records differ between {run.npz_path.name} and {ref.npz_path.name} (idx mismatch)"
    cols = [run.vocab.index(c) for c in codes]
    rcols = [ref.vocab.index(c) for c in codes]
    assert np.array_equal(z["Y"][:, cols], zr["Y"][:, rcols]) and np.array_equal(z["A"][:, cols], zr["A"][:, rcols]), \
        f"labels differ between {run.npz_path.name} and {ref.npz_path.name}"
    return z["P"][:, cols]


def significance(runs, out, trace, targets, B, seed, label_set):
    sec_b = f"stats: paired bootstrap ({B} resamples, seed {seed}) of macro AUROC difference, seed-0 runs"
    sec_d = "stats: per-label DeLong (Sun & Xu 2014), ours vs best baseline, Holm-corrected over all tests"
    main0 = {(r.target, r.method): r for r in runs if r.kind == "main" and r.seed == 0 and r.npz() is not None}
    rows, delong_rows, heat = [], [], {}
    for t in targets:
        ref = main0.get((t, OURS))
        if ref is None:
            warn(f"stats: no seed-0 'ours' run with npz for {t}; significance skipped for this target")
            continue
        others = [main0[(t, m)] for m in ordered_methods({m for (tt, m) in main0 if tt == t}) if m != OURS]
        if not others:
            warn(f"stats: no seed-0 baseline with npz for {t}")
            continue
        zr = ref.npz()
        pool = ref.d.get("seen", ref.vocab) if label_set == "seen" else ref.vocab
        codes = [c for c in pool if c in ref.vocab and all(c in o.vocab for o in others)]
        dropped = [c for c in pool if c in ref.vocab and c not in codes]
        if dropped:
            warn(f"stats {t}: codes {dropped} missing from some baseline vocab; excluded")
        codes = [c for c in codes if eligible(zr["Y"], zr["A"], ref.vocab.index(c))]
        if not codes:
            warn(f"stats {t}: no label with >= {MIN_POS} positives")
            continue
        Y = zr["Y"][:, [ref.vocab.index(c) for c in codes]]
        A = zr["A"][:, [ref.vocab.index(c) for c in codes]]
        n = len(Y)
        P = {r.method: aligned(r, ref, codes) for r in [ref] + others}
        masks = [A[:, k] > 0 for k in range(len(codes))]
        point = {m: np.array([auroc(Y[mk, k], p[mk, k]) for k, mk in enumerate(masks)]) for m, p in P.items()}
        # bootstrap: identical weights for all methods of this target
        boot = {m: [] for m in P}
        for W in bootstrap_weights(n, B, seed):
            for m, p in P.items():
                boot[m].append(np.stack([weighted_auc_boot(p[mk, k], Y[mk, k], W[:, mk])
                                         for k, mk in enumerate(masks)], 1))
        boot = {m: np.concatenate(v) for m, v in boot.items()}  # [B, L]
        ok_ref = point[OURS].mean()
        if ref.metric("test_seen", "auroc") is not None and label_set == "seen" and \
                not np.isclose(ok_ref, ref.metric("test_seen", "auroc"), atol=2e-3):  # npz P is float16
            warn(f"stats {t}: recomputed ours macro AUROC {ok_ref:.4f} != json test_seen {ref.metric('test_seen', 'auroc'):.4f}")
        for o in others:
            d0 = float(point[OURS].mean() - point[o.method].mean())
            db = np.nanmean(boot[OURS], 1) - np.nanmean(boot[o.method], 1)
            db = db[np.isfinite(db)]
            lo, hi = np.percentile(db, [2.5, 97.5])
            p = min(1.0, 2 * min((1 + (db <= 0).sum()) / (len(db) + 1), (1 + (db >= 0).sum()) / (len(db) + 1)))
            rows.append(dict(test="bootstrap_macro_auroc", target=t, ours=ref.file, baseline=o.method,
                             baseline_file=o.file, label="macro", n_labels=len(codes), n_records=n,
                             auroc_ours=float(point[OURS].mean()), auroc_baseline=float(point[o.method].mean()),
                             delta=d0, ci_lo=float(lo), ci_hi=float(hi), p=float(p), n_boot=len(db)))
            trace.add(sec_b, f"{t}: ours - {o.method}, delta [95% CI], p (L={len(codes)}, n={n})",
                      f"{d0:.4f} [{lo:.4f}, {hi:.4f}], p={p:.4g}", [ref.file, o.file])
        # best baseline for this target = highest macro AUROC (same labels, seed-0 npz) among baselines
        best = max(others, key=lambda o: point[o.method].mean())
        for k, c in enumerate(codes):
            mk = masks[k]
            a1, a2, z, p = fast_delong(P[OURS][mk, k], P[best.method][mk, k], Y[mk, k])
            delong_rows.append(dict(test="delong_per_label", target=t, ours=ref.file, baseline=best.method,
                                    baseline_file=best.file, label=abbr(c), code=c, n_records=int(mk.sum()),
                                    n_pos=int(Y[mk, k].sum()), auroc_ours=a1, auroc_baseline=a2, delta=a1 - a2,
                                    z=z, p=p))
            heat[(abbr(c), t)] = a1 - a2
    if rows:
        adj = holm([r["p"] for r in rows])
        for r, a in zip(rows, adj):
            r["p_holm"] = float(a)
    if delong_rows:
        adj = holm([r["p"] for r in delong_rows])
        for r, a in zip(delong_rows, adj):
            r["p_holm"] = float(a)
            trace.add(sec_d, f"{r['target']} / {r['label']}: ours - {r['baseline']} (AUROC {r['auroc_ours']:.4f} vs "
                             f"{r['auroc_baseline']:.4f}), p, p_holm",
                      f"{r['delta']:.4f}, p={r['p']:.3g}, p_holm={a:.3g}", [r["ours"], r["baseline_file"]])
    all_rows = rows + delong_rows
    if not all_rows:
        warn("stats: nothing computed; stats.csv / tab_stats.tex not written")
        return {}, []
    keys = ["test", "target", "baseline", "label", "code", "n_labels", "n_records", "n_pos", "auroc_ours",
            "auroc_baseline", "delta", "ci_lo", "ci_hi", "z", "p", "p_holm", "n_boot", "ours", "baseline_file"]
    with open(out / "stats.csv", "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in all_rows:
            f.write(",".join("" if r.get(k) is None else (f"{r[k]:.6g}" if isinstance(r[k], float) else str(r[k]))
                             for k in keys) + "\n")
    print(f"wrote {out / 'stats.csv'}")
    write_stats_tex(rows, delong_rows, out, targets, B)
    return heat, delong_rows


def fmt_p(p):
    return "$<$0.001" if p < 1e-3 else f"{p:.3f}"


def write_stats_tex(rows, delong_rows, out, targets, B):
    tts = [t for t in targets if any(r["target"] == t for r in rows)]
    base = ordered_methods({r["baseline"] for r in rows})
    cell = {(r["baseline"], r["target"]): r for r in rows}
    L = [r"\begin{table*}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{2pt}",
         rf"\caption{{Paired bootstrap ({B} resamples of test records) of the difference in macro AUROC "
         r"($\times 100$) between our method and each baseline (seed 0), with 95\% confidence interval and two-sided "
         r"$p$ (Holm-adjusted over all comparisons in the table). Bottom: number of labels for which our method is "
         r"significantly better / worse than the strongest baseline per target (DeLong test, Holm-adjusted "
         r"$p<0.05$) out of all evaluated labels.}", r"\label{tab:stats}",
         r"\begin{tabular}{l" + "cc" * len(tts) + "}", r"\toprule",
         " & ".join([""] + [rf"\multicolumn{{2}}{{c}}{{{TARGET_TEX.get(t, t)}}}" for t in tts]) + r" \\",
         " ".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(tts))),
         " & ".join(["Ours vs."] + [r"$\Delta$ [95\% CI]", r"$p_\mathrm{Holm}$"] * len(tts)) + r" \\", r"\midrule"]
    for b in base:
        row = [method_tex(b)]
        for t in tts:
            r = cell.get((b, t))
            if r is None:
                row += ["--", "--"]
            else:
                row += [rf"${100 * r['delta']:+.1f}$ [{100 * r['ci_lo']:.1f}, {100 * r['ci_hi']:.1f}]",
                        fmt_p(r["p_holm"])]
        L.append(" & ".join(row) + r" \\")
    if delong_rows:
        L += [r"\midrule", " & ".join(["DeLong vs.\\ best"] + [rf"\multicolumn{{2}}{{c}}{{{s}}}" for s in [
            (lambda rs: (f"{method_tex(rs[0]['baseline'])}: "
                         f"{sum(1 for r in rs if r['p_holm'] < .05 and r['delta'] > 0)} / "
                         f"{sum(1 for r in rs if r['p_holm'] < .05 and r['delta'] < 0)} of {len(rs)}") if rs else "--")
            ([r for r in delong_rows if r["target"] == t]) for t in tts]]) + r" \\"]
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    write_tex(out / "tab_stats.tex", L)
    print(f"wrote {out / 'tab_stats.tex'}")


# ----------------------------------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------------------------------
def setup_mpl():
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6, "lines.linewidth": 1.0,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "axes.prop_cycle": plt.cycler(color=OKABE)})


def fig_heatmap(heat, delong_rows, out, trace, targets):
    if not heat:
        warn("fig_perlabel_heatmap: no per-label comparison available; not drawn")
        return
    tts = [t for t in targets if any(k[1] == t for k in heat)]
    labs = sorted({k[0] for k in heat})
    M = np.full((len(labs), len(tts)), np.nan)
    for (l, t), v in heat.items():
        M[labs.index(l), tts.index(t)] = v
    sig = {(r["label"], r["target"]) for r in delong_rows if r["p_holm"] < 0.05}
    vmax = max(0.01, np.nanmax(np.abs(M)))
    h = max(4.0, 0.33 * len(labs) + 1.8) * CM
    fig, ax = plt.subplots(figsize=(8.5 * CM, h))
    cmap = plt.get_cmap("RdBu").copy(); cmap.set_bad("#DDDDDD")
    im = ax.imshow(M, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    for i, l in enumerate(labs):
        for j, t in enumerate(tts):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{100 * M[i, j]:+.1f}" + ("*" if (l, t) in sig else ""), ha="center", va="center",
                        fontsize=6, color="white" if abs(M[i, j]) > 0.6 * vmax else "black")
    ax.set_xticks(range(len(tts)), [TARGET_TEX.get(t, t) for t in tts])
    ax.set_yticks(range(len(labs)), labs)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    best = {r["target"]: METHOD_TEX.get(r["baseline"], r["baseline"]) for r in delong_rows}
    ax.set_xlabel("Held-out target database")
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label(r"$\Delta$AUROC (ours $-$ best baseline)")
    cb.ax.tick_params(labelsize=6)
    fig.savefig(out / "fig_perlabel_heatmap.pdf")
    plt.close(fig)
    trace.add("fig_perlabel_heatmap.pdf", "cells", "same values as DeLong rows above (delta); * = p_holm < 0.05",
              [])
    trace.add("fig_perlabel_heatmap.pdf", "best baseline per target",
              "; ".join(f"{t}: {b}" for t, b in best.items()), [])
    print(f"wrote {out / 'fig_perlabel_heatmap.pdf'}")


def fig_e4(runs, out, trace, targets):
    e4 = [r for r in runs if r.kind == "e4" and r.metric("test_seen", "auroc") is not None]
    if not e4:
        warn("fig_e4: no run with config.drop_rate > 0 (or e4- prefix); not drawn")
        return
    sec = "fig_e4.pdf (test_seen macro AUROC, mean over targets, error bar = std over seeds)"
    main = [r for r in runs if r.kind == "main" and r.metric("test_seen", "auroc") is not None]
    groups = defaultdict(list)
    for r in e4:
        groups[r.signature()].append(r)
    fig, ax = plt.subplots(figsize=(8.5 * CM, 6 * CM))
    markers = "osD^vP*X"
    for gi, (sig, rs) in enumerate(sorted(groups.items(), key=lambda kv: str(kv[0]))):
        c = dict(sig)
        strat = json.loads(c.get("labels", '"?"')) + (" + hierarchy" if json.loads(c.get("hierarchy", "false")) else "")
        head = json.loads(c.get("head", '"?"'))
        name = f"{strat} ({head} head)"
        pts = list(rs)
        if not any(r.drop_rate == 0 for r in rs):
            zero = [r for r in main if r.signature() == sig]
            if zero:
                pts += zero
            else:
                warn(f"fig_e4: no drop_rate=0 run with the same config for '{name}'; curve starts at first drop rate")
        xs, ys, es = [], [], []
        for dr in sorted({r.drop_rate for r in pts}):
            at = [r for r in pts if r.drop_rate == dr]
            seeds = sorted({r.seed for r in at})
            per_seed, files = [], []
            for s in seeds:
                rr = {r.target: r for r in at if r.seed == s}
                if all(t in rr for t in targets):
                    per_seed.append(np.mean([rr[t].metric("test_seen", "auroc") for t in targets]))
                    files += [rr[t].file for t in targets]
                else:
                    warn(f"fig_e4: '{name}' drop={dr} seed={s} lacks targets "
                         f"{[t for t in targets if t not in rr]}; seed excluded from this point")
            mu, sd, n = mean_std(per_seed)
            if mu is None:
                continue
            xs.append(dr); ys.append(mu); es.append(np.nan if sd is None else sd)
            trace.add(sec, f"{name} / drop {dr} ({n} seed(s))", mu, files)
        if xs:
            ax.errorbar(xs, ys, yerr=es, marker=markers[gi % len(markers)], ms=3.5, capsize=2, label=name,
                        color=OKABE[gi % len(OKABE)])
    ax.set_xlabel("Fraction of (source, label) annotations removed")
    ax.set_ylabel("Macro AUROC (seen labels)")
    ax.legend(frameon=False)
    ax.grid(axis="y", lw=0.3, alpha=0.5)
    fig.savefig(out / "fig_e4.pdf")
    plt.close(fig)
    print(f"wrote {out / 'fig_e4.pdf'}")


def fig_ablation(runs, out, trace, targets):
    abl = [r for r in runs if r.kind == "ablation" and r.metric("test_seen", "auroc") is not None]
    if not abl:
        warn("fig_ablation: no ablation runs (abl-* or loso-*-ours-<variant>-*); not drawn")
        return
    sec = "fig_ablation.pdf (test_seen macro AUROC; bar = mean over seeds, error = std)"
    ref = [r for r in runs if r.kind == "main" and r.method == OURS and r.metric("test_seen", "auroc") is not None]
    variants = [("ours", "Full model", ref)] + [
        (v, v, [r for r in abl if r.method == v]) for v in sorted({r.method for r in abl})]
    variants = [v for v in variants if v[2]]
    cols = [t for t in targets if any(r.target == t for v in variants for r in v[2])]
    fig, ax = plt.subplots(figsize=(17 * CM, 5.5 * CM))
    w = 0.8 / len(variants)
    for vi, (key, lab, rs) in enumerate(variants):
        mus, sds = [], []
        for t in cols:
            at = [r for r in rs if r.target == t]
            mu, sd, n = mean_std([r.metric("test_seen", "auroc") for r in at])
            mus.append(np.nan if mu is None else mu); sds.append(np.nan if sd is None else sd)
            if mu is not None:
                trace.add(sec, f"{lab} / {t} ({n} seed(s))", mu, [r.file for r in at])
        x = np.arange(len(cols)) + (vi - (len(variants) - 1) / 2) * w
        ax.bar(x, mus, w * 0.95, yerr=sds, capsize=1.5, label=lab, color=OKABE[vi % len(OKABE)],
               error_kw=dict(lw=0.6))
    ax.set_xticks(range(len(cols)), [TARGET_TEX.get(t, t) for t in cols])
    ax.set_ylabel("Macro AUROC (seen labels)")
    allv = [r.metric("test_seen", "auroc") for v in variants for r in v[2]]
    ax.set_ylim(max(0.0, min(allv) - 0.05), min(1.0, max(allv) + 0.03))
    ax.legend(frameon=False, ncol=min(4, len(variants)), loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.grid(axis="y", lw=0.3, alpha=0.5)
    fig.savefig(out / "fig_ablation.pdf")
    plt.close(fig)
    print(f"wrote {out / 'fig_ablation.pdf'}")


# ----------------------------------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=r"E:\Claude\ecg-paper\results\kaggle")
    ap.add_argument("--out", default=r"E:\Claude\ecg-paper\paper\generated")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--boot_seed", type=int, default=0)
    ap.add_argument("--stat_labels", choices=["seen", "all"], default="seen",
                    help="label pool for significance tests (default: seen labels, as in tab_loso)")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    setup_mpl()
    runs, externals = load_runs(a.results)
    print(f"loaded {len(runs)} runs ({', '.join(f'{k}={sum(r.kind == k for r in runs)}' for k in ['main', 'ablation', 'e4', 'openvocab', 'other'])}), "
          f"external zero-shot: {sorted(externals)}")
    for r in runs:
        if r.kind == "other":
            warn(f"{r.file}: not used by any table/figure (prefix={r.prefix}, mode={r.cfg.get('mode')})")
    present = [t for t in TARGETS if any(r.target == t and r.kind == "main" for r in runs)]
    for t in TARGETS:
        if t not in present:
            warn(f"no main LOSO run for target {t}; column omitted")
    targets = present or TARGETS
    trace = Trace()
    main_table(runs, ["auroc", "auprc"], {"auroc": "AUROC", "auprc": "AUPRC"}, "tab_loso.tex",
               r"Leave-one-database-out results: macro AUROC and AUPRC ($\times 100$) over labels seen in training, "
               r"mean$_{\pm\mathrm{std}}$ over seeds where more than one seed is available. Best per column in bold.",
               "tab:loso", out, trace, targets)
    main_table(runs, ["f1", "f1_oracle"], {"f1": "F1", "f1_oracle": "F1$^{*}$"}, "tab_loso_f1.tex",
               r"Leave-one-database-out macro F1 ($\times 100$) over seen labels, with thresholds tuned on source "
               r"validation data (F1) and the oracle threshold on the target (F1$^{*}$, upper bound).",
               "tab:loso_f1", out, trace, targets)
    unseen_table(runs, externals, out, trace, targets)
    heat, dl = significance(runs, out, trace, targets, a.n_boot, a.boot_seed, a.stat_labels)
    fig_heatmap(heat, dl, out, trace, targets)
    fig_e4(runs, out, trace, targets)
    fig_ablation(runs, out, trace, targets)
    header = (f"# Generated results summary\n\nResults folder: `{Path(a.results).resolve()}`  \n"
              f"Output folder: `{out.resolve()}`  \nRuns loaded: {len(runs)}; external zero-shot files: "
              f"{', '.join(e.file for e in externals.values()) or 'none'}  \n"
              f"All values are fractions (tables print them x100). Macro metrics use labels annotated in the "
              f"target (A==1) with >= {MIN_POS} positives, as in train.py `evaluate`.\n\n## Run index\n\n"
              "| file | kind | target | method | seed | drop_rate | head | labels |\n|---|---|---|---|---|---|---|---|\n"
              + "".join(f"| {r.file} | {r.kind} | {r.target} | {r.method} | {r.seed} | {r.drop_rate} | "
                        f"{r.cfg.get('head')} | {r.cfg.get('labels')} |\n" for r in runs))
    header += "\n## Warnings (skipped / missing inputs)\n\n" + ("".join(f"- {w}\n" for w in _WARNED) or "none\n")
    trace.write(out / "summary.md", header)
    print(f"wrote {out / 'summary.md'}")


if __name__ == "__main__":
    main()
