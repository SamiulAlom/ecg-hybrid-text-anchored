"""Write a FAKE results folder (random numbers!) in the exact train.py output format, to test aggregate.py.

  .venv/Scripts/python analysis/make_fake_results.py --out analysis/_fake/results

Nothing produced here is a real result. Every json carries "FAKE": true and name/config like train.py.
Deliberate gaps to exercise the warnings: no dann run for SHAONING, one json without npz, one unparsable tag.
"""
import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from labels import LABELS  # noqa: E402

TARGETS = ["CPSC", "PTBXL", "GEORGIA", "SHAONING"]
VOCAB = list(LABELS)[:16]
METHODS = {  # head, labels, hierarchy, dg, mixstyle, signal strength
    "naive": ("linear", "naive", False, "none", False, 0.8),
    "mask": ("linear", "mask", False, "none", False, 1.0),
    "li2024": ("linear", "mask", True, "none", False, 1.05),
    "coral": ("linear", "mask", True, "coral", False, 1.05),
    "dann": ("linear", "mask", True, "dann", False, 1.0),
    "groupdro": ("linear", "mask", True, "groupdro", False, 1.02),
    "mixstyle": ("linear", "mask", True, "none", True, 1.1),
    "ours": ("text", "mask_pseudo", True, "none", True, 1.25),
}


def evaluate(P, Y, A, cols, min_pos=5):
    """Same rule as train.py evaluate (threshold 0.5 stands in for the validation threshold)."""
    per, grid = {}, np.linspace(0.02, 0.98, 49)
    for j in cols:
        m = A[:, j] > 0
        y, p = Y[m, j], P[m, j]
        if m.sum() == 0 or y.sum() < min_pos or y.sum() == len(y):
            continue
        per[j] = dict(auroc=roc_auc_score(y, p), auprc=average_precision_score(y, p), f1=f1_score(y, p >= 0.5),
                      f1_oracle=max(f1_score(y, p >= t) for t in grid), n_pos=int(y.sum()), n=int(m.sum()),
                      prev=float(y.mean()))
    if not per:
        return {}, per
    macro = {k: float(np.mean([v[k] for v in per.values()])) for k in ("auroc", "auprc", "f1", "f1_oracle")}
    macro["n_labels"] = len(per)
    return macro, per


def target_data(t):
    rng = np.random.default_rng(zlib.crc32(t.encode()))
    n = int(rng.integers(300, 500))
    idx = np.sort(rng.choice(20000, n, replace=False))
    C = len(VOCAB)
    prev = rng.uniform(0.04, 0.3, C)
    Y = (rng.random((n, C)) < prev).astype(np.uint8)
    ann = rng.random(C) < 0.85  # labels the target database annotates
    A = np.repeat(ann[None].astype(np.uint8), n, 0)
    no_sup = list(rng.choice(C, 2, replace=False))  # labels no training source annotates
    return idx, Y, A, no_sup


def write_run(out, tag, t, method, seed, drop_rate=0.0, holdout=(), strength_delta=0.0, labels=None, hier=None):
    head, lab, h, dg, mix, s = METHODS[method]
    lab = labels or lab
    h = h if hier is None else hier
    idx, Y, A, no_sup = target_data(t)
    rng = np.random.default_rng(zlib.crc32(tag.encode()))
    C = len(VOCAB)
    hold = [VOCAB.index(c) for c in holdout]
    unseen = sorted(set(no_sup) | set(hold))
    seen = [j for j in range(C) if j not in unseen]
    q = rng.uniform(0.5, 1.5, C) * (s + strength_delta - 0.8 * drop_rate)
    q[unseen] *= (0.6 if head in ("text", "query") else 0.0)  # only text heads can transfer to unseen labels
    P = 1 / (1 + np.exp(-(q * (2 * Y.astype(np.float64) - 1) + rng.normal(0, 1.2, Y.shape) - 1.0)))
    m_all, per = evaluate(P, Y, A, list(range(C)))
    m_seen, _ = evaluate(P, Y, A, seen)
    m_unseen, _ = evaluate(P, Y, A, unseen)
    cfg = dict(prep="FAKE", text_emb="FAKE.npz" if head == "text" else None, out=str(out), tag=tag, mode="loso",
               target=t, head=head, labels=lab, hierarchy=h, dg=dg, dg_w=None, mixstyle=mix, lead_drop=0.1,
               holdout=list(holdout), min_source=10, drop_rate=drop_rate, loss="asl", epochs=30, bs=256, lr=1e-3,
               wd=1e-2, pseudo_w=0.5, pseudo_hi=0.9, pseudo_lo=0.05, ema=0.999, seed=seed, split_seed=0, max_train=0)
    res = dict(FAKE=True, name=tag, config=cfg, vocab=VOCAB, best_val_auroc=float(rng.uniform(.8, .9)), history=[],
               test_all=m_all, test_seen=m_seen, test_unseen=m_unseen, seen=[VOCAB[j] for j in seen],
               unseen=[VOCAB[j] for j in unseen], dropped=[],
               per_label={LABELS[VOCAB[j]][0]: v for j, v in per.items()}, minutes=float(rng.uniform(20, 60)))
    with open(out / f"{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    np.savez_compressed(out / f"{tag}_test.npz", P=P.astype(np.float16), Y=Y, A=A, idx=idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "_fake" / "results"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    for t in TARGETS:
        for m in METHODS:
            if m == "dann" and t == "SHAONING":
                continue  # deliberately missing run
            for s in ([0, 1, 2] if m in ("ours", "mixstyle") else [0]):
                write_run(out, f"loso-{t}-{m}-s{s}", t, m, s)
        for v, dlt, lab, h in [("nopseudo", -0.1, "mask", True), ("nohier", -0.05, None, False)]:
            write_run(out, f"abl-{t}-{v}-s0", t, "ours", 0, strength_delta=dlt, labels=lab, hier=h)
        write_run(out, f"ov-{t}-ours-s0", t, "ours", 0, holdout=VOCAB[:2])
        for dr in (0.25, 0.5, 0.75):
            write_run(out, f"e4-{t}-mask-d{dr}-s0", t, "mask", 0, drop_rate=dr)
            write_run(out, f"e4-{t}-naive-d{dr}-s0", t, "naive", 0, drop_rate=dr)
            write_run(out, f"e4-{t}-ours-d{dr}-s0", t, "ours", 0, drop_rate=dr)
        if t != "SHAONING":  # external zero-shot baseline for 3 targets
            idx, Y, A, _ = target_data(t)
            rng = np.random.default_rng(zlib.crc32(("merl" + t).encode()))
            P = 1 / (1 + np.exp(-(0.5 * (2 * Y.astype(np.float64) - 1) + rng.normal(0, 1.2, Y.shape) - 1)))
            np.savez_compressed(out / f"merl-{t}_test.npz", P=P.astype(np.float16), Y=Y, A=A, idx=idx)
            json.dump(dict(FAKE=True, name=f"merl-{t}", vocab=VOCAB), open(out / f"merl-{t}.json", "w"))
    # graceful-handling probes
    write_run(out, "loso-CPSC-coral-s1", "CPSC", "coral", 1)
    (out / "loso-CPSC-coral-s1_test.npz").unlink()  # json without npz
    write_run(out, "weird_tag_name", "CPSC", "mask", 0)
    print(f"FAKE results written to {out}")


if __name__ == "__main__":
    main()
