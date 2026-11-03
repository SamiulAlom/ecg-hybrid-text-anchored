"""Systematic zero-shot (held-out diagnosis) experiment and annotation-threshold sensitivity: command lists.

Zero-shot (E3): in every leave-one-database-out fold, the diagnoses that the held-out database annotates and that at
least one training source annotates (i.e. would normally be "seen") are split into fixed random groups of about
GROUP_SIZE. Each run withholds one group from training entirely (--holdout: the columns never enter the loss) and
scores it on the held-out database with the text head. Every eligible diagnosis is therefore scored zero-shot exactly
once per fold and seed. Sinus rhythm is never withheld. Groups are drawn once (seed 0) and shared by all seeds.

Threshold sensitivity: proposed method and Li-style baseline (both with weight averaging) with the annotation
threshold --min_source 5 and 20 instead of 10, all folds, seed 0.

Writes kaggle/build/code/colab_cmds_e3.txt and kaggle/zeroshot_groups.json.
Usage: python kaggle/make_zeroshot.py <prep_dir>
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from data import MAIN_SOURCES, build_vocab  # noqa: E402
from labels import LABELS  # noqa: E402

TARGETS = ["CPSC", "PTBXL", "GEORGIA", "SHAONING"]
GROUP_SIZE = 6
PROTECT = {"426783006"}  # sinus rhythm
BASE = "--prep {PREP} --mode loso --epochs 30"
OURS = ("--head hybrid --text_emb {EMB} --labels mask --hierarchy --eval_ema --ema_warmup --text_w 1.0 "
        "--seen_score avg")
LISTYLE = "--head linear --labels mask --hierarchy --eval_ema --ema_warmup"


def main(prep):
    meta = pd.read_csv(os.path.join(prep, "meta.csv"), dtype={"dx": str})
    meta["dx"] = meta["dx"].fillna("")
    vocab, annotated, _ = build_vocab(meta, min_source=10)
    ann = {s: set(annotated[s]) for s in MAIN_SOURCES}
    groups, cmds = {}, []
    rng = np.random.default_rng(0)
    for t in TARGETS:
        train_ann = set().union(*[ann[s] for s in MAIN_SOURCES if s != t])
        elig = sorted(c for c in vocab if c in ann[t] and c in train_ann and c not in PROTECT)
        perm = [elig[i] for i in rng.permutation(len(elig))]
        k = max(1, round(len(perm) / GROUP_SIZE))
        gs = [sorted(perm[i::k]) for i in range(k)]
        groups[t] = gs
        print(t, len(elig), "eligible ->", k, "groups:", [[LABELS[c][0] for c in g] for g in gs])
        for gi, g in enumerate(gs):
            for s in range(3):
                cmds.append(f"{BASE} --target {t} {OURS} --holdout {' '.join(g)} --seed {s} "
                            f"--tag e3-{t}-ours-g{gi}-s{s}")
    for t in TARGETS:
        for ms in (5, 20):
            cmds.append(f"{BASE} --target {t} {OURS} --min_source {ms} --seed 0 --tag thr{ms}-{t}-ours-s0")
            cmds.append(f"{BASE} --target {t} {LISTYLE} --min_source {ms} --seed 0 --tag thr{ms}-{t}-li2024-s0")
    json.dump(groups, open(os.path.join(ROOT, "kaggle", "zeroshot_groups.json"), "w"), indent=1)
    out = os.path.join(ROOT, "kaggle", "build", "code", "colab_cmds_e3.txt")
    open(out, "w").write("\n".join(cmds) + "\n")
    print(len(cmds), "commands ->", out)


if __name__ == "__main__":
    main(sys.argv[1])
