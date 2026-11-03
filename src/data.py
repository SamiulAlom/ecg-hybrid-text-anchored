"""Label matrices, per-source annotation masks and experiment splits."""
from collections import Counter

import numpy as np
import pandas as pd

from labels import LABELS, canonical

MAIN_SOURCES = ["CPSC", "PTBXL", "GEORGIA", "SHAONING"]

# child -> parent implications used for label completion ("if child then parent").
IMPLIES = {
    "164909002": ["6374002"],   # LBBB -> BBB
    "59118001": ["6374002"],    # RBBB -> BBB
    "426177001": ["426627000"],  # sinus bradycardia -> bradycardia
    "270492004": ["164947007"],  # 1st degree AV block -> prolonged PR
    "445118002": ["39732003"],   # LAnFB -> left axis deviation
    "59931005": ["164934002"],   # T inversion -> T wave abnormal
    "195042002": ["233917008"],  # 2nd degree AVB -> AV block
}


def parse_codes(dx):
    return {canonical(c) for c in str(dx).split(";") if c and c != "nan"}


def build_vocab(meta, min_total=50, min_source=10, sources=MAIN_SOURCES):
    """Codes we have a description for and that are annotated in at least one main source."""
    cnt = {s: Counter() for s in sources}
    for s, dx in zip(meta.source, meta.dx):
        if s in cnt:
            cnt[s].update(parse_codes(dx))
    vocab = []
    for c in LABELS:
        tot = sum(cnt[s][c] for s in sources)
        if tot >= min_total and any(cnt[s][c] >= min_source for s in sources):
            vocab.append(c)
    annotated = {s: {c for c in vocab if cnt[s][c] >= min_source} for s in sources}
    return vocab, annotated, cnt


def label_matrices(meta, vocab, annotated, complete_hierarchy=False):
    """Y: positives; A: 1 where the record's source annotates that label (else unknown)."""
    idx = {c: i for i, c in enumerate(vocab)}
    N, C = len(meta), len(vocab)
    Y = np.zeros((N, C), np.float32)
    A = np.zeros((N, C), np.float32)
    for n, (s, dx) in enumerate(zip(meta.source, meta.dx)):
        codes = parse_codes(dx)
        if complete_hierarchy:
            for c in list(codes):
                codes.update(IMPLIES.get(c, []))
        for c in codes:
            if c in idx:
                Y[n, idx[c]] = 1
        for c in annotated.get(s, ()):
            A[n, idx[c]] = 1
        if complete_hierarchy:  # a positive implied parent is known even if the source never codes it
            A[n] = np.maximum(A[n], Y[n])
    return Y, A


def drop_annotations(Y, A, meta, rate, seed, sources=MAIN_SOURCES, protect=("426783006",), vocab=None):
    """E4: remove a random fraction of (source, diagnosis) annotations, as if that source never coded it.
    Returns training copies; evaluation keeps the original labels. Sinus rhythm is protected."""
    rng = np.random.default_rng(1000 + seed)
    Y, A = Y.copy(), A.copy()
    keep = {vocab.index(c) for c in protect if vocab and c in vocab}
    dropped = []
    src = meta.source.values
    for s in sources:
        rows = src == s
        cols = [j for j in np.where(A[rows].max(0) > 0)[0] if j not in keep]
        for j in cols:
            if rng.random() < rate:
                A[rows, j] = 0
                Y[rows, j] = 0
                dropped.append((s, int(j)))
    return Y, A, dropped


def split_indices(meta, mode, target=None, seed=0, val_frac=0.1, test_frac=0.1):
    """mode='loso': test = all records of `target`; train/val = random split of other main sources.
    mode='indist': per-source random train/val/test split over all main sources (or just `target`)."""
    rng = np.random.default_rng(seed)
    src = meta.source.values
    main = np.isin(src, MAIN_SOURCES)
    if mode == "loso":
        test = np.where(src == target)[0]
        pool = np.where(main & (src != target))[0]
        rng.shuffle(pool)
        nv = int(len(pool) * val_frac)
        return pool[nv:], pool[:nv], test
    tr, va, te = [], [], []
    for s in ([target] if target else MAIN_SOURCES):
        ids = np.where(src == s)[0]
        rng.shuffle(ids)
        nt, nv = int(len(ids) * test_frac), int(len(ids) * val_frac)
        te.append(ids[:nt]); va.append(ids[nt:nt + nv]); tr.append(ids[nt + nv:])
    return np.concatenate(tr), np.concatenate(va), np.concatenate(te)


def load_prep(prep_dir):
    X = np.load(f"{prep_dir}/signals.npy", mmap_mode="r")
    meta = pd.read_csv(f"{prep_dir}/meta.csv", dtype={"dx": str})
    meta["dx"] = meta["dx"].fillna("")
    if "n100" not in meta:  # older 10 s preprocessing
        meta["n100"] = X.shape[2]
    return X, meta
