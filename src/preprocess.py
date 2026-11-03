"""Preprocess PhysioNet-2020 + Shaoxing/Ningbo WFDB records into compact 100 Hz arrays.

Runs on Kaggle (inputs auto-discovered under /kaggle/input) or locally (--roots ...).
Outputs (in --out):
  signals.npy   float16 [N, 12, L]   band-passed, resampled to 100 Hz, cropped/padded to L samples (20 s)
  meta.csv      one row per record: record, source, fs_orig, n_orig, n100 (valid samples), age, sex,
                dx (SNOMED codes ';')
  code_counts.csv  positives per SNOMED code per source (drives the "annotated vocabulary")
"""
import argparse
import os
import re
from collections import Counter, defaultdict
from functools import partial
from math import gcd
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, resample_poly, sosfiltfilt

TARGET_FS = 100
LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# record-name prefix -> source domain
PREFIX_SOURCE = {
    "A": "CPSC",     # CPSC2018
    "Q": "CPSC",     # CPSC-Extra
    "I": "INCART",
    "S": "PTB",
    "HR": "PTBXL",
    "E": "GEORGIA",
    "JS": "SHAONING",  # Shaoxing + Ningbo (PhysioNet 2021 release)
}


def source_of(record):
    for p in sorted(PREFIX_SOURCE, key=len, reverse=True):
        if record.startswith(p) and record[len(p):len(p) + 1].isdigit():
            return PREFIX_SOURCE[p]
    return "UNKNOWN"


def parse_header(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [l.strip() for l in f if l.strip()]
    rec, n_sig, fs, n_samp = lines[0].split()[:4]
    n_sig, fs, n_samp = int(n_sig), float(fs.split("/")[0]), int(n_samp)
    gains, baselines, names = [], [], []
    for l in lines[1:1 + n_sig]:
        parts = l.split()
        g = parts[2]
        m = re.match(r"([0-9.eE+-]+)(?:\(([-0-9]+)\))?", g)
        gains.append(float(m.group(1)) if m and float(m.group(1)) != 0 else 1000.0)
        baselines.append(int(m.group(2)) if m and m.group(2) else 0)
        names.append(parts[-1].upper())
    info = {"age": np.nan, "sex": "", "dx": ""}
    for l in lines[1 + n_sig:]:
        if l.startswith("#"):
            k, _, v = l[1:].partition(":")
            k, v = k.strip().lower(), v.strip()
            if k == "age":
                try:
                    info["age"] = float(v)
                except ValueError:
                    pass
            elif k == "sex":
                info["sex"] = v
            elif k == "dx":
                info["dx"] = ";".join(c.strip() for c in v.split(",") if c.strip())
    return rec, n_sig, fs, n_samp, np.array(gains), np.array(baselines), names, info


def load_signal(mat_path, gains, baselines, names):
    x = loadmat(mat_path)["val"].astype(np.float32)
    x = (x - baselines[:, None]) / gains[:, None]  # -> mV
    order = [names.index(l) if l in names else -1 for l in LEADS]
    out = np.zeros((12, x.shape[1]), np.float32)
    for i, j in enumerate(order):
        if j >= 0:
            out[i] = x[j]
    return np.nan_to_num(out)


def filter_resample(x, fs):
    sos = butter(2, [0.5, min(45.0, fs / 2 - 1)], btype="band", fs=fs, output="sos")
    x = sosfiltfilt(sos, x, axis=1)
    fs_i = int(round(fs))
    g = gcd(TARGET_FS, fs_i)
    return resample_poly(x, TARGET_FS // g, fs_i // g, axis=1).astype(np.float32)


def fit_length(x, L):
    if x.shape[1] >= L:
        return x[:, :L]
    return np.pad(x, ((0, 0), (0, L - x.shape[1])))


def find_headers(roots):
    for root in roots:
        for d, _, files in os.walk(root):
            for f in files:
                if f.endswith(".hea") and os.path.exists(os.path.join(d, f[:-4] + ".mat")):
                    yield os.path.join(d, f)


def process_one(h, L):
    rec = os.path.basename(h)[:-4]
    src = source_of(rec)
    try:
        _, n_sig, fs, n_samp, gains, base, names, info = parse_header(h)
        x = load_signal(h[:-4] + ".mat", gains, base, names)
        if x.shape[1] < fs * 5 or np.abs(x).max() < 1e-3:
            return "short_or_flat", rec, src, None, None
        x = filter_resample(x, fs)
        n100 = min(x.shape[1], L)
        x = fit_length(x, L)
    except Exception as e:  # corrupt files exist in the public releases
        print("ERR", rec, e)
        return "error", rec, src, None, None
    row = dict(record=rec, source=src, fs_orig=fs, n_orig=n_samp, n100=n100, **info)
    return "ok", rec, src, x.astype(np.float16), row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=["/kaggle/input"])
    ap.add_argument("--out", default="/kaggle/working/prep")
    ap.add_argument("--seconds", type=float, default=20.0,
                    help="stored length; training uses random 10 s crops inside the valid part (n100)")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--skip", nargs="*", default=["INCART"], help="sources to skip")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    L = int(args.seconds * TARGET_FS)

    headers = sorted(find_headers(args.roots))
    print(f"found {len(headers)} header/mat pairs")
    sigs, rows, skipped = [], [], Counter()
    todo, seen = [], set()
    for h in headers:  # skip/dedup sequentially so the kept records match the serial version
        rec = os.path.basename(h)[:-4]
        src = source_of(rec)
        if src in args.skip or rec in seen:
            skipped[src if src in args.skip else "duplicate"] += 1
            continue
        seen.add(rec)
        todo.append(h)
    with Pool(args.workers) as pool:  # I/O + scipy bound: one process per core, output order preserved
        for i, (status, rec, src, x, row) in enumerate(
                pool.imap(partial(process_one, L=L), todo, chunksize=64)):
            if status != "ok":
                skipped[status] += 1
                continue
            sigs.append(x)
            rows.append(row)
            if i % 5000 == 0:
                print(i, rec, src, flush=True)

    X = np.stack(sigs)
    meta = pd.DataFrame(rows)
    np.save(os.path.join(args.out, "signals.npy"), X)
    meta.to_csv(os.path.join(args.out, "meta.csv"), index=False)

    counts = defaultdict(Counter)
    for src, dx in zip(meta.source, meta.dx.fillna("")):
        for c in filter(None, str(dx).split(";")):
            counts[c][src] += 1
    cc = pd.DataFrame(counts).T.fillna(0).astype(int)
    cc["total"] = cc.sum(axis=1)
    cc.sort_values("total", ascending=False).to_csv(os.path.join(args.out, "code_counts.csv"))

    print("signals", X.shape, X.dtype, f"{X.nbytes / 1e9:.2f} GB")
    print(meta.source.value_counts())
    print("skipped", dict(skipped))


if __name__ == "__main__":
    main()
