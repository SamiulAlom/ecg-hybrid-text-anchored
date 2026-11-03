"""Write a tiny fake PhysioNet-style dataset (WFDB .hea + MATLAB v4 .mat) for local smoke tests.

The signals are random noise with a crude periodic beat; they are NOT used for any reported result.
"""
import argparse
import os

import numpy as np
from scipy.io import savemat

SPEC = {  # prefix: (fs, seconds, codes this fake source "annotates")
    "A": (500, 12, ["426783006", "164889003", "270492004", "59118001", "427172004"]),
    "HR": (500, 10, ["426783006", "164889003", "164934002", "39732003", "164873001"]),
    "E": (500, 10, ["426783006", "164889003", "427084000", "426177001", "164934002"]),
    "JS": (500, 10, ["426783006", "426177001", "427084000", "164890007", "59118001"]),
}
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_source", type=int, default=60)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    for prefix, (fs, sec, codes) in SPEC.items():
        d = os.path.join(args.out, f"src_{prefix}")
        os.makedirs(d, exist_ok=True)
        for k in range(1, args.per_source + 1):
            rec = f"{prefix}{k:05d}"
            n = fs * sec
            t = np.arange(n) / fs
            hr = rng.uniform(0.8, 2.0)
            beat = np.exp(-((t * hr) % 1 - 0.3) ** 2 / 0.001)
            x = (beat[None] * rng.uniform(0.5, 1.5, (12, 1)) + 0.05 * rng.standard_normal((12, n)))
            savemat(os.path.join(d, rec + ".mat"), {"val": (x * 1000).astype(np.int16)}, format="4")
            dx = rng.choice(codes, size=rng.integers(1, 3), replace=False)
            with open(os.path.join(d, rec + ".hea"), "w") as f:
                f.write(f"{rec} 12 {fs} {n} 05-Feb-2020 11:39:16\n")
                for lead in LEADS:
                    f.write(f"{rec}.mat 16+24 1000/mV 16 0 0 0 0 {lead}\n")
                f.write(f"#Age: {rng.integers(20, 90)}\n#Sex: {rng.choice(['Male', 'Female'])}\n")
                f.write(f"#Dx: {','.join(dx)}\n#Rx: Unknown\n#Hx: Unknown\n#Sx: Unknown\n")
    print("synthetic data written to", args.out)


if __name__ == "__main__":
    main()
