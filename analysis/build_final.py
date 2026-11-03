"""Collect the final runs from all result folders into results/final with the canonical tags aggregate.py expects.

Final method set (pre-registered, see kaggle/sweep_method.md): every baseline with an EMA teacher, and the hybrid
head (hyb-avg = H2) as "ours". GEORGIA seed 0 of H2 and of the hybrid ablations comes from the hybrid sweep (hsweep-GEORGIA-*-s0), and of the
text-only v2 ablation from the v2 sweep (S1, sweep-GEORGIA-v2-both-ema-s0).
Only the tag (file name and json "name") is changed; metrics and predictions are copied unchanged.
"""
import glob
import json
import os
import re
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "final")
MAIN = {"naiveema": "naive", "maskema": "mask", "listyleema": "li2024", "coralema": "coral", "dannema": "dann",
        "groupdroema": "groupdro", "mixstyleema": "mixstyle", "hyb-avg": "ours"}
ABL = {"hyb-lin": "ours-hyblinear", "hyb-avg-detach": "ours-textdetach", "hyb-avg-pseudo": "ours-pseudo",
       "oursv2": "ours-textonly"}
E2 = {"hyb-avg": "ours", "listyle": "li2024", "naive": "naive"}


def new_tag(name):
    if name == "sweep-GEORGIA-v2-both-ema-s0":  # S1 of the v2 sweep = the final v2 configuration
        return "abl-GEORGIA-ours-textonly-s0"
    m = re.match(r"^(loso|hsweep)-(\w+?)-(.+)-s(\d)$", name)
    if m and m[3] in MAIN and (m[1] == "loso" or (m[3] == "hyb-avg" and m[2] == "GEORGIA" and m[4] == "0")):
        return f"loso-{m[2]}-{MAIN[m[3]]}-s{m[4]}"
    if m and m[3] in ABL and (m[1] == "loso" or (m[2] == "GEORGIA" and m[4] == "0")):
        return f"abl-{m[2]}-{ABL[m[3]]}-s{m[4]}"
    m = re.match(r"^e2fix-(\w+?)-(.+)-r([\d.]+)-s(\d)$", name)
    if m and m[2] in E2:
        return f"e4-{m[1]}-{E2[m[2]]}r{m[3]}-s{m[4]}"
    return None


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT)
    got = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "**", "*.json"), recursive=True)):
        if os.sep + "final" + os.sep in f:
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if not isinstance(d, dict) or "config" not in d:
            continue
        t = new_tag(d.get("name", ""))
        if t is None:
            continue
        if t in got:
            print(f"duplicate {t}: {got[t]} and {f}; keeping the first")
            continue
        got[t] = f
        d["name"], d["source_tag"] = t, os.path.basename(f)[:-5]
        json.dump(d, open(os.path.join(OUT, t + ".json"), "w"))
        npz = f[:-5] + "_test.npz"
        if os.path.exists(npz):
            shutil.copy(npz, os.path.join(OUT, t + "_test.npz"))
        else:
            print("no npz for", f)
    print(len(got), "runs ->", OUT)


if __name__ == "__main__":
    main()
