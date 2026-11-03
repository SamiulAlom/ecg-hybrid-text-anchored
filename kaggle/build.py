"""Build Kaggle upload folders: a private code dataset + preprocessing and experiment kernels.

Usage:  python kaggle/build.py <kaggle_username>
Then:   kaggle datasets create -p kaggle/build/code        (first time; later: datasets version -m ...)
        kaggle kernels push -p kaggle/build/prep
        kaggle kernels push -p kaggle/build/run_<batch>
"""
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "kaggle", "build")
DATASETS = ["gamalasran/physionet-challenge-2020", "bjoernjostein/shaoxing-and-ningbo-first-hospital-database"]

FIND_CODE = '''import glob, os, sys, subprocess
CODE = os.path.dirname(glob.glob("/kaggle/input/**/train.py", recursive=True)[0])
sys.path.insert(0, CODE)
print("code at", CODE)
'''


def write_kernel(user, slug, title, code, gpu, kernel_sources=(), datasets=()):
    d = os.path.join(BUILD, slug)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "run.py"), "w") as f:
        f.write(code)
    meta = dict(id=f"{user}/{slug}", title=title, code_file="run.py", language="python", kernel_type="script",
                is_private=True, enable_gpu=gpu, enable_internet=False,
                dataset_sources=list(datasets), kernel_sources=list(kernel_sources), competition_sources=[])
    with open(os.path.join(d, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=1)


def main(user):
    shutil.rmtree(BUILD, ignore_errors=True)
    # 1) code dataset
    cd = os.path.join(BUILD, "code")
    os.makedirs(cd)
    for f in os.listdir(os.path.join(ROOT, "src")):
        if f.endswith((".py", ".npz")):
            shutil.copy(os.path.join(ROOT, "src", f), cd)
    with open(os.path.join(cd, "dataset-metadata.json"), "w") as f:
        json.dump(dict(title="ecg-paper-code", id=f"{user}/ecg-paper-code", licenses=[dict(name="CC0-1.0")]), f)

    code_ds = f"{user}/ecg-paper-code"
    # 2) preprocessing kernel (CPU)
    write_kernel(user, "ecg-prep", "ecg-prep", FIND_CODE + '''
subprocess.run([sys.executable, os.path.join(CODE, "preprocess.py"), "--roots", "/kaggle/input",
                "--out", "/kaggle/working/prep"], check=True)
''', gpu=False, datasets=DATASETS + [code_ds])

    # 3) experiment kernels (GPU), one per batch in experiments.py
    sys.path.insert(0, os.path.join(ROOT, "kaggle"))
    from experiments import BATCHES
    for bname, cmds in BATCHES.items():
        body = FIND_CODE + "PREP = glob.glob('/kaggle/input/**/prep/signals.npy', recursive=True)[0].rsplit('/', 1)[0]\n"
        body += "import time, train\n"
        body += f"CMDS = {json.dumps(cmds, indent=0)}\n"
        body += '''SUB = {"{PREP}": PREP, "{EMB}": os.path.join(CODE, "text_emb_biolord_full.npz"),
       "{EMB_NAME}": os.path.join(CODE, "text_emb_biolord_name.npz"),
       "{EMB_MPNET}": os.path.join(CODE, "text_emb_mpnet_full.npz")}
OUT, t0 = "/kaggle/working/results", time.time()
for i, c in enumerate(CMDS):
    for k, v in SUB.items():
        c = c.replace(k, v)
    tag = c.split("--tag ")[1].split()[0]
    if os.path.exists(f"{OUT}/{tag}.json"):
        continue
    if (time.time() - t0) / 3600 > 11.0:
        print("time budget reached; remaining runs skipped", flush=True); break
    print(f"=== {i + 1}/{len(CMDS)} {tag} ===", flush=True)
    try:
        train.main(c.split() + ["--out", OUT])
    except Exception as e:
        print("FAILED", tag, repr(e), flush=True)
'''
        write_kernel(user, f"ecg-run-{bname}", f"ecg-run-{bname}", body, gpu=True,
                     kernel_sources=[f"{user}/ecg-prep"], datasets=[code_ds])
    # 4) MERL zero-shot comparator (inference only; needs internet for code + weights)
    with open(os.path.join(ROOT, "kaggle", "merl", "merl_zeroshot.py"), encoding="utf-8") as f:
        merl = f.read()
    write_kernel(user, "ecg-merl", "ecg-merl", merl, gpu=True,
                 kernel_sources=[f"{user}/ecg-prep"], datasets=[code_ds])
    meta_p = os.path.join(BUILD, "ecg-merl", "kernel-metadata.json")
    meta = json.load(open(meta_p))
    meta["enable_internet"] = True
    json.dump(meta, open(meta_p, "w"), indent=1)
    print("built", sorted(os.listdir(BUILD)))


if __name__ == "__main__":
    main(sys.argv[1])
