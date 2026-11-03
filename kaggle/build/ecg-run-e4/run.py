import glob, os, sys, subprocess
CODE = os.path.dirname(glob.glob("/kaggle/input/**/train.py", recursive=True)[0])
sys.path.insert(0, CODE)
print("code at", CODE)
PREP = glob.glob('/kaggle/input/**/prep/signals.npy', recursive=True)[0].rsplit('/', 1)[0]
import time, train
CMDS = [
"--prep {PREP} --mode indist --target PTBXL --epochs 30 --head linear --labels naive --drop_rate 0.0 --seed 0 --tag e4-PTBXL-naive-r0.0-s0",
"--prep {PREP} --mode indist --target PTBXL --epochs 30 --head linear --labels mask --hierarchy --drop_rate 0.0 --seed 0 --tag e4-PTBXL-listyle-r0.0-s0",
"--prep {PREP} --mode indist --target PTBXL --epochs 30 --head text --text_emb {EMB} --labels mask_pseudo --hierarchy --mixstyle --drop_rate 0.0 --seed 0 --tag e4-PTBXL-ours-r0.0-s0",
"--prep {PREP} --mode indist --target PTBXL --epochs 30 --head linear --labels naive --drop_rate 0.25 --seed 0 --tag e4-PTBXL-naive-r0.25-s0",
"--prep {PREP} --mode indist --target PTBXL --epochs 30 --head linear --labels mask --hierarchy --drop_rate 0.25 --seed 0 --tag e4-PTBXL-listyle-r0.25-s0",
"--prep {PREP} --mode indist --target PTBXL --epochs 30 --head text --text_emb {EMB} --labels mask_pseudo --hierarchy --mixstyle --drop_rate 0.25 --seed 0 --tag e4-PTBXL-ours-r0.25-s0",
"--prep {PREP} --mode indist --target PTBXL --epochs 30 --head linear --labels naive --drop_rate 0.5 --seed 0 --tag e4-PTBXL-naive-r0.5-s0",
"--prep {PREP} --mode indist --target PTBXL --epochs 30 --head linear --labels mask --hierarchy --drop_rate 0.5 --seed 0 --tag e4-PTBXL-listyle-r0.5-s0",
"--prep {PREP} --mode indist --target PTBXL --epochs 30 --head text --text_emb {EMB} --labels mask_pseudo --hierarchy --mixstyle --drop_rate 0.5 --seed 0 --tag e4-PTBXL-ours-r0.5-s0"
]
SUB = {"{PREP}": PREP, "{EMB}": os.path.join(CODE, "text_emb_biolord_full.npz"),
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
