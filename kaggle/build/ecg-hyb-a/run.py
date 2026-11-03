import glob, os, sys, subprocess
CODE = os.path.dirname(glob.glob("/kaggle/input/**/train.py", recursive=True)[0])
sys.path.insert(0, CODE)
print("code at", CODE)
PREP = glob.glob('/kaggle/input/**/prep/signals.npy', recursive=True)[0].rsplit('/', 1)[0]
import time, train
CMDS = [
"--prep {PREP} --mode loso --epochs 30 --target CPSC --head hybrid --text_emb {EMB} --labels mask --hierarchy --eval_ema --ema_warmup --text_w 1.0 --seen_score avg --seed 0 --tag loso-CPSC-hyb-avg-s0",
"--prep {PREP} --mode loso --epochs 30 --target PTBXL --head hybrid --text_emb {EMB} --labels mask --hierarchy --eval_ema --ema_warmup --text_w 1.0 --seen_score avg --seed 0 --tag loso-PTBXL-hyb-avg-s0",
"--prep {PREP} --mode loso --epochs 30 --target SHAONING --head hybrid --text_emb {EMB} --labels mask --hierarchy --eval_ema --ema_warmup --text_w 1.0 --seen_score avg --seed 0 --tag loso-SHAONING-hyb-avg-s0",
"--prep {PREP} --mode loso --epochs 30 --target CPSC --head hybrid --text_emb {EMB} --labels mask --hierarchy --eval_ema --ema_warmup --text_w 1.0 --seen_score avg --seed 1 --tag loso-CPSC-hyb-avg-s1",
"--prep {PREP} --mode loso --epochs 30 --target PTBXL --head hybrid --text_emb {EMB} --labels mask --hierarchy --eval_ema --ema_warmup --text_w 1.0 --seen_score avg --seed 1 --tag loso-PTBXL-hyb-avg-s1",
"--prep {PREP} --mode loso --epochs 30 --target GEORGIA --head hybrid --text_emb {EMB} --labels mask --hierarchy --eval_ema --ema_warmup --text_w 1.0 --seen_score avg --seed 1 --tag loso-GEORGIA-hyb-avg-s1",
"--prep {PREP} --mode loso --epochs 30 --target SHAONING --head hybrid --text_emb {EMB} --labels mask --hierarchy --eval_ema --ema_warmup --text_w 1.0 --seen_score avg --seed 1 --tag loso-SHAONING-hyb-avg-s1",
"--prep {PREP} --mode loso --epochs 30 --target CPSC --head linear --labels mask --eval_ema --ema_warmup --seed 1 --tag loso-CPSC-maskema-s1",
"--prep {PREP} --mode loso --epochs 30 --target PTBXL --head linear --labels mask --eval_ema --ema_warmup --seed 1 --tag loso-PTBXL-maskema-s1",
"--prep {PREP} --mode loso --epochs 30 --target GEORGIA --head linear --labels mask --eval_ema --ema_warmup --seed 1 --tag loso-GEORGIA-maskema-s1",
"--prep {PREP} --mode loso --epochs 30 --target SHAONING --head linear --labels mask --eval_ema --ema_warmup --seed 1 --tag loso-SHAONING-maskema-s1"
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
