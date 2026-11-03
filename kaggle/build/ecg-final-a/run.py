import glob, os, sys, subprocess
CODE = os.path.dirname(glob.glob("/kaggle/input/**/train.py", recursive=True)[0])
sys.path.insert(0, CODE)
print("code at", CODE)
PREP = glob.glob('/kaggle/input/**/prep/signals.npy', recursive=True)[0].rsplit('/', 1)[0]
import time, train
CMDS = [
"--prep {PREP} --mode loso --epochs 30 --target CPSC --head text --text_emb {EMB} --labels mask_pseudo --hier_soft --pseudo_soft --pseudo_start 3 --ema_warmup --mixstyle --diag --residual both --eval_ema --seed 0 --tag loso-CPSC-oursv2-s0",
"--prep {PREP} --mode loso --epochs 30 --target PTBXL --head text --text_emb {EMB} --labels mask_pseudo --hier_soft --pseudo_soft --pseudo_start 3 --ema_warmup --mixstyle --diag --residual both --eval_ema --seed 0 --tag loso-PTBXL-oursv2-s0",
"--prep {PREP} --mode loso --epochs 30 --target SHAONING --head text --text_emb {EMB} --labels mask_pseudo --hier_soft --pseudo_soft --pseudo_start 3 --ema_warmup --mixstyle --diag --residual both --eval_ema --seed 0 --tag loso-SHAONING-oursv2-s0",
"--prep {PREP} --mode loso --epochs 30 --target CPSC --head linear --labels naive --eval_ema --ema_warmup --seed 0 --tag loso-CPSC-naiveema-s0",
"--prep {PREP} --mode loso --epochs 30 --target CPSC --head linear --labels mask --eval_ema --ema_warmup --seed 0 --tag loso-CPSC-maskema-s0",
"--prep {PREP} --mode loso --epochs 30 --target CPSC --head linear --labels mask --hierarchy --eval_ema --ema_warmup --seed 0 --tag loso-CPSC-listyleema-s0",
"--prep {PREP} --mode loso --epochs 30 --target CPSC --head linear --labels mask --hierarchy --dg coral --eval_ema --ema_warmup --seed 0 --tag loso-CPSC-coralema-s0",
"--prep {PREP} --mode loso --epochs 30 --target CPSC --head linear --labels mask --hierarchy --dg dann --eval_ema --ema_warmup --seed 0 --tag loso-CPSC-dannema-s0",
"--prep {PREP} --mode loso --epochs 30 --target CPSC --head linear --labels mask --hierarchy --dg groupdro --eval_ema --ema_warmup --seed 0 --tag loso-CPSC-groupdroema-s0",
"--prep {PREP} --mode loso --epochs 30 --target CPSC --head linear --labels mask --hierarchy --mixstyle --eval_ema --ema_warmup --seed 0 --tag loso-CPSC-mixstyleema-s0",
"--prep {PREP} --mode loso --epochs 30 --target PTBXL --head linear --labels naive --eval_ema --ema_warmup --seed 0 --tag loso-PTBXL-naiveema-s0",
"--prep {PREP} --mode loso --epochs 30 --target PTBXL --head linear --labels mask --eval_ema --ema_warmup --seed 0 --tag loso-PTBXL-maskema-s0",
"--prep {PREP} --mode loso --epochs 30 --target PTBXL --head linear --labels mask --hierarchy --eval_ema --ema_warmup --seed 0 --tag loso-PTBXL-listyleema-s0",
"--prep {PREP} --mode loso --epochs 30 --target PTBXL --head linear --labels mask --hierarchy --dg coral --eval_ema --ema_warmup --seed 0 --tag loso-PTBXL-coralema-s0",
"--prep {PREP} --mode loso --epochs 30 --target PTBXL --head linear --labels mask --hierarchy --dg dann --eval_ema --ema_warmup --seed 0 --tag loso-PTBXL-dannema-s0",
"--prep {PREP} --mode loso --epochs 30 --target PTBXL --head linear --labels mask --hierarchy --dg groupdro --eval_ema --ema_warmup --seed 0 --tag loso-PTBXL-groupdroema-s0",
"--prep {PREP} --mode loso --epochs 30 --target PTBXL --head linear --labels mask --hierarchy --mixstyle --eval_ema --ema_warmup --seed 0 --tag loso-PTBXL-mixstyleema-s0",
"--prep {PREP} --mode loso --epochs 30 --target CPSC --head text --text_emb {EMB} --labels mask_pseudo --hier_soft --pseudo_soft --pseudo_start 3 --ema_warmup --mixstyle --diag --residual both --eval_ema --seed 1 --tag loso-CPSC-oursv2-s1",
"--prep {PREP} --mode loso --epochs 30 --target PTBXL --head text --text_emb {EMB} --labels mask_pseudo --hier_soft --pseudo_soft --pseudo_start 3 --ema_warmup --mixstyle --diag --residual both --eval_ema --seed 1 --tag loso-PTBXL-oursv2-s1",
"--prep {PREP} --mode loso --epochs 30 --target GEORGIA --head text --text_emb {EMB} --labels mask_pseudo --hier_soft --pseudo_soft --pseudo_start 3 --ema_warmup --mixstyle --diag --residual both --eval_ema --seed 1 --tag loso-GEORGIA-oursv2-s1",
"--prep {PREP} --mode loso --epochs 30 --target SHAONING --head text --text_emb {EMB} --labels mask_pseudo --hier_soft --pseudo_soft --pseudo_start 3 --ema_warmup --mixstyle --diag --residual both --eval_ema --seed 1 --tag loso-SHAONING-oursv2-s1",
"--prep {PREP} --mode loso --epochs 30 --target CPSC --head text --text_emb {EMB} --labels mask_pseudo --hier_soft --pseudo_soft --pseudo_start 3 --ema_warmup --mixstyle --diag --residual both --eval_ema --seed 2 --tag loso-CPSC-oursv2-s2",
"--prep {PREP} --mode loso --epochs 30 --target PTBXL --head text --text_emb {EMB} --labels mask_pseudo --hier_soft --pseudo_soft --pseudo_start 3 --ema_warmup --mixstyle --diag --residual both --eval_ema --seed 2 --tag loso-PTBXL-oursv2-s2",
"--prep {PREP} --mode loso --epochs 30 --target GEORGIA --head text --text_emb {EMB} --labels mask_pseudo --hier_soft --pseudo_soft --pseudo_start 3 --ema_warmup --mixstyle --diag --residual both --eval_ema --seed 2 --tag loso-GEORGIA-oursv2-s2",
"--prep {PREP} --mode loso --epochs 30 --target SHAONING --head text --text_emb {EMB} --labels mask_pseudo --hier_soft --pseudo_soft --pseudo_start 3 --ema_warmup --mixstyle --diag --residual both --eval_ema --seed 2 --tag loso-SHAONING-oursv2-s2"
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
