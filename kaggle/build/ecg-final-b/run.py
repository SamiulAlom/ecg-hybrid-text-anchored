import glob, os, sys, subprocess
CODE = os.path.dirname(glob.glob("/kaggle/input/**/train.py", recursive=True)[0])
sys.path.insert(0, CODE)
print("code at", CODE)
PREP = glob.glob('/kaggle/input/**/prep/signals.npy', recursive=True)[0].rsplit('/', 1)[0]
import time, train
CMDS = [
"--prep {PREP} --mode loso --epochs 30 --target GEORGIA --head linear --labels naive --eval_ema --ema_warmup --seed 0 --tag loso-GEORGIA-naiveema-s0",
"--prep {PREP} --mode loso --epochs 30 --target GEORGIA --head linear --labels mask --eval_ema --ema_warmup --seed 0 --tag loso-GEORGIA-maskema-s0",
"--prep {PREP} --mode loso --epochs 30 --target GEORGIA --head linear --labels mask --hierarchy --eval_ema --ema_warmup --seed 0 --tag loso-GEORGIA-listyleema-s0",
"--prep {PREP} --mode loso --epochs 30 --target GEORGIA --head linear --labels mask --hierarchy --dg coral --eval_ema --ema_warmup --seed 0 --tag loso-GEORGIA-coralema-s0",
"--prep {PREP} --mode loso --epochs 30 --target GEORGIA --head linear --labels mask --hierarchy --dg dann --eval_ema --ema_warmup --seed 0 --tag loso-GEORGIA-dannema-s0",
"--prep {PREP} --mode loso --epochs 30 --target GEORGIA --head linear --labels mask --hierarchy --dg groupdro --eval_ema --ema_warmup --seed 0 --tag loso-GEORGIA-groupdroema-s0",
"--prep {PREP} --mode loso --epochs 30 --target GEORGIA --head linear --labels mask --hierarchy --mixstyle --eval_ema --ema_warmup --seed 0 --tag loso-GEORGIA-mixstyleema-s0",
"--prep {PREP} --mode loso --epochs 30 --target SHAONING --head linear --labels naive --eval_ema --ema_warmup --seed 0 --tag loso-SHAONING-naiveema-s0",
"--prep {PREP} --mode loso --epochs 30 --target SHAONING --head linear --labels mask --eval_ema --ema_warmup --seed 0 --tag loso-SHAONING-maskema-s0",
"--prep {PREP} --mode loso --epochs 30 --target SHAONING --head linear --labels mask --hierarchy --eval_ema --ema_warmup --seed 0 --tag loso-SHAONING-listyleema-s0",
"--prep {PREP} --mode loso --epochs 30 --target SHAONING --head linear --labels mask --hierarchy --dg coral --eval_ema --ema_warmup --seed 0 --tag loso-SHAONING-coralema-s0",
"--prep {PREP} --mode loso --epochs 30 --target SHAONING --head linear --labels mask --hierarchy --dg dann --eval_ema --ema_warmup --seed 0 --tag loso-SHAONING-dannema-s0",
"--prep {PREP} --mode loso --epochs 30 --target SHAONING --head linear --labels mask --hierarchy --dg groupdro --eval_ema --ema_warmup --seed 0 --tag loso-SHAONING-groupdroema-s0",
"--prep {PREP} --mode loso --epochs 30 --target SHAONING --head linear --labels mask --hierarchy --mixstyle --eval_ema --ema_warmup --seed 0 --tag loso-SHAONING-mixstyleema-s0",
"--prep {PREP} --mode loso --epochs 30 --target CPSC --head linear --labels mask --hierarchy --eval_ema --ema_warmup --seed 1 --tag loso-CPSC-listyleema-s1",
"--prep {PREP} --mode loso --epochs 30 --target CPSC --head linear --labels mask --hierarchy --mixstyle --eval_ema --ema_warmup --seed 1 --tag loso-CPSC-mixstyleema-s1",
"--prep {PREP} --mode loso --epochs 30 --target PTBXL --head linear --labels mask --hierarchy --eval_ema --ema_warmup --seed 1 --tag loso-PTBXL-listyleema-s1",
"--prep {PREP} --mode loso --epochs 30 --target PTBXL --head linear --labels mask --hierarchy --mixstyle --eval_ema --ema_warmup --seed 1 --tag loso-PTBXL-mixstyleema-s1",
"--prep {PREP} --mode loso --epochs 30 --target GEORGIA --head linear --labels mask --hierarchy --eval_ema --ema_warmup --seed 1 --tag loso-GEORGIA-listyleema-s1",
"--prep {PREP} --mode loso --epochs 30 --target GEORGIA --head linear --labels mask --hierarchy --mixstyle --eval_ema --ema_warmup --seed 1 --tag loso-GEORGIA-mixstyleema-s1",
"--prep {PREP} --mode loso --epochs 30 --target SHAONING --head linear --labels mask --hierarchy --eval_ema --ema_warmup --seed 1 --tag loso-SHAONING-listyleema-s1",
"--prep {PREP} --mode loso --epochs 30 --target SHAONING --head linear --labels mask --hierarchy --mixstyle --eval_ema --ema_warmup --seed 1 --tag loso-SHAONING-mixstyleema-s1",
"--prep {PREP} --mode loso --epochs 30 --target CPSC --head linear --labels mask --hierarchy --eval_ema --ema_warmup --seed 2 --tag loso-CPSC-listyleema-s2",
"--prep {PREP} --mode loso --epochs 30 --target CPSC --head linear --labels mask --hierarchy --mixstyle --eval_ema --ema_warmup --seed 2 --tag loso-CPSC-mixstyleema-s2",
"--prep {PREP} --mode loso --epochs 30 --target PTBXL --head linear --labels mask --hierarchy --eval_ema --ema_warmup --seed 2 --tag loso-PTBXL-listyleema-s2",
"--prep {PREP} --mode loso --epochs 30 --target PTBXL --head linear --labels mask --hierarchy --mixstyle --eval_ema --ema_warmup --seed 2 --tag loso-PTBXL-mixstyleema-s2",
"--prep {PREP} --mode loso --epochs 30 --target GEORGIA --head linear --labels mask --hierarchy --eval_ema --ema_warmup --seed 2 --tag loso-GEORGIA-listyleema-s2",
"--prep {PREP} --mode loso --epochs 30 --target GEORGIA --head linear --labels mask --hierarchy --mixstyle --eval_ema --ema_warmup --seed 2 --tag loso-GEORGIA-mixstyleema-s2",
"--prep {PREP} --mode loso --epochs 30 --target SHAONING --head linear --labels mask --hierarchy --eval_ema --ema_warmup --seed 2 --tag loso-SHAONING-listyleema-s2",
"--prep {PREP} --mode loso --epochs 30 --target SHAONING --head linear --labels mask --hierarchy --mixstyle --eval_ema --ema_warmup --seed 2 --tag loso-SHAONING-mixstyleema-s2",
"--prep {PREP} --mode indist --target PTBXL --epochs 30 --head linear --labels naive --eval_ema --ema_warmup --drop_rate 0.0 --seed 0 --tag e4-PTBXL-naive-r0.0-s0",
"--prep {PREP} --mode indist --target PTBXL --epochs 30 --head linear --labels mask --hierarchy --eval_ema --ema_warmup --drop_rate 0.0 --seed 0 --tag e4-PTBXL-listyle-r0.0-s0",
"--prep {PREP} --mode indist --target PTBXL --epochs 30 --head text --text_emb {EMB} --labels mask_pseudo --hier_soft --pseudo_soft --pseudo_start 3 --ema_warmup --mixstyle --diag --residual both --eval_ema --drop_rate 0.0 --seed 0 --tag e4-PTBXL-oursv2-r0.0-s0",
"--prep {PREP} --mode indist --target PTBXL --epochs 30 --head linear --labels naive --eval_ema --ema_warmup --drop_rate 0.25 --seed 0 --tag e4-PTBXL-naive-r0.25-s0",
"--prep {PREP} --mode indist --target PTBXL --epochs 30 --head linear --labels mask --hierarchy --eval_ema --ema_warmup --drop_rate 0.25 --seed 0 --tag e4-PTBXL-listyle-r0.25-s0",
"--prep {PREP} --mode indist --target PTBXL --epochs 30 --head text --text_emb {EMB} --labels mask_pseudo --hier_soft --pseudo_soft --pseudo_start 3 --ema_warmup --mixstyle --diag --residual both --eval_ema --drop_rate 0.25 --seed 0 --tag e4-PTBXL-oursv2-r0.25-s0",
"--prep {PREP} --mode indist --target PTBXL --epochs 30 --head linear --labels naive --eval_ema --ema_warmup --drop_rate 0.5 --seed 0 --tag e4-PTBXL-naive-r0.5-s0",
"--prep {PREP} --mode indist --target PTBXL --epochs 30 --head linear --labels mask --hierarchy --eval_ema --ema_warmup --drop_rate 0.5 --seed 0 --tag e4-PTBXL-listyle-r0.5-s0",
"--prep {PREP} --mode indist --target PTBXL --epochs 30 --head text --text_emb {EMB} --labels mask_pseudo --hier_soft --pseudo_soft --pseudo_start 3 --ema_warmup --mixstyle --diag --residual both --eval_ema --drop_rate 0.5 --seed 0 --tag e4-PTBXL-oursv2-r0.5-s0"
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
