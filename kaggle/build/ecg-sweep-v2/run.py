import glob, os, sys, subprocess
CODE = os.path.dirname(glob.glob("/kaggle/input/**/train.py", recursive=True)[0])
sys.path.insert(0, CODE)
print("code at", CODE)
PREP = glob.glob('/kaggle/input/**/prep/signals.npy', recursive=True)[0].rsplit('/', 1)[0]
import time, train
CMDS = [
"--prep {PREP} --mode loso --epochs 30 --target GEORGIA --head text --text_emb {EMB} --labels mask_pseudo --hier_soft --pseudo_soft --pseudo_start 3 --ema_warmup --mixstyle --diag --residual both --eval_ema --seed 0 --tag sweep-GEORGIA-v2-both-ema-s0",
"--prep {PREP} --mode loso --epochs 30 --target GEORGIA --head text --text_emb {EMB} --labels mask_pseudo --hier_soft --pseudo_soft --pseudo_start 3 --ema_warmup --mixstyle --diag --residual both --seed 0 --tag sweep-GEORGIA-v2-both-s0",
"--prep {PREP} --mode loso --epochs 30 --target GEORGIA --head text --text_emb {EMB} --labels mask_pseudo --hier_soft --pseudo_soft --pseudo_start 3 --ema_warmup --mixstyle --diag --residual query --eval_ema --seed 0 --tag sweep-GEORGIA-v2-query-ema-s0",
"--prep {PREP} --mode loso --epochs 30 --target GEORGIA --head text --text_emb {EMB} --labels mask_pseudo --hier_soft --pseudo_soft --pseudo_start 3 --ema_warmup --mixstyle --diag --eval_ema --seed 0 --tag sweep-GEORGIA-v2-none-ema-s0",
"--prep {PREP} --mode loso --epochs 30 --target GEORGIA --head text --text_emb {EMB} --labels mask_pseudo --hier_soft --pseudo_soft --pseudo_start 3 --ema_warmup --mixstyle --diag --residual both --eval_ema --holdout 47665007 251146004 698252002 164917005 10370003 --seed 0 --tag sweep-GEORGIA-v2-both-ema-zsproxy-s0",
"--prep {PREP} --mode loso --epochs 30 --target GEORGIA --head text --text_emb {EMB} --labels mask_pseudo --hier_soft --pseudo_soft --pseudo_start 3 --ema_warmup --mixstyle --diag --eval_ema --holdout 47665007 251146004 698252002 164917005 10370003 --seed 0 --tag sweep-GEORGIA-v2-none-ema-zsproxy-s0"
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
