import glob, os, sys, subprocess
CODE = os.path.dirname(glob.glob("/kaggle/input/**/train.py", recursive=True)[0])
sys.path.insert(0, CODE)
print("code at", CODE)
PREP = glob.glob('/kaggle/input/**/prep/signals.npy', recursive=True)[0].rsplit('/', 1)[0]
import time, train
CMDS = [
"--prep {PREP} --mode loso --epochs 30 --target GEORGIA --head hybrid --text_emb {EMB} --labels mask --hierarchy --eval_ema --ema_warmup --text_w 1.0 --seen_score linear --seed 0 --tag hsweep-GEORGIA-hyb-lin-s0",
"--prep {PREP} --mode loso --epochs 30 --target GEORGIA --head hybrid --text_emb {EMB} --labels mask --hierarchy --eval_ema --ema_warmup --text_w 1.0 --seen_score avg --seed 0 --tag hsweep-GEORGIA-hyb-avg-s0",
"--prep {PREP} --mode loso --epochs 30 --target GEORGIA --head hybrid --text_emb {EMB} --labels mask --hierarchy --eval_ema --ema_warmup --text_w 1.0 --seen_score avg --text_detach --seed 0 --tag hsweep-GEORGIA-hyb-avg-detach-s0",
"--prep {PREP} --mode loso --epochs 30 --target GEORGIA --head hybrid --text_emb {EMB} --labels mask_pseudo --pseudo_soft --pseudo_start 3 --hierarchy --eval_ema --ema_warmup --text_w 1.0 --seen_score avg --diag --seed 0 --tag hsweep-GEORGIA-hyb-avg-pseudo-s0",
"--prep {PREP} --mode loso --epochs 30 --target GEORGIA --head hybrid --text_emb {EMB} --labels mask --hierarchy --eval_ema --ema_warmup --text_w 1.0 --seen_score avg --holdout 47665007 251146004 698252002 164917005 10370003 --seed 0 --tag hsweep-GEORGIA-hyb-avg-zsproxy-s0",
"--prep {PREP} --mode loso --epochs 30 --target GEORGIA --head hybrid --text_emb {EMB} --labels mask --hierarchy --eval_ema --ema_warmup --text_w 1.0 --seen_score avg --text_detach --holdout 47665007 251146004 698252002 164917005 10370003 --seed 0 --tag hsweep-GEORGIA-hyb-avg-detach-zsproxy-s0"
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
