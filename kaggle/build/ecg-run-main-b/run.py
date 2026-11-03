import glob, os, sys, subprocess
CODE = os.path.dirname(glob.glob("/kaggle/input/**/train.py", recursive=True)[0])
sys.path.insert(0, CODE)
print("code at", CODE)
PREP = glob.glob('/kaggle/input/**/prep/signals.npy', recursive=True)[0].rsplit('/', 1)[0]
import time, train
CMDS = [
"--prep {PREP} --mode loso --epochs 30 --target GEORGIA --head linear --labels naive --seed 0 --tag loso-GEORGIA-naive-s0",
"--prep {PREP} --mode loso --epochs 30 --target GEORGIA --head linear --labels mask --seed 0 --tag loso-GEORGIA-mask-s0",
"--prep {PREP} --mode loso --epochs 30 --target GEORGIA --head linear --labels mask --hierarchy --seed 0 --tag loso-GEORGIA-listyle-s0",
"--prep {PREP} --mode loso --epochs 30 --target GEORGIA --head linear --labels mask --hierarchy --dg coral --seed 0 --tag loso-GEORGIA-coral-s0",
"--prep {PREP} --mode loso --epochs 30 --target GEORGIA --head linear --labels mask --hierarchy --dg dann --seed 0 --tag loso-GEORGIA-dann-s0",
"--prep {PREP} --mode loso --epochs 30 --target GEORGIA --head linear --labels mask --hierarchy --dg groupdro --seed 0 --tag loso-GEORGIA-groupdro-s0",
"--prep {PREP} --mode loso --epochs 30 --target GEORGIA --head linear --labels mask --hierarchy --mixstyle --seed 0 --tag loso-GEORGIA-mixstyle-s0",
"--prep {PREP} --mode loso --epochs 30 --target GEORGIA --head text --text_emb {EMB} --labels mask_pseudo --hierarchy --mixstyle --seed 0 --tag loso-GEORGIA-ours-s0",
"--prep {PREP} --mode loso --epochs 30 --target SHAONING --head linear --labels naive --seed 0 --tag loso-SHAONING-naive-s0",
"--prep {PREP} --mode loso --epochs 30 --target SHAONING --head linear --labels mask --seed 0 --tag loso-SHAONING-mask-s0",
"--prep {PREP} --mode loso --epochs 30 --target SHAONING --head linear --labels mask --hierarchy --seed 0 --tag loso-SHAONING-listyle-s0",
"--prep {PREP} --mode loso --epochs 30 --target SHAONING --head linear --labels mask --hierarchy --dg coral --seed 0 --tag loso-SHAONING-coral-s0",
"--prep {PREP} --mode loso --epochs 30 --target SHAONING --head linear --labels mask --hierarchy --dg dann --seed 0 --tag loso-SHAONING-dann-s0",
"--prep {PREP} --mode loso --epochs 30 --target SHAONING --head linear --labels mask --hierarchy --dg groupdro --seed 0 --tag loso-SHAONING-groupdro-s0",
"--prep {PREP} --mode loso --epochs 30 --target SHAONING --head linear --labels mask --hierarchy --mixstyle --seed 0 --tag loso-SHAONING-mixstyle-s0",
"--prep {PREP} --mode loso --epochs 30 --target SHAONING --head text --text_emb {EMB} --labels mask_pseudo --hierarchy --mixstyle --seed 0 --tag loso-SHAONING-ours-s0",
"--prep {PREP} --mode loso --epochs 30 --target GEORGIA --head text --text_emb {EMB} --labels mask --hierarchy --mixstyle --seed 0 --tag loso-GEORGIA-abl-textmask-s0",
"--prep {PREP} --mode loso --epochs 30 --target GEORGIA --head linear --labels mask_pseudo --hierarchy --mixstyle --seed 0 --tag loso-GEORGIA-abl-linpseudo-s0",
"--prep {PREP} --mode loso --epochs 30 --target GEORGIA --head query --labels mask_pseudo --hierarchy --mixstyle --seed 0 --tag loso-GEORGIA-abl-query-s0",
"--prep {PREP} --mode loso --epochs 30 --target GEORGIA --head text --text_emb {EMB} --labels mask_pseudo --mixstyle --seed 0 --tag loso-GEORGIA-abl-noh-s0",
"--prep {PREP} --mode loso --epochs 30 --target GEORGIA --head text --text_emb {EMB} --labels mask_pseudo --hierarchy --seed 0 --tag loso-GEORGIA-abl-nomix-s0",
"--prep {PREP} --mode loso --epochs 30 --target GEORGIA --head text --text_emb {EMB_NAME} --labels mask_pseudo --hierarchy --mixstyle --seed 0 --tag loso-GEORGIA-abl-name-s0",
"--prep {PREP} --mode loso --epochs 30 --target GEORGIA --head text --text_emb {EMB_MPNET} --labels mask_pseudo --hierarchy --mixstyle --seed 0 --tag loso-GEORGIA-abl-mpnet-s0",
"--prep {PREP} --mode loso --epochs 30 --target SHAONING --head text --text_emb {EMB} --labels mask --hierarchy --mixstyle --seed 0 --tag loso-SHAONING-abl-textmask-s0",
"--prep {PREP} --mode loso --epochs 30 --target SHAONING --head linear --labels mask_pseudo --hierarchy --mixstyle --seed 0 --tag loso-SHAONING-abl-linpseudo-s0",
"--prep {PREP} --mode loso --epochs 30 --target SHAONING --head query --labels mask_pseudo --hierarchy --mixstyle --seed 0 --tag loso-SHAONING-abl-query-s0",
"--prep {PREP} --mode loso --epochs 30 --target SHAONING --head text --text_emb {EMB} --labels mask_pseudo --mixstyle --seed 0 --tag loso-SHAONING-abl-noh-s0",
"--prep {PREP} --mode loso --epochs 30 --target SHAONING --head text --text_emb {EMB} --labels mask_pseudo --hierarchy --seed 0 --tag loso-SHAONING-abl-nomix-s0",
"--prep {PREP} --mode loso --epochs 30 --target SHAONING --head text --text_emb {EMB_NAME} --labels mask_pseudo --hierarchy --mixstyle --seed 0 --tag loso-SHAONING-abl-name-s0",
"--prep {PREP} --mode loso --epochs 30 --target SHAONING --head text --text_emb {EMB_MPNET} --labels mask_pseudo --hierarchy --mixstyle --seed 0 --tag loso-SHAONING-abl-mpnet-s0"
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
