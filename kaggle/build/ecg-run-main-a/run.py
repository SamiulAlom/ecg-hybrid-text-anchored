import glob, os, sys, subprocess
CODE = os.path.dirname(glob.glob("/kaggle/input/**/train.py", recursive=True)[0])
sys.path.insert(0, CODE)
print("code at", CODE)
PREP = glob.glob('/kaggle/input/**/prep/signals.npy', recursive=True)[0].rsplit('/', 1)[0]
import time, train
CMDS = [
"--prep {PREP} --mode loso --epochs 30 --target CPSC --head linear --labels naive --seed 0 --tag loso-CPSC-naive-s0",
"--prep {PREP} --mode loso --epochs 30 --target CPSC --head linear --labels mask --seed 0 --tag loso-CPSC-mask-s0",
"--prep {PREP} --mode loso --epochs 30 --target CPSC --head linear --labels mask --hierarchy --seed 0 --tag loso-CPSC-listyle-s0",
"--prep {PREP} --mode loso --epochs 30 --target CPSC --head linear --labels mask --hierarchy --dg coral --seed 0 --tag loso-CPSC-coral-s0",
"--prep {PREP} --mode loso --epochs 30 --target CPSC --head linear --labels mask --hierarchy --dg dann --seed 0 --tag loso-CPSC-dann-s0",
"--prep {PREP} --mode loso --epochs 30 --target CPSC --head linear --labels mask --hierarchy --dg groupdro --seed 0 --tag loso-CPSC-groupdro-s0",
"--prep {PREP} --mode loso --epochs 30 --target CPSC --head linear --labels mask --hierarchy --mixstyle --seed 0 --tag loso-CPSC-mixstyle-s0",
"--prep {PREP} --mode loso --epochs 30 --target CPSC --head text --text_emb {EMB} --labels mask_pseudo --hierarchy --mixstyle --seed 0 --tag loso-CPSC-ours-s0",
"--prep {PREP} --mode loso --epochs 30 --target PTBXL --head linear --labels naive --seed 0 --tag loso-PTBXL-naive-s0",
"--prep {PREP} --mode loso --epochs 30 --target PTBXL --head linear --labels mask --seed 0 --tag loso-PTBXL-mask-s0",
"--prep {PREP} --mode loso --epochs 30 --target PTBXL --head linear --labels mask --hierarchy --seed 0 --tag loso-PTBXL-listyle-s0",
"--prep {PREP} --mode loso --epochs 30 --target PTBXL --head linear --labels mask --hierarchy --dg coral --seed 0 --tag loso-PTBXL-coral-s0",
"--prep {PREP} --mode loso --epochs 30 --target PTBXL --head linear --labels mask --hierarchy --dg dann --seed 0 --tag loso-PTBXL-dann-s0",
"--prep {PREP} --mode loso --epochs 30 --target PTBXL --head linear --labels mask --hierarchy --dg groupdro --seed 0 --tag loso-PTBXL-groupdro-s0",
"--prep {PREP} --mode loso --epochs 30 --target PTBXL --head linear --labels mask --hierarchy --mixstyle --seed 0 --tag loso-PTBXL-mixstyle-s0",
"--prep {PREP} --mode loso --epochs 30 --target PTBXL --head text --text_emb {EMB} --labels mask_pseudo --hierarchy --mixstyle --seed 0 --tag loso-PTBXL-ours-s0",
"--prep {PREP} --mode loso --epochs 30 --target CPSC --head text --text_emb {EMB} --labels mask --hierarchy --mixstyle --seed 0 --tag loso-CPSC-abl-textmask-s0",
"--prep {PREP} --mode loso --epochs 30 --target CPSC --head linear --labels mask_pseudo --hierarchy --mixstyle --seed 0 --tag loso-CPSC-abl-linpseudo-s0",
"--prep {PREP} --mode loso --epochs 30 --target CPSC --head query --labels mask_pseudo --hierarchy --mixstyle --seed 0 --tag loso-CPSC-abl-query-s0",
"--prep {PREP} --mode loso --epochs 30 --target CPSC --head text --text_emb {EMB} --labels mask_pseudo --mixstyle --seed 0 --tag loso-CPSC-abl-noh-s0",
"--prep {PREP} --mode loso --epochs 30 --target CPSC --head text --text_emb {EMB} --labels mask_pseudo --hierarchy --seed 0 --tag loso-CPSC-abl-nomix-s0",
"--prep {PREP} --mode loso --epochs 30 --target CPSC --head text --text_emb {EMB_NAME} --labels mask_pseudo --hierarchy --mixstyle --seed 0 --tag loso-CPSC-abl-name-s0",
"--prep {PREP} --mode loso --epochs 30 --target CPSC --head text --text_emb {EMB_MPNET} --labels mask_pseudo --hierarchy --mixstyle --seed 0 --tag loso-CPSC-abl-mpnet-s0",
"--prep {PREP} --mode loso --epochs 30 --target PTBXL --head text --text_emb {EMB} --labels mask --hierarchy --mixstyle --seed 0 --tag loso-PTBXL-abl-textmask-s0",
"--prep {PREP} --mode loso --epochs 30 --target PTBXL --head linear --labels mask_pseudo --hierarchy --mixstyle --seed 0 --tag loso-PTBXL-abl-linpseudo-s0",
"--prep {PREP} --mode loso --epochs 30 --target PTBXL --head query --labels mask_pseudo --hierarchy --mixstyle --seed 0 --tag loso-PTBXL-abl-query-s0",
"--prep {PREP} --mode loso --epochs 30 --target PTBXL --head text --text_emb {EMB} --labels mask_pseudo --mixstyle --seed 0 --tag loso-PTBXL-abl-noh-s0",
"--prep {PREP} --mode loso --epochs 30 --target PTBXL --head text --text_emb {EMB} --labels mask_pseudo --hierarchy --seed 0 --tag loso-PTBXL-abl-nomix-s0",
"--prep {PREP} --mode loso --epochs 30 --target PTBXL --head text --text_emb {EMB_NAME} --labels mask_pseudo --hierarchy --mixstyle --seed 0 --tag loso-PTBXL-abl-name-s0",
"--prep {PREP} --mode loso --epochs 30 --target PTBXL --head text --text_emb {EMB_MPNET} --labels mask_pseudo --hierarchy --mixstyle --seed 0 --tag loso-PTBXL-abl-mpnet-s0"
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
