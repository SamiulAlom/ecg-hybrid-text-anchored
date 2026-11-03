import glob, os, sys, subprocess
CODE = os.path.dirname(glob.glob("/kaggle/input/**/train.py", recursive=True)[0])
sys.path.insert(0, CODE)
print("code at", CODE)
PREP = glob.glob('/kaggle/input/**/prep/signals.npy', recursive=True)[0].rsplit('/', 1)[0]
import time, train
CMDS = [
"--prep {PREP} --mode loso --epochs 30 --target CPSC --head text --text_emb {EMB} --labels mask_pseudo --hierarchy --mixstyle --seed 1 --tag loso-CPSC-ours-s1",
"--prep {PREP} --mode loso --epochs 30 --target CPSC --head linear --labels mask --hierarchy --seed 1 --tag loso-CPSC-listyle-s1",
"--prep {PREP} --mode loso --epochs 30 --target CPSC --head linear --labels mask --hierarchy --mixstyle --seed 1 --tag loso-CPSC-mixstyle-s1",
"--prep {PREP} --mode loso --epochs 30 --target PTBXL --head text --text_emb {EMB} --labels mask_pseudo --hierarchy --mixstyle --seed 1 --tag loso-PTBXL-ours-s1",
"--prep {PREP} --mode loso --epochs 30 --target PTBXL --head linear --labels mask --hierarchy --seed 1 --tag loso-PTBXL-listyle-s1",
"--prep {PREP} --mode loso --epochs 30 --target PTBXL --head linear --labels mask --hierarchy --mixstyle --seed 1 --tag loso-PTBXL-mixstyle-s1",
"--prep {PREP} --mode loso --epochs 30 --target GEORGIA --head text --text_emb {EMB} --labels mask_pseudo --hierarchy --mixstyle --seed 1 --tag loso-GEORGIA-ours-s1",
"--prep {PREP} --mode loso --epochs 30 --target GEORGIA --head linear --labels mask --hierarchy --seed 1 --tag loso-GEORGIA-listyle-s1",
"--prep {PREP} --mode loso --epochs 30 --target GEORGIA --head linear --labels mask --hierarchy --mixstyle --seed 1 --tag loso-GEORGIA-mixstyle-s1",
"--prep {PREP} --mode loso --epochs 30 --target SHAONING --head text --text_emb {EMB} --labels mask_pseudo --hierarchy --mixstyle --seed 1 --tag loso-SHAONING-ours-s1",
"--prep {PREP} --mode loso --epochs 30 --target SHAONING --head linear --labels mask --hierarchy --seed 1 --tag loso-SHAONING-listyle-s1",
"--prep {PREP} --mode loso --epochs 30 --target SHAONING --head linear --labels mask --hierarchy --mixstyle --seed 1 --tag loso-SHAONING-mixstyle-s1",
"--prep {PREP} --mode loso --epochs 30 --target CPSC --head text --text_emb {EMB} --labels mask_pseudo --hierarchy --mixstyle --seed 2 --tag loso-CPSC-ours-s2",
"--prep {PREP} --mode loso --epochs 30 --target CPSC --head linear --labels mask --hierarchy --seed 2 --tag loso-CPSC-listyle-s2",
"--prep {PREP} --mode loso --epochs 30 --target CPSC --head linear --labels mask --hierarchy --mixstyle --seed 2 --tag loso-CPSC-mixstyle-s2",
"--prep {PREP} --mode loso --epochs 30 --target PTBXL --head text --text_emb {EMB} --labels mask_pseudo --hierarchy --mixstyle --seed 2 --tag loso-PTBXL-ours-s2",
"--prep {PREP} --mode loso --epochs 30 --target PTBXL --head linear --labels mask --hierarchy --seed 2 --tag loso-PTBXL-listyle-s2",
"--prep {PREP} --mode loso --epochs 30 --target PTBXL --head linear --labels mask --hierarchy --mixstyle --seed 2 --tag loso-PTBXL-mixstyle-s2",
"--prep {PREP} --mode loso --epochs 30 --target GEORGIA --head text --text_emb {EMB} --labels mask_pseudo --hierarchy --mixstyle --seed 2 --tag loso-GEORGIA-ours-s2",
"--prep {PREP} --mode loso --epochs 30 --target GEORGIA --head linear --labels mask --hierarchy --seed 2 --tag loso-GEORGIA-listyle-s2",
"--prep {PREP} --mode loso --epochs 30 --target GEORGIA --head linear --labels mask --hierarchy --mixstyle --seed 2 --tag loso-GEORGIA-mixstyle-s2",
"--prep {PREP} --mode loso --epochs 30 --target SHAONING --head text --text_emb {EMB} --labels mask_pseudo --hierarchy --mixstyle --seed 2 --tag loso-SHAONING-ours-s2",
"--prep {PREP} --mode loso --epochs 30 --target SHAONING --head linear --labels mask --hierarchy --seed 2 --tag loso-SHAONING-listyle-s2",
"--prep {PREP} --mode loso --epochs 30 --target SHAONING --head linear --labels mask --hierarchy --mixstyle --seed 2 --tag loso-SHAONING-mixstyle-s2"
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
