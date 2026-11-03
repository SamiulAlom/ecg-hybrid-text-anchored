"""Colab runner: executes colab_cmds.txt in parallel and uploads results to samiulalom01/ecg-colab-results.
Called by the launcher cell after the code dataset is downloaded to /content/code."""
import subprocess, os
U = "samiulalom01"
os.makedirs("/content/prep", exist_ok=True); os.makedirs("/content/results", exist_ok=True)
import glob
if not glob.glob("/content/prep/**/signals.npy", recursive=True):
    print("downloading preprocessed signals (~4 GB)...", flush=True)
    subprocess.run(["kaggle", "kernels", "output", f"{U}/ecg-prep", "-p", "/content/prep", "-q"], check=True)
r = subprocess.run(["kaggle", "datasets", "download", f"{U}/ecg-colab-results", "-p", "/content/results", "--unzip", "-q"],
                   capture_output=True, text=True)
print("earlier results:", "none" if r.returncode else len(os.listdir("/content/results")), flush=True)
import glob, os, subprocess, time, json, threading
CODE, OUT = "/content/code", "/content/results"
PREP = os.path.dirname(glob.glob("/content/prep/**/signals.npy", recursive=True)[0])
N_PAR = 4  # parallel runs on one A100 (each holds ~4 GB of signals on the GPU); use 2 on L4/T4
SUB = {"{PREP}": PREP, "{EMB}": f"{CODE}/text_emb_biolord_full.npz",
       "{EMB_NAME}": f"{CODE}/text_emb_biolord_name.npz", "{EMB_MPNET}": f"{CODE}/text_emb_mpnet_full.npz"}
cmds = [l.strip() for l in open(f"{CODE}/colab_cmds.txt") if l.strip() and not l.startswith("#")]
todo = []
for c in cmds:
    for k, v in SUB.items():
        c = c.replace(k, v)
    tag = c.split("--tag ")[1].split()[0]
    if not os.path.exists(f"{OUT}/{tag}.json"):
        todo.append((tag, c))
print(f"{len(cmds)} runs listed, {len(todo)} to do")

meta = {"title": "ecg-colab-results", "id": "samiulalom01/ecg-colab-results", "licenses": [{"name": "CC0-1.0"}]}
json.dump(meta, open(f"{OUT}/dataset-metadata.json", "w"))
lock = threading.Lock()
def upload(msg):
    with lock:
        r = subprocess.run(["kaggle", "datasets", "version", "-p", OUT, "-m", msg, "-q"], capture_output=True, text=True)
        if "not found" in (r.stdout + r.stderr).lower() or r.returncode:
            r = subprocess.run(["kaggle", "datasets", "create", "-p", OUT, "-q"], capture_output=True, text=True)
        print("upload:", (r.stdout + r.stderr).strip()[-200:], flush=True)

running, t0 = {}, time.time()
while todo or running:
    while todo and len(running) < N_PAR:
        tag, c = todo.pop(0)
        log = open(f"{OUT}/{tag}.log", "w")
        running[tag] = subprocess.Popen(["python", f"{CODE}/train.py", *c.split(), "--out", OUT], cwd=CODE,
                                        stdout=log, stderr=subprocess.STDOUT)
        print(f"[{(time.time()-t0)/60:6.1f} min] start {tag}", flush=True)
    time.sleep(30)
    for tag, p in list(running.items()):
        if p.poll() is not None:
            del running[tag]
            ok = os.path.exists(f"{OUT}/{tag}.json")
            print(f"[{(time.time()-t0)/60:6.1f} min] {'done' if ok else 'FAILED'} {tag} ({len(todo)} queued)", flush=True)
            if not ok:
                print(open(f"{OUT}/{tag}.log").read()[-1500:])
            threading.Thread(target=upload, args=(f"{tag} {'done' if ok else 'failed'}",)).start()
print("ALL DONE in", round((time.time() - t0) / 60), "min")
import time; time.sleep(90)
subprocess.run(["kaggle", "datasets", "version", "-p", OUT, "-m", "final", "-q"])
