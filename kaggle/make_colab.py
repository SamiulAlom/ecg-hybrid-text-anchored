"""Build notebooks/colab_runner.ipynb: runs the experiment list on a Colab GPU and uploads results to Kaggle.

The run list is read from colab_cmds.txt inside the private code dataset (samiulalom01/ecg-paper-code), so the
notebook never needs editing: update the list by versioning the dataset, then re-run the notebook.
Results go to the private dataset samiulalom01/ecg-colab-results after every finished run.

Usage: python kaggle/make_colab.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "notebooks", "colab_runner.ipynb")


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t}


CELLS = [
    md("# ECG experiments on Colab\n"
       "1. **Runtime → Change runtime type → A100 GPU** (High-RAM if offered).\n"
       "2. Add your Kaggle credentials in the 🔑 **Secrets** panel (left sidebar) as `KAGGLE_USERNAME` and "
       "`KAGGLE_KEY`, and switch on notebook access for both. (Alternatively, cell 2 asks you to upload kaggle.json.)\n"
       "3. **Runtime → Run all.** Keep this tab open. Results upload to your private Kaggle dataset "
       "`ecg-colab-results` after every finished run, so nothing is lost if the session ends; re-running skips "
       "finished runs."),
    code("!nvidia-smi --query-gpu=name,memory.total --format=csv"),
    code('''import os, json
try:
    from google.colab import userdata
    os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
    os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
    print("Kaggle credentials loaded from Colab Secrets")
except Exception as e:
    print("Secrets not available (", e, ") - upload kaggle.json")
    from google.colab import files
    files.upload()
    os.makedirs("/root/.kaggle", exist_ok=True)
    os.replace("kaggle.json", "/root/.kaggle/kaggle.json"); os.chmod("/root/.kaggle/kaggle.json", 0o600)
!pip -q install --upgrade kaggle'''),
    code('''%%bash
set -e
U=samiulalom01
rm -rf /content/code && mkdir -p /content/code /content/prep /content/results
kaggle datasets download $U/ecg-paper-code -p /content/code --unzip -q
if [ ! -f /content/prep/signals.npy ]; then kaggle kernels output $U/ecg-prep -p /content/prep -q; fi
ls -la /content/code /content/prep | head -40
# previously uploaded results (so finished runs are skipped)
kaggle datasets download $U/ecg-colab-results -p /content/results --unzip -q 2>/dev/null || echo "no earlier results"
ls /content/results | wc -l'''),
    code('''import glob, os, subprocess, time, json, threading
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
print("ALL DONE in", round((time.time() - t0) / 60), "min")'''),
    code('''import time; time.sleep(60)  # let the last upload finish
!kaggle datasets version -p /content/results -m "final" -q || kaggle datasets create -p /content/results -q'''),
]

if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    nb = {"cells": CELLS, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                       "kernelspec": {"name": "python3", "display_name": "Python 3"}}}
    json.dump(nb, open(OUT, "w", encoding="utf-8"), indent=1)
    print("wrote", OUT)
