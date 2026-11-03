"""Build self-contained Kaggle notebooks (code embedded with %%writefile) for manual import.

Usage: python kaggle/make_notebooks.py   -> notebooks/01_ecg_prep.ipynb, notebooks/02_ecg_pilot.ipynb
"""
import base64
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
OUT = os.path.join(ROOT, "notebooks")
sys.path.insert(0, os.path.join(ROOT, "kaggle"))
from experiments import BATCHES  # noqa: E402

CODE_DIR = "/kaggle/working/code"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text}


def writefile(name):
    with open(os.path.join(SRC, name), encoding="utf-8") as f:
        return code(f"%%writefile {CODE_DIR}/{name}\n" + f.read())


def notebook(cells):
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                         "language_info": {"name": "python"}}}


def save(nb, name):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("wrote", os.path.join(OUT, name))


def prep_nb():
    return notebook([
        md("# 01 - ECG preprocessing\n"
           "Inputs: `gamalasran/physionet-challenge-2020` and "
           "`bjoernjostein/shaoxing-and-ningbo-first-hospital-database`. Accelerator: **None (CPU)**.\n"
           "Output: `/kaggle/working/prep/` (signals.npy, meta.csv, code_counts.csv)."),
        code(f"import os; os.makedirs('{CODE_DIR}', exist_ok=True)"),
        writefile("preprocess.py"),
        code(f"!python {CODE_DIR}/preprocess.py --roots /kaggle/input --out /kaggle/working/prep"),
        code("import pandas as pd\n"
             "print(pd.read_csv('/kaggle/working/prep/meta.csv').source.value_counts())\n"
             "pd.read_csv('/kaggle/working/prep/code_counts.csv', index_col=0).head(60)"),
    ])


def run_nb(batch):
    with open(os.path.join(SRC, "text_emb_biolord_full.npz"), "rb") as f:
        emb_b64 = base64.b64encode(f.read()).decode()
    cmds = BATCHES[batch]
    runner = f'''import glob, json, os, sys, time
sys.path.insert(0, "{CODE_DIR}")
import train
PREP = os.path.dirname(glob.glob("/kaggle/input/**/prep/signals.npy", recursive=True)[0])
EMB = "{CODE_DIR}/text_emb_biolord_full.npz"
OUT = "/kaggle/working/results"
BUDGET_H = 11.0  # stop launching new runs before Kaggle's 12 h limit
CMDS = {json.dumps(cmds, indent=1)}
t0 = time.time()
for i, c in enumerate(CMDS):
    tag = c.split("--tag ")[1].split()[0]
    if os.path.exists(f"{{OUT}}/{{tag}}.json"):
        continue
    if (time.time() - t0) / 3600 > BUDGET_H:
        print("time budget reached; remaining runs skipped"); break
    print(f"=== {{i + 1}}/{{len(CMDS)}} {{tag}} ===", flush=True)
    try:
        train.main(c.replace("{{PREP}}", PREP).replace("{{EMB}}", EMB).split() + ["--out", OUT])
    except Exception as e:
        print("FAILED", tag, repr(e), flush=True)
'''
    summary = '''import glob, json, pandas as pd
rows = []
for f in sorted(glob.glob("/kaggle/working/results/*.json")):
    r = json.load(open(f))
    rows.append(dict(run=r["name"], auroc=r["test_all"].get("auroc"), auprc=r["test_all"].get("auprc"),
                     f1=r["test_all"].get("f1"), auroc_seen=r["test_seen"].get("auroc"),
                     auroc_unseen=r["test_unseen"].get("auroc"), minutes=r["minutes"]))
df = pd.DataFrame(rows); df.to_csv("/kaggle/working/results/summary.csv", index=False)
pd.set_option("display.width", 200); df.round(4)
'''
    return notebook([
        md(f"# 02 - ECG experiments: batch `{batch}` ({len(cmds)} runs)\n"
           "Input: your notebook **ecg-prep** (its output). Accelerator: **GPU T4** (or P100).\n"
           "Run with **Save Version -> Save & Run All (Commit)** so it keeps running in the background."),
        code(f"import os; os.makedirs('{CODE_DIR}', exist_ok=True)\n!nvidia-smi"),
        writefile("labels.py"), writefile("data.py"), writefile("models.py"), writefile("train.py"),
        code(f"import base64\nopen('{CODE_DIR}/text_emb_biolord_full.npz', 'wb').write(base64.b64decode(\n"
             f"'{emb_b64}'))"),
        code(runner),
        code(summary),
    ])


if __name__ == "__main__":
    save(prep_nb(), "01_ecg_prep.ipynb")
    for b in BATCHES:
        save(run_nb(b), f"02_ecg_{b}.ipynb")
