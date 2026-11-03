"""MERL zero-shot baseline (Liu et al., ICML 2024, arXiv 2403.06659) on our leave-one-source-out (LOSO) test sets.

Inference only: no training, no fine-tuning, and no use of target labels except to score the results.
Requirements: a Kaggle GPU with internet ON. The script downloads the official MERL code (GitHub, MIT license),
the released checkpoint (public Google Drive folder), and the MedCPT text encoder (HuggingFace).

Outputs (in --out, same format as src/train.py):
  merl-<TARGET>.json          name, vocab, test_all/test_seen/test_unseen (train.py evaluate()), per_label, notes
  merl-<TARGET>_test.npz      P (float16), Y, A (uint8, label_matrices without hierarchy), idx (test rows)
  merl-<TARGET>_cos.npz       raw window-averaged cosine scores S (float32) for test and val rows
Prompt variants get their own tag: merl-* (MERL's CKEPE prompts), merl_desc-* (our text_of(code,'full')),
merl_name-* (bare lowercase class name). Using the ViT-tiny checkpoint adds "_vit" to the tag (merl_vit-*).

Example (Kaggle):
  python merl_zeroshot.py --prep /kaggle/input/ecg-prep/prep --code /kaggle/input/ecg-code/src \
      --out /kaggle/working/results
Local CPU smoke test without weights (fake encoders, checks data/prompt/eval/output plumbing only):
  python merl_zeroshot.py --prep data/prep_synth --code src --merl_repo <copy of MERL repo> --dry_run --out /tmp/x
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import types
from math import gcd

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly

MERL_REPO = "https://github.com/cheliu-computation/MERL-ICML2024"
MERL_COMMIT = "2a38649285e16eff75b69aeb64f2366b380c1a9e"  # main @ 2025-09-18 (latest when this script was written)
MEDCPT = "ncbi/MedCPT-Query-Encoder"
# Official release: https://drive.google.com/drive/folders/13wb4DppUciMn-Y_qC2JRWTbZdz3xX0w2 (public, no login)
CKPTS = {
    "resnet18": dict(file="res18_best_ckpt.pth", id="14LNYvwtbbUwmd_1rxVYf31X4p2HInjSe", size=457260616),
    "vit_tiny": dict(file="vit_tiny_best_ckpt.pth", id="1dmnx510T_wcJthXhC4QnhjQyzvm_UUoR", size=461895413),
}
MERL_FS, MERL_LEN = 500, 5000  # MIMIC-IV-ECG: 500 Hz, 10 s
OUR_FS, WIN = 100, 1000        # our prep: 100 Hz; 10 s windows as in train.py
STRIDE = 500                   # 5 s stride, as train.py predict()

# Our SNOMED code -> key in MERL's zeroshot/CKEPE_prompt.json (keys are PTB-XL SCP / CPSC / Chapman-CSN class
# names). None = no matching CKEPE entry; MERL's prompt is then the bare class name.
CKEPE_KEY = {
    "164889003": "AFIB",     # atrial fibrillation
    "164890007": "AFLT",     # atrial flutter
    "6374002": None,         # bundle branch block (no generic BBB entry)
    "426627000": None,       # bradycardia (only sinus bradycardia exists)
    "164909002": "LBBB",     # left bundle branch block (CSN key; CLBBB is the PTB-XL equivalent)
    "59118001": "RBBB",      # right bundle branch block (CSN key; CRBBB is the PTB-XL equivalent)
    "270492004": "1AVB",
    "713426002": "IRBBB",
    "39732003": "ALS",       # axis left shift = left axis deviation
    "445118002": "LAFB",
    "251146004": "LVOLT",
    "698252002": "IVCD",
    "426783006": "SR",
    "63593006": "PAC",       # "... supraventricular premature complex"
    "10370003": "PACE",
    "365413008": None,       # poor R wave progression
    "17338001": "VPB",
    "164947007": "LPR",
    "111975006": "LNGQT",
    "164917005": "AQW",      # abnormal Q wave
    "47665007": "ARS",       # axis right shift = right axis deviation
    "427393009": "SARRH",
    "426177001": "SB",
    "427084000": "STACH",
    "164934002": "TAB",
    "59931005": "INVT",
    "164865005": "MI",
    "164873001": "LVH",
    "429622005": "STD",
    "164931005": "STE",
    "55827005": "VCLVH",     # voltage criteria for LVH
    "195042002": "2AVB",
    "233917008": "AVB",
    "428750005": "NST",      # non-specific ST changes
    "67741000119109": "LAO/LAE",
    "446358003": "RAH",
    "89792004": "RVH",
    "713422000": "AT",
    "426761007": "SVT",
    "251198002": "CR",       # clockwise rotation
    "251199005": "CCR",      # counter-clockwise rotation
    "164912004": "PWC",
    "164942001": "FQRS",
    "164930006": None,       # ST interval abnormal
    "164937009": "UW",
}

NOTES = {
    "method": "MERL zero-shot (Liu et al., ICML 2024, arXiv 2403.06659), official code and released checkpoint, "
              "no training or fine-tuning on any of our data.",
    "pretraining_data": "MIMIC-IV-ECG v1.0 (BIDMC, Boston; 771,693 ECG-report pairs, 500 Hz, 10 s). It shares no "
                        "records with our databases (CPSC2018/CPSC-Extra, PTB-XL, Georgia, Shaoxing/Ningbo), "
                        "which come from other hospitals and countries.",
    "checkpoint_selection_caveat": "The released *_best_ckpt.pth was selected during pretraining by the average "
                                   "zero-shot AUROC on the validation splits of PTB-XL (super/sub/form/rhythm), "
                                   "CPSC2018 (ICBEB) and Chapman-Shaoxing (CSN) (see utils/utils_trainer.py and "
                                   "pretrain/config.yaml val_sets). Those are part of our CPSC, PTBXL and SHAONING "
                                   "test sets. MERL never trained on them, but it was model-selected on them, and "
                                   "its CKEPE prompts were written for their class names. GEORGIA was never used.",
    "paper_eval": "The paper evaluates zero-shot on PTB-XL (4 subsets), CPSC2018 and CSN test splits, with dataset-"
                  "specific label sets. Our evaluation uses our SNOMED vocabulary and protocol, so the numbers are "
                  "not comparable to the paper's.",
}


# ---------------------------------------------------------------- downloads
def sh(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def get_repo(work, commit):
    dst = os.path.join(work, "MERL-ICML2024")
    if os.path.exists(os.path.join(dst, "utils", "utils_builder.py")):
        return dst
    try:
        sh(["git", "clone", "-q", MERL_REPO, dst])
        sh(["git", "-C", dst, "checkout", "-q", commit])
    except Exception as e:  # no git: use the GitHub archive of the pinned commit
        print("git failed, using the codeload zip:", e)
        import io, zipfile, requests
        r = requests.get(f"https://codeload.github.com/cheliu-computation/MERL-ICML2024/zip/{commit}", timeout=120)
        r.raise_for_status()
        zipfile.ZipFile(io.BytesIO(r.content)).extractall(work)
        os.replace(os.path.join(work, f"MERL-ICML2024-{commit}"), dst)
    return dst


def _looks_like_ckpt(path, size):
    if not os.path.exists(path):
        return False
    with open(path, "rb") as f:
        magic = f.read(2)
    return magic == b"PK" and (size is None or os.path.getsize(path) == size)  # torch zip format


def get_ckpt(work, backbone, local=None):
    if local:
        assert os.path.exists(local), local
        return local
    info = CKPTS[backbone]
    dst = os.path.join(work, info["file"])
    if _looks_like_ckpt(dst, info["size"]):
        return dst
    import requests
    url = f"https://drive.usercontent.google.com/download?id={info['id']}&export=download&confirm=t"
    print(f"downloading {info['file']} ({info['size'] / 1e6:.0f} MB) from Google Drive", flush=True)
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(dst + ".part", "wb") as f:
                for chunk in r.iter_content(1 << 22):
                    f.write(chunk)
        os.replace(dst + ".part", dst)
    except Exception as e:
        print("direct download failed:", e)
    if not _looks_like_ckpt(dst, info["size"]):  # Drive quota page / HTML: try gdown
        print("direct download did not return the checkpoint; trying gdown", flush=True)
        sh([sys.executable, "-m", "pip", "install", "-q", "gdown"])
        import gdown
        gdown.download(id=info["id"], output=dst, quiet=False)
    if not _looks_like_ckpt(dst, info["size"]):
        raise RuntimeError(
            f"Could not download {info['file']}. Download it manually from "
            f"https://drive.google.com/drive/folders/13wb4DppUciMn-Y_qC2JRWTbZdz3xX0w2, attach it as a Kaggle "
            f"dataset and pass --ckpt /kaggle/input/<dataset>/{info['file']}")
    return dst


# ---------------------------------------------------------------- encoders
class MERLEncoder:
    """Calls the official ECGCLIP methods exactly as zeroshot/test_zeroshot.py and utils/zeroshot_val.py do."""

    def __init__(self, repo, ckpt, backbone, device):
        sys.path.insert(0, os.path.join(repo, "utils"))
        try:  # utils_builder.py starts with `from cgi import test` (unused); cgi was removed in Python 3.13
            import cgi  # noqa: F401
        except ImportError:
            sys.modules["cgi"] = types.SimpleNamespace(test=None)
        try:  # imported but unused by utils_builder.py
            import torchvision  # noqa: F401
        except ImportError:
            sys.modules["torchvision"] = types.ModuleType("torchvision")
        for mod in ("einops",):  # vit1d.py needs einops (imported by utils_builder even for ResNet)
            try:
                __import__(mod)
            except ImportError:
                sh([sys.executable, "-m", "pip", "install", "-q", mod])
        import utils_builder
        cfg = {"ecg_model": backbone, "num_leads": 12, "text_model": MEDCPT, "free_layers": 6,
               "feature_dim": 768, "projection_head": {"mlp_hidden_size": 256, "projection_size": 256}}
        model = utils_builder.ECGCLIP(cfg)
        try:
            sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        except Exception as e:
            print("weights_only load failed, retrying with weights_only=False (official checkpoint):", e)
            sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        for k in ("model_state_dict", "state_dict", "model"):
            if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
                sd = sd[k]
        sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        bad = [k for k in list(missing) + list(unexpected) if not k.endswith("position_ids")]
        if bad:
            raise RuntimeError(f"checkpoint does not match ECGCLIP({backbone}): {bad[:20]}")
        print(f"loaded {ckpt}: {len(sd)} tensors (ignored: {list(missing) + list(unexpected)})", flush=True)
        self.model = model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def text(self, prompts):
        """get_class_emd(): lowercase, tokenize one prompt at a time (max_length 256), pooler_output -> proj_t -> L2."""
        out = []
        for t in prompts:
            # same arguments as ECGCLIP._tokenize(); called directly because batch_encode_plus was removed in
            # transformers 5
            tok = self.model.tokenizer([t.lower()], add_special_tokens=True, truncation=True, max_length=256,
                                       padding="max_length", return_tensors="pt")
            e = self.model.get_text_emb(tok.input_ids.to(self.device), tok.attention_mask.to(self.device))
            e = self.model.proj_t(e)
            e = e / e.norm(dim=-1, keepdim=True)
            e = e.mean(0)
            out.append(e / e.norm())
        return torch.stack(out)  # [C, 256]

    @torch.no_grad()
    def ecg(self, x):
        e = self.model.ext_ecg_emb(x.to(self.device))
        return e / e.norm(dim=-1, keepdim=True)  # [B, 256]


class FakeEncoder:
    """--dry_run: deterministic random encoders with MERL's shapes, for CPU plumbing tests only."""

    def __init__(self, device):
        g = torch.Generator().manual_seed(0)
        self.W = torch.randn(256, 12 * 50, generator=g)
        self.device = device

    def text(self, prompts):
        rows = []
        for t in prompts:
            seed = int(hashlib.md5(t.lower().encode()).hexdigest()[:8], 16)
            rows.append(torch.randn(256, generator=torch.Generator().manual_seed(seed)))
        return F.normalize(torch.stack(rows), dim=-1)

    def ecg(self, x):
        assert x.shape[1:] == (12, MERL_LEN), x.shape
        z = F.adaptive_avg_pool1d(x, 50).flatten(1) @ self.W.T
        return F.normalize(z, dim=-1)


# ---------------------------------------------------------------- signals
def to_merl_input(xw, lead_swap=True):
    """xw: [B,12,1000] mV at 100 Hz (our lead order) -> [B,12,5000] in MERL's input space.
    1. Polyphase upsampling 100 -> 500 Hz, line-padded edges (MERL is pretrained on 500 Hz, 10 s = 5000 samples).
    2. Lead order: our I,II,III,aVR,aVL,aVF,V1-V6 -> MIMIC-IV-ECG I,II,III,aVR,aVF,aVL,V1-V6 (MERL's downstream
       loader swaps rows 4 and 5 the same way, finetune/finetune_dataset.py).
    3. Global min-max to [0,1] over all 12 leads and samples of the window (MERL pretrain/finetune preprocessing)."""
    g = gcd(MERL_FS, OUR_FS)
    x = resample_poly(np.nan_to_num(xw.astype(np.float32)), MERL_FS // g, OUR_FS // g, axis=2,
                      padtype="line").astype(np.float32)  # "line" padding avoids edge ringing at window borders
    x = x[:, :, :MERL_LEN]
    if lead_swap:
        x = x[:, [0, 1, 2, 3, 5, 4, 6, 7, 8, 9, 10, 11]]
    mn = x.min(axis=(1, 2), keepdims=True)
    mx = x.max(axis=(1, 2), keepdims=True)
    return (x - mn) / (mx - mn + 1e-8)


def window_starts(n_valid, L, mode):
    """Same windows as train.py predict(): 10 s, stride 5 s, the first always used, others if inside the valid part."""
    if mode == "first":
        return [0]
    return [k for k in range(0, L - WIN + 1, STRIDE) if k == 0 or k + WIN <= max(n_valid, WIN)]


def embed_records(enc, X, nvalid, rows, windows, lead_swap, bs, max_windows=None):
    """Returns [len(rows), 256]: mean over windows of L2-normalised ECG embeddings. Because the score is a dot
    product with a fixed text embedding, this equals averaging the per-window cosine similarities."""
    out = np.zeros((len(rows), 256), np.float32)
    L = X.shape[2]
    t0 = time.time()
    for i in range(0, len(rows), bs):
        b = rows[i:i + bs]
        xb = np.asarray(X[b], dtype=np.float32)
        acc = np.zeros((len(b), 256), np.float32)
        cnt = np.zeros((len(b), 1), np.float32)
        ks = sorted({k for r in b for k in window_starts(int(nvalid[r]), L, windows)})
        for k in ks:
            use = np.array([k in window_starts(int(nvalid[r]), L, windows) for r in b])
            x = torch.from_numpy(to_merl_input(xb[use, :, k:k + WIN], lead_swap))
            acc[use] += enc.ecg(x).float().cpu().numpy()
            cnt[use] += 1
        out[i:i + len(b)] = acc / cnt
        if (i // bs) % 20 == 0:
            print(f"  embedded {i + len(b)}/{len(rows)} records ({time.time() - t0:.0f}s)", flush=True)
    return out


# ---------------------------------------------------------------- prompts
def build_prompts(vocab, LABELS, text_of, ckepe, style):
    prompts, src = [], []
    for c in vocab:
        name = LABELS[c][1]
        if style == "ckepe":
            k = CKEPE_KEY.get(c)
            if k is not None and k in ckepe:
                prompts.append(ckepe[k]); src.append(f"CKEPE:{k}")
            else:
                if k is not None:
                    print(f"WARNING: CKEPE key {k!r} for {c} not in CKEPE_prompt.json; using class name")
                prompts.append(name); src.append("name (no CKEPE entry)")
        elif style == "desc":
            prompts.append(text_of(c, "full")); src.append("text_of(code,'full')")
        elif style == "name":
            prompts.append(name); src.append("name")
        else:
            raise ValueError(style)
    return prompts, src


def to_prob(S_val, S_test, calib):
    """Cosine -> [0,1]. Monotone per label, so AUROC/AUPRC equal those of the raw cosine. The statistics come from
    the source-domain validation scores (no target data, no labels).
    perclass: sigmoid((s - mu_c) / sd_c); global: MERL's non-softmax formula sigmoid((s - mu) / sd) with one mu/sd;
    none: (s + 1) / 2."""
    if calib == "none":
        return (S_val + 1) / 2, (S_test + 1) / 2
    ref = S_val if len(S_val) > 1 else S_test
    ax = 0 if calib == "perclass" else None
    mu, sd = ref.mean(ax), ref.std(ax) + 1e-8
    sig = lambda s: 1 / (1 + np.exp(-(s - mu) / sd))
    return sig(S_val), sig(S_test)


# ---------------------------------------------------------------- main
def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", default=None, help="folder with signals.npy + meta.csv (src/preprocess.py output); "
                    "default: first /kaggle/input/**/prep/signals.npy")
    ap.add_argument("--code", default=None, help="folder containing our data.py / labels.py / train.py; "
                    "default: first /kaggle/input/**/train.py")
    ap.add_argument("--out", default="/kaggle/working/results")
    ap.add_argument("--work", default="/kaggle/working/merl_assets", help="where the repo/checkpoint/cache go")
    ap.add_argument("--targets", nargs="+", default=["CPSC", "PTBXL", "GEORGIA", "SHAONING"])
    ap.add_argument("--backbone", choices=list(CKPTS), default="resnet18", help="paper's main model is ResNet18")
    ap.add_argument("--ckpt", default=None, help="local *_best_ckpt.pth (skips the Google Drive download)")
    ap.add_argument("--merl_repo", default=None, help="existing checkout of MERL-ICML2024 (skips git clone)")
    ap.add_argument("--merl_commit", default=MERL_COMMIT)
    ap.add_argument("--prompts", nargs="+", default=["ckepe", "desc", "name"], choices=["ckepe", "desc", "name"])
    ap.add_argument("--windows", choices=["all", "first"], default="all",
                    help="all: average over 10 s windows (stride 5 s), as train.py; first: first 10 s only")
    ap.add_argument("--calib", choices=["perclass", "global", "none"], default="perclass")
    ap.add_argument("--no_lead_swap", action="store_true", help="debug only: do not reorder aVL/aVF")
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--min_source", type=int, default=10)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--max_records", type=int, default=0, help="debug: subsample test/val rows per target")
    ap.add_argument("--dry_run", action="store_true", help="fake encoders (CPU plumbing test, numbers meaningless)")
    return ap.parse_args(argv)


def main(argv=None):
    a = get_args(argv)
    import glob
    if a.prep is None:
        a.prep = os.path.dirname(glob.glob("/kaggle/input/**/prep/signals.npy", recursive=True)[0])
    if a.code is None:
        a.code = os.path.dirname(glob.glob("/kaggle/input/**/train.py", recursive=True)[0])
    print(f"prep={a.prep} code={a.code}", flush=True)
    sys.path.insert(0, os.path.abspath(a.code))
    from data import build_vocab, label_matrices, load_prep, split_indices
    from labels import LABELS, text_of
    from train import evaluate, tune_thresholds

    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.work, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_start = time.time()

    repo = a.merl_repo or get_repo(a.work, a.merl_commit)
    with open(os.path.join(repo, "zeroshot", "CKEPE_prompt.json"), encoding="utf-8") as f:
        ckepe = json.load(f)
    if a.dry_run:
        enc, ckpt = FakeEncoder(dev), None
    else:
        ckpt = get_ckpt(a.work, a.backbone, a.ckpt)
        enc = MERLEncoder(repo, ckpt, a.backbone, dev)

    X, meta = load_prep(a.prep)
    nvalid = meta.n100.values.astype(np.int64)
    vocab, annotated, _ = build_vocab(meta, min_source=a.min_source)
    Y0, A0 = label_matrices(meta, vocab, annotated)  # original labels, no hierarchy completion (as train.py eval)
    C = len(vocab)
    print(f"device={dev} records={len(meta)} vocab={C} signals={X.shape} windows={a.windows}", flush=True)

    # text embeddings, one set per prompt style
    text = {}
    for style in a.prompts:
        prompts, src = build_prompts(vocab, LABELS, text_of, ckepe, style)
        text[style] = (enc.text(prompts).float().cpu().numpy(), prompts, src)

    # splits; then embed every needed record once (cached across targets and prompt styles)
    splits = {}
    for t in a.targets:
        tr, va, te = split_indices(meta, "loso", t, seed=a.split_seed)
        if len(te) == 0:
            print(f"skip {t}: no records")
            continue
        if a.max_records:
            rng = np.random.default_rng(0)
            te = np.sort(rng.permutation(te)[:a.max_records]); va = np.sort(rng.permutation(va)[:a.max_records])
        splits[t] = (np.sort(tr), np.sort(va), np.sort(te))
    need = np.unique(np.concatenate([np.concatenate([v, e]) for _, v, e in splits.values()]))
    cache = os.path.join(a.work, f"emb_{'fake' if a.dry_run else a.backbone}_{a.windows}"
                                 f"{'_noswap' if a.no_lead_swap else ''}_{len(meta)}.npz")
    E = np.full((len(meta), 256), np.nan, np.float32)
    if os.path.exists(cache) and not a.max_records:
        z = np.load(cache); E[z["rows"]] = z["E"]
    todo = need[np.isnan(E[need, 0])]
    if len(todo):
        print(f"embedding {len(todo)} records", flush=True)
        E[todo] = embed_records(enc, X, nvalid, todo, a.windows, not a.no_lead_swap, a.bs)
        done = np.where(~np.isnan(E[:, 0]))[0]
        if not a.max_records:
            np.savez(cache, rows=done, E=E[done])

    for t, (tr, va, te) in splits.items():
        no_sup = set(np.where(A0[tr].sum(0) == 0)[0].tolist())  # as train.py: seen = annotated by a training source
        seen = [j for j in range(C) if j not in no_sup]
        unseen = [j for j in range(C) if j in no_sup]
        for style, (T, prompts, src) in text.items():
            S_val, S_te = E[va] @ T.T, E[te] @ T.T
            P_val, P_te = to_prob(S_val, S_te, a.calib)
            th = tune_thresholds(P_val, Y0[va], A0[va], list(range(C))) if len(va) else None
            m_val, _ = evaluate(P_val, Y0[va], A0[va], list(range(C))) if len(va) else ({}, {})
            m_all, per = evaluate(P_te, Y0[te], A0[te], list(range(C)), th)
            m_seen, _ = evaluate(P_te, Y0[te], A0[te], seen, th)
            m_unseen, _ = evaluate(P_te, Y0[te], A0[te], unseen, th)
            name = f"merl{'' if a.backbone == 'resnet18' else '_vit'}{'' if style == 'ckepe' else '_' + style}-{t}"
            if a.dry_run:
                name = "DRYRUN_" + name
            res = dict(name=name, config=dict(vars(a), mode="loso", target=t, method="merl_zeroshot", prompt_style=style,
                                              merl_repo=MERL_REPO, merl_commit=a.merl_commit, text_model=MEDCPT,
                                              checkpoint=None if a.dry_run else os.path.basename(ckpt)),
                       vocab=vocab, best_val_auroc=m_val.get("auroc"), history=[],
                       test_all=m_all, test_seen=m_seen, test_unseen=m_unseen,
                       seen=[vocab[j] for j in seen], unseen=[vocab[j] for j in unseen], dropped=[],
                       per_label={LABELS[vocab[j]][0]: v for j, v in per.items()},
                       prompts={c: dict(abbr=LABELS[c][0], source=s, text=p) for c, p, s in zip(vocab, prompts, src)},
                       input_spec=dict(fs=MERL_FS, samples=MERL_LEN, leads="I,II,III,aVR,aVF,aVL,V1-V6",
                                       norm="global min-max [0,1] per 10 s window",
                                       source="our 100 Hz 0.5-45 Hz band-passed mV, upsampled x5 (resample_poly)"),
                       val=dict(n=int(len(va)), metrics=m_val), notes=NOTES,
                       minutes=(time.time() - t_start) / 60)
            with open(os.path.join(a.out, name + ".json"), "w") as f:
                json.dump(res, f, indent=1)
            np.savez_compressed(os.path.join(a.out, name + "_test.npz"), P=P_te.astype(np.float16),
                                Y=Y0[te].astype(np.uint8), A=A0[te].astype(np.uint8), idx=te)
            np.savez_compressed(os.path.join(a.out, name + "_cos.npz"), S_test=S_te.astype(np.float32), idx=te,
                                S_val=S_val.astype(np.float32), idx_val=va)
            print(f"[{name}] test={len(te)} all={m_all} seen={m_seen} unseen={m_unseen}", flush=True)
    print(f"done in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
