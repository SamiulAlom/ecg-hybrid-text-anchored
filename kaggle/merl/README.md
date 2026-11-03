# MERL zero-shot baseline

`merl_zeroshot.py` runs the released MERL model zero-shot on our four LOSO test sets (CPSC, PTBXL, GEORGIA, SHAONING). It only runs inference: nothing is trained or fine-tuned. It writes `merl*-<TARGET>.json` and `merl*-<TARGET>_test.npz` in the same format as `src/train.py`.

MERL is described in Liu, Wan, Ouyang, Shah, Bai, Arcucci, *Zero-Shot ECG Classification with Multimodal Learning and Test-time Clinical Knowledge Enhancement*, ICML 2024, arXiv [2403.06659](https://arxiv.org/abs/2403.06659).

## Sources

| What | Where | License / access |
|---|---|---|
| Code | https://github.com/cheliu-computation/MERL-ICML2024 (`.../MERL` redirects here), pinned to commit `2a38649285e16eff75b69aeb64f2366b380c1a9e` (main, 2025-09-18) | MIT (c) 2024 Che Liu |
| Checkpoints | Public Google Drive folder https://drive.google.com/drive/folders/13wb4DppUciMn-Y_qC2JRWTbZdz3xX0w2 | No separate license is stated; the repo is MIT. Pretraining data are ODbL 1.0. |
| `res18_best_ckpt.pth` (full CLIP model: ECG encoder, MedCPT, projections) | Drive file id `14LNYvwtbbUwmd_1rxVYf31X4p2HInjSe`, 457,260,616 B | **Default.** The paper's main model is ResNet18. |
| `vit_tiny_best_ckpt.pth` | id `1dmnx510T_wcJthXhC4QnhjQyzvm_UUoR`, 461,895,413 B | Used with `--backbone vit_tiny` (ablation model) |
| `res18_best_encoder.pth`, `vit_tiny_best_encoder.pth` | ids `1NG__hPjm8_5c9BiUmCzjlC2Og6ArCtww`, `16R8se-3kO09I5yE7E4COnBo84-vGPMsK` | ECG encoder only, for linear probing. Not used here. |
| Text encoder | HuggingFace `ncbi/MedCPT-Query-Encoder`. The architecture and tokenizer download from HF; the weights are overwritten by the checkpoint. | Public |
| Prompts | `zeroshot/CKEPE_prompt.json` in the repo (131 class keys) | MIT (repo) |

**Weight availability.** On 2026-11-02 all four Drive files were publicly downloadable without a login. A 1 KB range request returned `206` with the correct `Content-Disposition` and total size.

The script downloads the checkpoint from `drive.usercontent.google.com/download?id=…&confirm=t`. It then checks that the file has the torch zip magic and the expected byte size. If that fails, it falls back to `gdown`.

Google Drive can block popular files with a temporary download quota. If that happens, download `res18_best_ckpt.pth` by hand from the folder above, attach it to the notebook as a Kaggle dataset, and pass `--ckpt /kaggle/input/<dataset>/res18_best_ckpt.pth`.

## What MERL expects, and what we feed it

This section comes from the repo code: `pretrain/preprocess.ipynb`, `finetune/finetune_dataset.py`, `utils/utils_builder.py`, `utils/zeroshot_val.py`, `zeroshot/zeroshot_config.yaml`.

**MERL's input**
- 12 leads × 5000 samples: 500 Hz, 10 s, as in MIMIC-IV-ECG.
- The ResNet18 attention-pool positional embedding is fixed at 313 = ceil(5000/16) positions. The ViT-tiny uses `seq_len=5000` with patch size 50. Both therefore need exactly 5000 samples.
- Lead order is MIMIC-IV-ECG's: I, II, III, aVR, **aVF, aVL**, V1–V6. The downstream loader swaps rows 4 and 5 of standard-order data.
- Normalization: one global min-max over the whole 12×5000 record, mapping it to [0,1]. No filtering, no per-lead scaling, no unit conversion. During pretraining the [0,1] signals were stored as int16 ×1000 and divided by 1000 when loaded.

**Our pipeline**, from `signals.npy` (float16, mV, 100 Hz, band-passed 0.5–45 Hz, I,II,III,aVR,aVL,aVF,V1–V6):
1. Take 10 s windows (1000 samples) at the same positions as `train.py predict()`. That is stride 5 s; the first window is always used, later ones only if they lie inside the record's valid part (`n100`). `--windows first` uses only the first 10 s.
2. Upsample each window 100 → 500 Hz with `scipy.signal.resample_poly(up=5, padtype="line")`, giving 5000 samples.
3. Swap aVL and aVF to match MIMIC order.
4. Apply MERL's global min-max to [0,1] per window.
5. Run `ext_ecg_emb`, then L2-normalize. The per-window embeddings are averaged, which is the same as averaging the per-window cosine scores.

Records shorter than 10 s keep their zero padding inside the window. MERL handles CPSC2018 the same way: it keeps 2500 samples, zero-pads to 5000, then applies min-max.

**Text and scoring** (`get_class_emd` / `get_ecg_emd`)
- Each prompt is lowercased and tokenized with MedCPT (max_length 256, padded to max length).
- The text embedding is `pooler_output` → `proj_t` → L2 normalization.
- The score is the cosine similarity between ECG and prompt embeddings. MERL's evaluation code uses this raw cosine for AUROC (`softmax_eval=True` returns the cosine unchanged).

**Prompt variants.** Each variant is saved under its own tag:
- `merl-<T>`: MERL's own CKEPE prompts. Each of our SNOMED codes is mapped by hand to the closest CKEPE key; the table is `CKEPE_KEY` in the script, and each JSON records the source of every prompt. 41 of our 45 label definitions have a CKEPE entry. The other 4 (BBB, Brady, PRWP, STIAb) fall back to the bare class name, which is MERL's no-CKEPE baseline style.
- `merl_desc-<T>`: our `text_of(code, 'full')` descriptions, lowercased as MERL does.
- `merl_name-<T>`: bare lowercase class name.

**Probabilities.** MERL never defines calibrated probabilities. `train.py`-style F1 needs scores in [0,1], so the cosine is passed through a monotone per-label map: `--calib perclass` (default) is `sigmoid((s - mu_c)/sd_c)`.
- `mu_c` and `sd_c` come from the unlabeled source-domain validation records (`va` from `split_indices`). Target data are never used.
- Because the map is monotone, AUROC and AUPRC are identical to the raw cosine.
- `--calib global` uses one mean and std, which is MERL's `softmax_eval=False` formula. `--calib none` uses `(s+1)/2`.
- F1 thresholds are tuned on the validation sources with `train.py tune_thresholds`, as for our models. `f1_oracle` is computed on the test set, as in `train.py`.
- Raw cosines for test and val are saved in `merl*-<T>_cos.npz`, so any other calibration can be recomputed offline.

**Evaluation.** This is identical to `train.py`:
- `build_vocab(meta, min_source=10)`.
- `label_matrices(..., complete_hierarchy=False)`.
- `split_indices(meta, 'loso', T, seed=0)`.
- `evaluate()` on all labels, on seen labels (annotated, A=1, by at least one training source: `A0[tr].sum(0) > 0`), and on unseen labels.

## Data overlap and caveats

These are recorded in each JSON under `notes`.

- **Pretraining data.** MERL was pretrained only on MIMIC-IV-ECG v1.0. That is 771,693 ECG–report pairs from 161k BIDMC (Boston) patients, at 500 Hz for 10 s; reports with fewer than 4 words were dropped (code; the paper says fewer than three). It shares no records with CPSC2018/CPSC-Extra, PTB-XL, Georgia (Emory) or Shaoxing/Ningbo. The paper and the PhysioNet page confirm this. The paper mentions neither Georgia nor the PhysioNet 2020/2021 challenges.
- **Checkpoint selection used our test databases.** This comes from the code; the paper does not state it. `utils/utils_trainer.py` saves `*_bestZeroShotAll_ckpt.pth` at the epoch with the best average zero-shot AUROC on the `val_sets` in `pretrain/config.yaml`. Those are the validation splits of PTB-XL (super/sub/form/rhythm), ICBEB/CPSC2018 and Chapman/CSN. The released files are named `*_best_ckpt.pth`, so they are presumably these checkpoints.
  - MERL therefore never trained on PTBXL, CPSC or SHAONING, but its released checkpoint was chosen using labeled records from all three.
  - Its CKEPE prompts were also written for those datasets' class names; they were generated by GPT-4 from SCP-ECG and web knowledge.
  - GEORGIA is the only target with no MERL exposure at all.
- **The paper's own evaluation** covers PTB-XL (4 subsets), CPSC2018 and CSN test splits, each with its own label set. Its average zero-shot AUROC is 75.2% over 6 datasets. Our numbers use a different label space and protocol and are not comparable.
- **Signal deviations from MERL's data:**
  - Our signals are band-passed at 0.5–45 Hz and stored at 100 Hz, then upsampled back to 500 Hz. Content above 45 Hz, including pacemaker spikes, is missing, and baseline wander has been removed. MIMIC-IV-ECG inputs to MERL were unfiltered.
  - Min-max normalization is sensitive to spikes. We apply no clipping, matching MERL; `train.py` clips to ±5 mV.
  - Windows are averaged over up to 20 s, whereas MERL uses the first 10 s, or 5 s for CPSC2018.
- **Label mapping is approximate** for some codes:
  - The CSN keys `LBBB` and `RBBB` were used, and their CKEPE texts also mention fascicular and incomplete blocks.
  - `CR` (clockwise rotation) has a CKEPE text that also says "counterclockwise rotation".
  - NSSTTA maps to `NST`, and SVPB maps to `PAC`.
  - Check `prompts` in the JSON.
- **Checkpoint loading.** The script loads the checkpoint with `strict=False`. It accepts only `position_ids` buffer mismatches, which differ across `transformers` versions; any other mismatch raises an error.
- **Library compatibility.** Tokenization calls the tokenizer directly, because `batch_encode_plus`, which `ECGCLIP._tokenize` uses, no longer exists in transformers 5. Missing `cgi` (Python ≥ 3.13) and `torchvision` imports are stubbed, since both are imported but unused. `einops` is pip-installed if absent.

## Running on Kaggle

Notebook settings:
- **Accelerator:** GPU (T4 or P100 is enough). **Internet: ON.** Internet is needed for git clone, the Google Drive download (~460 MB) and MedCPT from HuggingFace (~440 MB).
- **Inputs:**
  1. The notebook that produced our `prep/` folder (`signals.npy`, `meta.csv`).
  2. The code dataset that contains `src/` (`data.py`, `labels.py`, `train.py`, `models.py`). Upload `kaggle/merl/merl_zeroshot.py` with it, or paste it into the notebook.
  3. Optional: a dataset with `res18_best_ckpt.pth`, if the Drive download is blocked.

```bash
# paths are auto-discovered if omitted: first /kaggle/input/**/prep/signals.npy and /kaggle/input/**/train.py
python merl_zeroshot.py --out /kaggle/working/results            # ResNet18, prompts ckepe+desc+name, all 4 targets
python merl_zeroshot.py --out /kaggle/working/results --backbone vit_tiny   # optional ablation (tags merl_vit*)
# explicit paths / manual checkpoint:
python merl_zeroshot.py --prep /kaggle/input/<prep-notebook>/prep --code /kaggle/input/<code-dataset> \
    --ckpt /kaggle/input/<ckpt-dataset>/res18_best_ckpt.pth --out /kaggle/working/results
```

Each record is embedded once and cached in `/kaggle/working/merl_assets/emb_<backbone>_<windows>_<N>.npz`. The cache is reused across targets and prompt variants. Expected runtime is minutes on a T4.

Outputs per target T:
- `merl-T.json` / `merl-T_test.npz`: CKEPE prompts, the primary baseline.
- `merl_desc-T.*`: our descriptions.
- `merl_name-T.*`: class names.
- `*_cos.npz`: raw cosine scores.

Useful flags:
- `--windows first` reproduces MERL's single-10 s protocol.
- `--calib global|none`.
- `--prompts ckepe`.
- `--max_records 500` for a quick check.
- `--merl_repo <path>` to use an existing checkout.

## Local smoke test (CPU, no weights)

```bash
.venv/Scripts/python kaggle/merl/merl_zeroshot.py --prep data/prep_synth --code src \
    --merl_repo <local copy of MERL-ICML2024> --dry_run --out <tmp>/out --work <tmp>/work
```

`--dry_run` replaces MERL with deterministic random encoders. It tests data loading, windowing, resampling, lead swap, prompt construction, calibration, `evaluate()` and the output files; the metrics it prints are meaningless. Outputs are prefixed `DRYRUN_`.

The `MERLEncoder` wrapper (checkpoint unwrapping, `module.` stripping, strict-key check, text and ECG paths, for both ResNet18 and ViT-tiny) was also tested. That test used a tiny random BERT standing in for MedCPT, with transformers 5.22.1, torch 2.15 CPU, Python 3.11.
