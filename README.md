# Hybrid text-anchored learning from heterogeneously annotated ECG databases

Code, run commands, logged results and paper sources for:

> **Hybrid text-anchored learning from heterogeneously annotated ECG databases for cross-database and open-vocabulary 12-lead ECG diagnosis.** S. Alom (manuscript in preparation).

Public 12-lead ECG databases each annotate a different subset of diagnoses, so pooling them naively turns "never coded here" into "absent". This repository implements:
- a **hybrid classifier**: a 1D SE-ResNet encoder read by a per-diagnosis linear head and a text-anchored label-query head that uses frozen BioLORD-2023 description embeddings;
- trained with a **masked asymmetric loss** on source-annotated entries only, with implication completion and weight averaging (EMA);
- evaluated **leave-one-database-out** on CPSC, PTB-XL, Georgia and Shaoxing–Ningbo (87,661 recordings, 43 diagnoses), against seven label-handling and domain-generalization baselines.

## Main results (3 seeds × 4 held-out databases)

| | Seen diagnoses, macro AUROC | Unseen diagnoses (Shaoxing–Ningbo), macro AUROC |
|---|---|---|
| Naive pooling | 85.6 | 53.2 |
| Masked loss | 86.3 | 52.1 |
| Li-style hierarchical mask | 86.3 | 51.3 |
| CORAL / DANN / GroupDRO / MixStyle | 86.1 / 85.6 / 86.1 / 86.1 | 51–53 |
| **Hybrid (ours)** | **86.6** | **67.0** |

Systematic zero-shot test (120 withheld diagnosis–database pairs, 3 seeds): macro AUROC 70.2, 88% of pairs above chance; MERL (report-pretrained, zero-shot) 77.3 on the same pairs, but 48.0–62.6 vs our 67.0 on the five genuinely unseen Shaoxing–Ningbo diagnoses.

See the paper (`paper/`) for statistics, ablations, the annotation-removal experiment and the full development history, including negative results.

## Repository layout

```
src/            preprocessing, labels and descriptions, model, training
  preprocess.py   WFDB records -> 100 Hz arrays (signals.npy, meta.csv, code_counts.csv)
  labels.py       vocabulary: SNOMED-CT code -> (abbreviation, name, description)
  embed_labels.py frozen text-encoder embeddings of the descriptions (text_emb_*.npz included)
  data.py         vocabulary/annotation rules, label matrices, implication rules, splits, annotation removal
  models.py       SE-ResNet encoder, linear head, text-anchored query head, hybrid model
  train.py        training/evaluation for every method (flags select the method)
  colab_run.py    parallel runner used on Google Colab
kaggle/         experiment definitions and the exact commands that were run
  experiments.py  methods, ablations and batches; build.py creates Kaggle kernels
  build/*/run.py  exact command lists of every Kaggle batch; build/code/colab_cmds.txt for Colab
  sweep_method.md pre-specified selection rules for the method sweeps
analysis/       aggregation, statistics, figures and tables
  build_final.py      collects the final runs into results/final with canonical tags
  aggregate.py        main tables, paired bootstrap, DeLong + Holm, figures
  seed_stats.py       Wilcoxon over folds x seeds, unseen and E2 summaries
  zeroshot_e3.py, calibration.py, make_final_figures.py, make_supp_tables.py, make_supp_e3.py
results/final/  the 153 run JSONs of E1/E2/ablations (metrics, per-label results, histories, configs)
results/e3/     the 76 run JSONs of the zero-shot (E3) and annotation-threshold experiments
results/merl_zeroshot/  MERL zero-shot results (three prompt variants x four databases)
paper/          LaTeX sources (Springer sn-jnl), generated tables and figures
notebooks/      Colab runner notebook
```

Per-record test predictions (`*_test.npz`, about 240 MB) are not included because of their size; they are needed only for the bootstrap/DeLong tests and are available on request.

## Reproducing

1. **Data.** Download the PhysioNet/CinC 2020 training data (CPSC, CPSC-Extra, PTB-XL, Georgia) and the Shaoxing–Ningbo database. On Kaggle they are available as `gamalasran/physionet-challenge-2020` and `bjoernjostein/shaoxing-and-ningbo-first-hospital-database`.
2. **Environment.** Python ≥ 3.10:
   ```bash
   pip install -r requirements.txt
   ```
3. **Preprocess:**
   ```bash
   python src/preprocess.py --roots <physionet_dir> <shaoxing_dir> --out prep
   ```
4. **Train one run.** This is the proposed method with Georgia held out:
   ```bash
   python src/train.py --prep prep --mode loso --epochs 30 --target GEORGIA --head hybrid \
     --text_emb src/text_emb_biolord_full.npz --labels mask --hierarchy --eval_ema --ema_warmup \
     --text_w 1.0 --seen_score avg --seed 0 --tag loso-GEORGIA-hyb-avg-s0 --out results/run
   ```
   Every baseline, ablation and experiment command is listed in `kaggle/build/*/run.py` and `kaggle/build/code/colab_cmds.txt`. One run takes about 12 minutes on a single GPU (P100/T4/A100).
5. **Aggregate:**
   ```bash
   python analysis/build_final.py
   python analysis/aggregate.py --results results/final --out paper/generated
   python analysis/seed_stats.py
   python analysis/make_final_figures.py
   python analysis/make_supp_tables.py
   ```

## Citation

Citation details will be added on publication.

## License

Code: MIT (see `LICENSE`). The ECG databases are subject to their own licenses; see PhysioNet.
