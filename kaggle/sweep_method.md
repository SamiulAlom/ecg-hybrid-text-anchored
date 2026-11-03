# Method-v2 configuration sweep (one fold: GEORGIA held out, seed 0)

## Selection rule (fixed before any run; read this first)

- **Configurations are chosen on VALIDATION AUROC only**: `best_val_auroc` in each result JSON. This is the macro
  AUROC on the validation split of the held-in sources (CPSC, PTBXL, SHAONING). **Test metrics (`test_*`,
  `per_label`, `*_test.npz`) must not be opened or used for any choice.** The chosen configuration is then frozen
  and run on all four LOSO folds and all seeds.
- Baselines are **not** re-tuned. Their code paths are unchanged: all new flags default to off, and with the defaults
  the modified `train.py`/`models.py` give bit-identical histories, test metrics and test predictions to the code
  that produced `ecg-run-main-b`. This was checked on CPU for naive, mask, listyle, coral, dann, groupdro,
  mixstyle, ours-v1, query-v1 and linpseudo-v1.
- `--eval_ema --ema_warmup` (validate, select and test the EMA weights) is a **generic** improvement that any method
  can use. If the chosen configuration uses it, the baselines must also be rerun with `--eval_ema --ema_warmup` in
  every fold. The paper then reports the comparison with EMA for all methods, and can also report it without EMA
  for all methods. S1 vs S2 measures how much EMA adds on its own.
- `--diag` only adds logging: pseudo-label rates, pseudo loss, hierarchy violation, and the other model's val AUROC.
  It does not change training.

## Commands

Placeholders are as in `kaggle/experiments.py`: `{PREP}` is the prep folder and `{EMB}` is
`text_emb_biolord_full.npz`. Every command shares this common v2 part:

```
V2 = --head text --text_emb {EMB} --labels mask_pseudo --hier_soft --pseudo_soft --pseudo_start 3 --ema_warmup --mixstyle --diag
```

`--pseudo_start 3` is the end of the OneCycle warm-up (pct_start 0.1 of 30 epochs). The ramp (`--pseudo_ramp 0.3`,
the default) then reaches full weight at epoch 12.

They are ranked by expected strength. S5 and S6 are zero-shot proxies for S1 and S4 and are not candidates
themselves.

```
S1 --prep {PREP} --mode loso --epochs 30 --target GEORGIA --head text --text_emb {EMB} --labels mask_pseudo --hier_soft --pseudo_soft --pseudo_start 3 --ema_warmup --mixstyle --diag --residual both --eval_ema --seed 0 --tag sweep-GEORGIA-v2-both-ema-s0
S2 --prep {PREP} --mode loso --epochs 30 --target GEORGIA --head text --text_emb {EMB} --labels mask_pseudo --hier_soft --pseudo_soft --pseudo_start 3 --ema_warmup --mixstyle --diag --residual both --seed 0 --tag sweep-GEORGIA-v2-both-s0
S3 --prep {PREP} --mode loso --epochs 30 --target GEORGIA --head text --text_emb {EMB} --labels mask_pseudo --hier_soft --pseudo_soft --pseudo_start 3 --ema_warmup --mixstyle --diag --residual query --eval_ema --seed 0 --tag sweep-GEORGIA-v2-query-ema-s0
S4 --prep {PREP} --mode loso --epochs 30 --target GEORGIA --head text --text_emb {EMB} --labels mask_pseudo --hier_soft --pseudo_soft --pseudo_start 3 --ema_warmup --mixstyle --diag --eval_ema --seed 0 --tag sweep-GEORGIA-v2-none-ema-s0
S5 --prep {PREP} --mode loso --epochs 30 --target GEORGIA --head text --text_emb {EMB} --labels mask_pseudo --hier_soft --pseudo_soft --pseudo_start 3 --ema_warmup --mixstyle --diag --residual both --eval_ema --holdout 47665007 251146004 698252002 164917005 10370003 --seed 0 --tag sweep-GEORGIA-v2-both-ema-zsproxy-s0
S6 --prep {PREP} --mode loso --epochs 30 --target GEORGIA --head text --text_emb {EMB} --labels mask_pseudo --hier_soft --pseudo_soft --pseudo_start 3 --ema_warmup --mixstyle --diag --eval_ema --holdout 47665007 251146004 698252002 164917005 10370003 --seed 0 --tag sweep-GEORGIA-v2-none-ema-zsproxy-s0
```

(Drop the `S1`…`S6` labels when pasting them into the `CMDS` list of a run script.)

What each run tests:

| id | residual | final weights | purpose |
|----|----------|---------------|---------|
| S1 | both (per-class bias + query readout + pooled readout) | EMA | full v2 method |
| S2 | both | student | how much EMA selection adds (generic; see rule above) |
| S3 | query (bias + query readout only) | EMA | lighter residual, kept inside the text-query pathway |
| S4 | none (pure text-anchored head) | EMA | collapse fixes alone, no residual |
| S5 | as S1 + 5 diagnoses withheld | EMA | zero-shot proxy for S1 |
| S6 | as S4 + 5 diagnoses withheld | EMA | zero-shot proxy for S4 |

The withheld diagnoses in S5 and S6 are RAD, LQRSV, NSIVCB, QAb and PR. They are annotated by PTBXL and SHAONING,
have 114 to 188 validation positives each, take part in no implication rule, and are deliberately **not** the five
unseen test diagnoses of the SHAONING fold (LVHV, CCR, CCWR, PWC, UAb). They are scored on the **validation**
records (`val_holdout_auroc` in `history`), so no test data is used.

## Decision procedure (pre-registered)

1. Candidates are S1–S4. Take the one with the highest `best_val_auroc`.
2. Residual guard for unseen diagnoses: if the winner uses a residual, read `val_holdout_auroc` of S5 and S6 at
   their best-validation epoch (the epoch where `val_auroc` is highest). If S5 is more than 0.02 below S6, the
   residual hurts zero-shot transfer. In that case choose S4.
3. If two candidates are within 0.002 `best_val_auroc`, prefer the simpler one, in this order: S4 < S3 < S1. S2 is
   only chosen over S1 if it is better by more than 0.002.
4. Stability check, for reporting only: in the chosen run, the per-epoch `val_auroc` should show no drop larger than
   0.02 between consecutive epochs after `--pseudo_start`. `teacher_val_auroc` or `student_val_auroc` and
   `pseudo_pos_frac` are logged per epoch to document this.

For reference, the best validation AUROC of earlier GEORGIA runs is: mask/naive/listyle 0.948, mixstyle 0.946,
abl-textmask 0.941, abl-noh 0.939, ours-v1 0.912 (collapsed).

## Optional diagnostic (not part of selection)

Original method with logging, to document the collapse mechanism for the paper (pseudo-positive rate and teacher
val AUROC around epoch 6):

```
--prep {PREP} --mode loso --epochs 30 --target GEORGIA --head text --text_emb {EMB} --labels mask_pseudo --hierarchy --mixstyle --diag --seed 0 --tag diag-GEORGIA-ours-v1-s0
```

## After selection

Freeze the flags. Then run the chosen configuration (`ours-v2`) on CPSC, PTBXL, GEORGIA and SHAONING with seeds
0, 1 and 2. Add the v2 ablations, each removing one component:

- `--hier_soft` → `--hierarchy`
- remove `--pseudo_soft`
- remove `--residual`
- `--labels mask`

If EMA was chosen, also run the baselines with `--eval_ema --ema_warmup`.

---

# Hybrid sweep (GEORGIA held out, seed 0)

## Rules (fixed before any run)

- **Selection uses VALIDATION only.** The quantities are `best_val_auroc` and `val_holdout_auroc`, both computed on
  the validation split of the held-in sources. **No test metric is opened for any choice.**
- `--head hybrid` scores **validation with the same rule as test**: seen diagnoses by `--seen_score`, unseen or
  held-out diagnoses by the text head. `best_val_auroc` and checkpoint selection therefore already reflect the
  scoring rule.
- **Bar:** listyle+EMA on GEORGIA, seed 0. `best_val_auroc` = **0.9531**, read from
  `results/kaggle/ecg-final-b/results/loso-GEORGIA-listyleema-s0.json`. For reference, mixstyle+EMA is 0.9505,
  v2-S1 is 0.9456, and v2-S4 (pure text head) is 0.9390.
- Baseline code paths are unchanged. All hybrid flags default to off. Old and new code give bit-identical outputs
  for 13 configurations, including all baselines with and without EMA, v1 and v2, on two synthetic sets.
  `--head hybrid --text_w 0 --seen_score linear` reproduces listyle+EMA exactly.
- Every hybrid run uses exactly the listyle+EMA training recipe (`--labels mask --hierarchy --eval_ema --ema_warmup`,
  no MixStyle). Only the head differs, so the comparison with the bar is like-for-like.

## Commands

```
H1 --prep {PREP} --mode loso --epochs 30 --target GEORGIA --head hybrid --text_emb {EMB} --labels mask --hierarchy --eval_ema --ema_warmup --text_w 1.0 --seen_score linear --seed 0 --tag hsweep-GEORGIA-hyb-lin-s0
H2 --prep {PREP} --mode loso --epochs 30 --target GEORGIA --head hybrid --text_emb {EMB} --labels mask --hierarchy --eval_ema --ema_warmup --text_w 1.0 --seen_score avg --seed 0 --tag hsweep-GEORGIA-hyb-avg-s0
H3 --prep {PREP} --mode loso --epochs 30 --target GEORGIA --head hybrid --text_emb {EMB} --labels mask --hierarchy --eval_ema --ema_warmup --text_w 1.0 --seen_score avg --text_detach --seed 0 --tag hsweep-GEORGIA-hyb-avg-detach-s0
H4 --prep {PREP} --mode loso --epochs 30 --target GEORGIA --head hybrid --text_emb {EMB} --labels mask_pseudo --pseudo_soft --pseudo_start 3 --hierarchy --eval_ema --ema_warmup --text_w 1.0 --seen_score avg --diag --seed 0 --tag hsweep-GEORGIA-hyb-avg-pseudo-s0
Z1 --prep {PREP} --mode loso --epochs 30 --target GEORGIA --head hybrid --text_emb {EMB} --labels mask --hierarchy --eval_ema --ema_warmup --text_w 1.0 --seen_score avg --holdout 47665007 251146004 698252002 164917005 10370003 --seed 0 --tag hsweep-GEORGIA-hyb-avg-zsproxy-s0
Z2 --prep {PREP} --mode loso --epochs 30 --target GEORGIA --head hybrid --text_emb {EMB} --labels mask --hierarchy --eval_ema --ema_warmup --text_w 1.0 --seen_score avg --text_detach --holdout 47665007 251146004 698252002 164917005 10370003 --seed 0 --tag hsweep-GEORGIA-hyb-avg-detach-zsproxy-s0
```

(Drop the `H*` and `Z*` labels when pasting.)

| id | encoder gets text-head gradient | seen score | text branch pseudo-labels | what it tests |
|----|----|----|----|----|
| H1 | yes | linear | no | Does the text head, as an auxiliary task, improve the shared encoder? Seen scoring is the baseline's. |
| H2 | yes | mean of both probabilities | no | Same, plus ensembling the two heads on seen diagnoses. |
| H3 | no (`--text_detach`) | mean of both probabilities | no | Ensemble gain without any change to the encoder's training signal. Not bit-identical to listyle: the decoder's dropout draws random numbers. |
| H4 | yes | mean of both probabilities | soft, from epoch 3 | Whether distilling the EMA teacher into the text branch on unknown entries helps. |
| Z1 | as H2, 5 diagnoses withheld | — | — | zero-shot proxy for H1, H2 and H4 |
| Z2 | as H3, 5 diagnoses withheld | — | — | zero-shot proxy for H3 |

The withheld diagnoses are the same as in the v2 sweep: RAD, LQRSV, NSIVCB, QAb and PR. They are scored on
validation records. Reference values from that sweep, at the best-validation epoch: v2-S4 (pure text head) 0.679,
v2-S1 0.686.

Under `--hierarchy`, H4's text-branch pseudo-labels again face the positive-only parent columns that caused the v1
collapse. They are made safer by soft targets, the late start and the warmed-up EMA. Check `pseudo_pos_frac` and
the validation curve in H4 before trusting its number.

## Decision procedure (pre-registered)

1. **Seen performance.** Among H1–H4, take the highest `best_val_auroc`.
2. **Zero-shot guard.** The winner's proxy is Z2 if it uses `--text_detach`, otherwise Z1. Read `val_holdout_auroc`
   at that proxy run's best-validation epoch. The winner passes if this value is ≥ 0.659 (v2-S4's 0.679 minus 0.02).
   If it fails, go to the next-best H run whose proxy passes.
3. **Ties.** Within 0.001 `best_val_auroc`, prefer the simpler configuration: H1 < H2 < H3 < H4.
4. **Against the bar.** If the selected run's `best_val_auroc` is below 0.9531, the hybrid does not beat listyle+EMA
   on validation. Report that honestly; do not adjust anything further on this fold. The fallback claim is "seen
   performance of the linear head plus zero-shot ability": choose H1 if it passed the guard and is within 0.002 of
   the bar.
5. Freeze the chosen flags and run them on all 4 folds × 3 seeds. Ablations: `--text_w 0` (which equals
   listyle+EMA), `--seen_score linear` vs `avg`, and `--text_detach`.

All differences on one fold and one seed are expected to be of order 0.002–0.005. A configuration that wins here
still has to beat the baselines across all folds and seeds before the paper can claim a win.
