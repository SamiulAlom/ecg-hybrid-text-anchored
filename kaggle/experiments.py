"""Experiment batches. Each batch becomes one Kaggle GPU kernel (runner stops launching new runs at ~11 h and
skips runs whose results already exist, so a batch can be re-pushed to finish).

Placeholders: {PREP} prep folder, {EMB} BioLORD full descriptions, {EMB_NAME} BioLORD names only,
{EMB_MPNET} general-domain MPNet full descriptions.
"""

TARGETS = ["CPSC", "PTBXL", "GEORGIA", "SHAONING"]
BASE = "--prep {PREP} --mode loso --epochs 30"
OURS = "--head text --text_emb {EMB} --labels mask_pseudo --hierarchy --mixstyle"

# Comparison methods (all share encoder, augmentation incl. lead dropout 0.1, optimiser and selection rule)
METHODS = {
    "naive": "--head linear --labels naive",
    "mask": "--head linear --labels mask",
    "listyle": "--head linear --labels mask --hierarchy",  # Li et al. 2024-style label handling
    "coral": "--head linear --labels mask --hierarchy --dg coral",
    "dann": "--head linear --labels mask --hierarchy --dg dann",
    "groupdro": "--head linear --labels mask --hierarchy --dg groupdro",
    "mixstyle": "--head linear --labels mask --hierarchy --mixstyle",
    "ours": OURS,
}
# Ablations isolating each component of "ours"
ABLATIONS = {
    "abl-textmask": "--head text --text_emb {EMB} --labels mask --hierarchy --mixstyle",      # - pseudo-labels
    "abl-linpseudo": "--head linear --labels mask_pseudo --hierarchy --mixstyle",             # - text head
    "abl-query": "--head query --labels mask_pseudo --hierarchy --mixstyle",                  # learned queries, no text
    "abl-noh": "--head text --text_emb {EMB} --labels mask_pseudo --mixstyle",                # - hierarchy
    "abl-nomix": "--head text --text_emb {EMB} --labels mask_pseudo --hierarchy",             # - MixStyle
    "abl-name": "--head text --text_emb {EMB_NAME} --labels mask_pseudo --hierarchy --mixstyle",   # names only
    "abl-mpnet": "--head text --text_emb {EMB_MPNET} --labels mask_pseudo --hierarchy --mixstyle",  # general encoder
}


def loso(spec, targets, seeds):
    return [f"{BASE} --target {t} {spec[m]} --seed {s} --tag loso-{t}-{m}-s{s}"
            for s in seeds for t in targets for m in spec]


def e4(rates, seed=0):
    # controlled annotation removal on PTB-XL in-distribution (random record split)
    spec = {"naive": METHODS["naive"], "listyle": METHODS["listyle"], "ours": OURS}
    return [f"--prep {{PREP}} --mode indist --target PTBXL --epochs 30 {v} --drop_rate {r} --seed {seed} "
            f"--tag e4-PTBXL-{k}-r{r}-s{seed}" for r in rates for k, v in spec.items()]


BATCHES = {
    # GPU session 1 and 2 run concurrently: main comparison + ablations, seed 0
    "main-a": loso(METHODS, ["CPSC", "PTBXL"], [0]) + loso(ABLATIONS, ["CPSC", "PTBXL"], [0]),
    "main-b": loso(METHODS, ["GEORGIA", "SHAONING"], [0]) + loso(ABLATIONS, ["GEORGIA", "SHAONING"], [0]),
    # second wave: extra seeds for ours + strongest baselines (edit list after main results), E4
    "seeds": loso({k: METHODS[k] for k in ("ours", "listyle", "mixstyle")}, TARGETS, [1, 2]),
    "e4": e4([0.0, 0.25, 0.5]),
}
