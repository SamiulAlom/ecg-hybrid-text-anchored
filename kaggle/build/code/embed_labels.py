"""Embed every label description once with a frozen text encoder -> npz(codes, emb)."""
import argparse

import numpy as np
from sentence_transformers import SentenceTransformer

from labels import LABELS, text_of

MODELS = {
    "biolord": "FremyCompany/BioLORD-2023",          # clinical concept/definition encoder
    "pubmedbert": "NeuML/pubmedbert-base-embeddings",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",  # general-domain control
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="biolord", choices=MODELS)
    ap.add_argument("--style", default="full", choices=["full", "name"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    codes = list(LABELS)
    enc = SentenceTransformer(MODELS[a.model], device="cpu")
    emb = enc.encode([text_of(c, a.style) for c in codes], normalize_embeddings=True)
    np.savez(a.out, codes=np.array(codes), emb=emb.astype(np.float32))
    print(a.model, a.style, emb.shape, "->", a.out)


if __name__ == "__main__":
    main()
