"""Figure: cosine similarity between frozen text anchors (BioLORD-2023, full descriptions) for the diagnoses
of the scored vocabulary, ordered by hierarchical clustering. Uses only the embedding file used in training."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from labels import LABELS, SCORED_26  # noqa: E402

plt.rcParams.update({"font.size": 7, "font.family": "sans-serif", "axes.linewidth": 0.5})
z = np.load(os.path.join(HERE, "..", "src", "text_emb_biolord_full.npz"))
codes = z["codes"].tolist()
E = z["emb"][[codes.index(c) for c in SCORED_26]]
S = E @ E.T
order = leaves_list(linkage(E, "average", metric="cosine"))
S = S[np.ix_(order, order)]
names = [LABELS[SCORED_26[i]][0] for i in order]

fig, ax = plt.subplots(figsize=(17 / 2.54 * 0.62, 17 / 2.54 * 0.55))
im = ax.imshow(S, cmap="viridis", vmin=0.2, vmax=1.0)
ax.set_xticks(range(len(names)), names, rotation=90)
ax.set_yticks(range(len(names)), names)
ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
cb.set_label("cosine similarity of text anchors")
fig.tight_layout()
os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
fig.savefig(os.path.join(HERE, "figures", "fig_text_similarity.pdf"))
fig.savefig(os.path.join(HERE, "figures", "fig_text_similarity.png"), dpi=300)
print("saved fig_text_similarity; closest pairs:")
off = S - np.eye(len(S)) * 2
for k in np.argsort(-off, axis=None)[:12:2]:
    i, j = divmod(k, len(S))
    print(f"  {names[i]}-{names[j]}: {S[i, j]:.2f}")
