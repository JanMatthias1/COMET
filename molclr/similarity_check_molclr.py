"""
similarity_check_molclr.py

Pairwise cosine similarity analysis for MolCLR lipid embeddings.
Identical logic to similarity_check_schnet.py and similarity_check_3dinfomax.py.

Usage:
  python similarity_check_molclr.py \
      --embeddings /path/to/molclr_lipid_embeddings.npy \
      --output_plot /path/to/molclr_similarity_matrix.png
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

parser = argparse.ArgumentParser()
parser.add_argument("--embeddings",   required=True,
                    help=".npy file from molclr_inference.py")
parser.add_argument("--output_plot",  required=True,
                    help="Path for output heatmap PNG")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load embeddings
# ---------------------------------------------------------------------------
data = np.load(args.embeddings, allow_pickle=True).item()
smiles_list = list(data.keys())
embeddings  = np.array([data[s] for s in smiles_list])

print(f"Loaded {len(smiles_list)} embeddings, dim={embeddings.shape[1]}")

# ---------------------------------------------------------------------------
# Pairwise cosine similarity
# ---------------------------------------------------------------------------
sim_matrix = cosine_similarity(embeddings)

# ---------------------------------------------------------------------------
# Print matrix
# ---------------------------------------------------------------------------
# Shorten labels for display
labels = []
for smi in smiles_list:
    lbl = smi[:20] + "..." if len(smi) > 20 else smi
    labels.append(lbl)

header = f"{'':25s}" + "".join(f"{l:25s}" for l in labels)
print("\nPairwise Cosine Similarity Matrix:")
print(header)
for i, row_label in enumerate(labels):
    row = f"{row_label:25s}" + "".join(f"{sim_matrix[i, j]:25.4f}" for j in range(len(labels)))
    print(row)

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
n = len(smiles_list)
off_diag = sim_matrix[np.triu_indices(n, k=1)]

print(f"\nOff-diagonal cosine similarity stats:")
print(f"  Mean:   {off_diag.mean():.4f}")
print(f"  Std:    {off_diag.std():.4f}")
print(f"  Min:    {off_diag.min():.4f}")
print(f"  Max:    {off_diag.max():.4f}")

# Warn if embeddings are collapsed
if off_diag.mean() > 0.90:
    print("\n  WARNING: High mean similarity (>0.90) — embeddings may be collapsed.")
elif off_diag.std() < 0.05:
    print("\n  WARNING: Low std (<0.05) — low discriminability between molecules.")
else:
    print("\n  Embeddings appear discriminative.")

# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(sim_matrix, vmin=-1.0, vmax=1.0, cmap='coolwarm')
plt.colorbar(im, ax=ax, label='Cosine Similarity')
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(labels, rotation=90, fontsize=7)
ax.set_yticklabels(labels, fontsize=7)
ax.set_title('MolCLR GIN — Pairwise Cosine Similarity (Lipid Embeddings)')
plt.tight_layout()
plt.savefig(args.output_plot, dpi=150)
print(f"\nSaved heatmap to {args.output_plot}")
