import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from numpy.linalg import norm
import os

base_dir = "/home/jmatthi2/deep_learning/COMET/heatmaps"
output_dir = os.path.join(base_dir, "figures")
os.makedirs(output_dir, exist_ok=True)

# Load name -> smiles mapping from CSV
csv_path = os.path.join(base_dir, "lance_lipid_smiles.csv")
df = pd.read_csv(csv_path)
smiles_to_name = dict(zip(df['smiles'], df['name']))

embedding_files = {
    "GIN":    "gin_lipid_embeddings.npy",
    "MolCLR": "molclr_lipid_embeddings.npy",
    "SchNet": "schnet_lipid_embeddings.npy",
    "UniMol": "unimol_lipid_embeddings.npy",
    "3D Infomax": "3dinfomax_lipid_embeddings.npy"
}

# --- Pass 1: compute all similarity matrices and find global vmin ---
all_sim_matrices = {}
all_labels = {}

for model_name, fname in embedding_files.items():
    embeddings = np.load(os.path.join(base_dir, fname), allow_pickle=True).item()
    smiles_list = list(embeddings.keys())
    emb_matrix = np.stack(list(embeddings.values()))
    n = len(smiles_list)

    labels = [smiles_to_name.get(s, s[:20] + '...') for s in smiles_list]

    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim_matrix[i, j] = np.dot(emb_matrix[i], emb_matrix[j]) / (
                norm(emb_matrix[i]) * norm(emb_matrix[j])
            )

    all_sim_matrices[model_name] = sim_matrix
    all_labels[model_name] = labels

# Global scale: shared vmin across all models, vmax always 1.0
global_vmin = min(sim[~np.eye(len(labels), dtype=bool)].min()
                  for sim, labels in zip(all_sim_matrices.values(), all_labels.values()))
global_vmax = 1.0
print(f"Global vmin={global_vmin:.4f}  vmax={global_vmax:.4f}")

# --- Pass 2: plot ---
for model_name in embedding_files:
    print(f"\n=== {model_name} ===")
    sim_matrix = all_sim_matrices[model_name]
    labels = all_labels[model_name]
    n = len(labels)

    # Warn if any SMILES had no name match
    for l in labels:
        if l.endswith('...'):
            print(f"  WARNING - no name found for label: {l}")

    # Print warnings for nearly identical embeddings
    print("  Similarity warnings (sim > 0.99):")
    found = False
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] > 0.99:
                found = True
                print(f"    WARNING - nearly identical embeddings:")
                print(f"      {labels[i]}")
                print(f"      {labels[j]}")
                print(f"      Cosine sim: {sim_matrix[i, j]:.4f}")
    if not found:
        print("    None found.")

    # Off-diagonal stats
    mask = ~np.eye(n, dtype=bool)
    off_diag = sim_matrix[mask]
    print(f"  Off-diagonal  mean={off_diag.mean():.4f}  std={off_diag.std():.4f}"
          f"  min={off_diag.min():.4f}  max={off_diag.max():.4f}")

    # Plot similarity matrix
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(sim_matrix, cmap='plasma', vmin=global_vmin, vmax=global_vmax)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    # Annotate cells
    mid = global_vmin + (global_vmax - global_vmin) * 0.6
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{sim_matrix[i, j]:.2f}',
                    ha='center', va='center', fontsize=6,
                    color='black' if sim_matrix[i, j] > mid else 'white')

    plt.colorbar(im, ax=ax, label='Cosine Similarity')
    ax.set_title(f'Pairwise Cosine Similarity of {model_name} Lipid Embeddings',
                 fontsize=13, pad=20)
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"{model_name.lower()}_similarity_matrix.png")
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {output_path}")

print("\nDone.")