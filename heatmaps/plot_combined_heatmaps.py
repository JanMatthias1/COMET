import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from numpy.linalg import norm
import os
import pickle

def load_npy_embeddings(filepath):
    """Load embeddings from .npy file using raw pickle with bytes encoding."""
    with open(filepath, 'rb') as f:
        # Read numpy header
        magic = f.read(6)
        if magic[:2] != b'\x93N':
            raise ValueError(f"Not a valid numpy file: {filepath}")
        
        version = f.read(2)
        header_len_bytes = f.read(2)
        
        # Get header length
        header_len = int.from_bytes(header_len_bytes, byteorder='little')
        header = f.read(header_len)
        
        # Rest is pickle data - load with bytes encoding to avoid module issues
        remaining = f.read()
    
    # Load pickle with bytes encoding
    try:
        raw_data = pickle.loads(remaining, encoding='bytes')
        
        # If it's a 0-dimensional array, extract the item (the dictionary)
        if isinstance(raw_data, np.ndarray) and raw_data.ndim == 0:
            embeddings = raw_data.item()
        else:
            embeddings = raw_data
        
        # Convert bytes keys/values to strings
        if isinstance(embeddings, dict):
            result = {}
            for k, v in embeddings.items():
                key = k.decode('utf-8') if isinstance(k, bytes) else k
                val = v if isinstance(v, np.ndarray) else np.array(v)
                result[key] = val
            return result
        return embeddings
    except Exception as e:
        print(f"Error loading pickle from {filepath}: {e}")
        return {}

base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "figures")
os.makedirs(output_dir, exist_ok=True)

# Load name -> smiles mapping from CSV
csv_path = os.path.join(base_dir, "lance_lipid_smiles.csv")
df = pd.read_csv(csv_path)
smiles_to_name = dict(zip(df['smiles'], df['name']))

# Only 4 models: GIN, SchNet, UniMol, 3D Infomax (no MolCLR)
embedding_files = {
    "GIN":    "gin_lipid_embeddings.npy",
    "SchNet": "schnet_lipid_embeddings.npy",
    "UniMol": "unimol_lipid_embeddings.npy",
    "3D Infomax": "3dinfomax_lipid_embeddings.npy"
}

# --- Pass 1: compute all similarity matrices and find global vmin ---
all_sim_matrices = {}
all_labels = {}

for model_name, fname in embedding_files.items():
    fpath = os.path.join(base_dir, fname)
    print(f"Loading {model_name}...")
    embeddings = load_npy_embeddings(fpath)
    
    if not embeddings:
        print(f"ERROR: Could not load {fname}")
        continue
    
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
if not all_sim_matrices:
    print("ERROR: No embeddings loaded successfully!")
    exit(1)

global_vmin = min(sim[~np.eye(len(labels), dtype=bool)].min()
                  for sim, labels in zip(all_sim_matrices.values(), all_labels.values()))
global_vmax = 1.0
print(f"Global vmin={global_vmin:.4f}  vmax={global_vmax:.4f}")

# --- Pass 2: create combined figure with 2x2 subplots ---
# Collect successfully loaded models
loaded_models = [model for model in embedding_files.keys() if model in all_sim_matrices]
print(f"Successfully loaded models: {loaded_models}")

if len(loaded_models) != 4:
    print(f"WARNING: Expected 4 models, but only {len(loaded_models)} loaded successfully")

fig, axes = plt.subplots(2, 2, figsize=(26, 22))  # Increased width to make room for colorbar
axes_flat = axes.flatten()

for idx, model_name in enumerate(loaded_models):
    print(f"\n=== {model_name} ===")
    sim_matrix = all_sim_matrices[model_name]
    labels = all_labels[model_name]
    n = len(labels)
    ax = axes_flat[idx]

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
    im = ax.imshow(sim_matrix, cmap='plasma', vmin=global_vmin, vmax=global_vmax)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)

    # Annotate cells
    mid = global_vmin + (global_vmax - global_vmin) * 0.6
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{sim_matrix[i, j]:.2f}',
                    ha='center', va='center', fontsize=5,
                    color='black' if sim_matrix[i, j] > mid else 'white')

    ax.set_title(f'{model_name}', fontsize=26, pad=15, fontweight='bold')

# Adjust layout to make room for colorbar
plt.subplots_adjust(right=0.85)  # Leave 15% space on the right

# Add colorbar
cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])  # Position in the reserved space
cbar = fig.colorbar(im, cax=cbar_ax, label='Cosine Similarity')
cbar.ax.tick_params(labelsize=24)  # Make tick labels size 24
cbar.ax.set_ylabel('Cosine Similarity', fontsize=40)  # Keep label at 40

fig.suptitle('Pairwise Cosine Similarity of Lipid Embeddings', fontsize=28, y=0.995, fontweight='bold')

# Force a draw to ensure everything renders properly
fig.canvas.draw()

output_path = os.path.join(output_dir, "combined_similarity_heatmaps.png")
plt.savefig(output_path, dpi=600, bbox_inches='tight')
plt.close()
print(f"\nSaved combined figure to {output_path}")

print("\nDone.")
