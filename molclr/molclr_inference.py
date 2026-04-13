
"""
molclr_inference.py
 
Loads the MolCLR pretrained GIN model from the cloned repo and extracts
graph-level embeddings for each lipid SMILES in lance_lipid_smiles.csv.
 
Output: molclr_lipid_embeddings.npy
  dict mapping SMILES string -> numpy array (300-dim)
 
Usage:
  python molclr_inference.py \
      --smiles_csv /path/to/lance_lipid_smiles.csv \
      --repo_dir   /path/to/MolCLR \
      --output     /path/to/molclr_lipid_embeddings.npy
"""
 
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
 
# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--smiles_csv", required=True,
                    help="CSV with a 'smiles' column")
parser.add_argument("--repo_dir",   required=True,
                    help="Path to cloned MolCLR repo")
parser.add_argument("--output",     required=True,
                    help="Output .npy file path")
args = parser.parse_args()
 
# Add MolCLR repo to path so we can import its modules
sys.path.insert(0, args.repo_dir)
 
# ---------------------------------------------------------------------------
# Imports from MolCLR repo
# ---------------------------------------------------------------------------
from models.ginet_molclr import GINet
from torch_geometric.data import Data, DataLoader
from rdkit import Chem
 
# ---------------------------------------------------------------------------
# Atom / bond featurisation — matches MolCLR dataset.py exactly.
# x      : (N, 2) Long  — [atomic_num_idx, chirality_idx]
# edge_attr: (E, 2) Long — [bond_type_idx, bond_direction_idx]
# ---------------------------------------------------------------------------
 
ATOM_LIST = list(range(1, 119))          # atomic numbers 1-118
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]
 
BOND_TYPE_LIST = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
BOND_DIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
]
 
def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
 
    atom_features = []
    for atom in mol.GetAtoms():
        anum = atom.GetAtomicNum()
        chirality = atom.GetChiralTag()
        a_idx = ATOM_LIST.index(anum) if anum in ATOM_LIST else 0
        c_idx = CHIRALITY_LIST.index(chirality) if chirality in CHIRALITY_LIST else 0
        atom_features.append([a_idx, c_idx])
 
    x = torch.tensor(atom_features, dtype=torch.long)
 
    edge_index = []
    edge_attr  = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bt = bond.GetBondType()
        bd = bond.GetBondDir()
        bt_idx = BOND_TYPE_LIST.index(bt) if bt in BOND_TYPE_LIST else 0
        bd_idx = BOND_DIR_LIST.index(bd)  if bd in BOND_DIR_LIST  else 0
        edge_index += [[i, j], [j, i]]
        edge_attr  += [[bt_idx, bd_idx], [bt_idx, bd_idx]]
 
    if len(edge_index) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, 2), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr  = torch.tensor(edge_attr,  dtype=torch.long)
 
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
 
# ---------------------------------------------------------------------------
# Load SMILES
# ---------------------------------------------------------------------------
df = pd.read_csv(args.smiles_csv)
smiles_list = df['smiles'].dropna().unique().tolist()
print(f"Loaded {len(smiles_list)} unique SMILES.")
 
# Build graphs
graphs = []
valid_smiles = []
for smi in smiles_list:
    g = smiles_to_graph(smi)
    if g is not None:
        graphs.append(g)
        valid_smiles.append(smi)
    else:
        print(f"  WARNING: could not parse SMILES: {smi}")
 
print(f"Successfully converted {len(valid_smiles)} SMILES to graphs.")
 
# ---------------------------------------------------------------------------
# Load pretrained MolCLR GIN model
# ---------------------------------------------------------------------------
ckpt_path = os.path.join(args.repo_dir, "ckpt", "pretrained_gin", "checkpoints", "model.pth")
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(
        f"Pretrained checkpoint not found at {ckpt_path}.\n"
        "Expected: MolCLR/ckpt/pretrained_gin/checkpoints/model.pth"
    )
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
 
model = GINet(
    num_layer=5,
    emb_dim=300,
    feat_dim=512,
    drop_ratio=0.0,
    pool='mean'
)
 
state_dict = torch.load(ckpt_path, map_location=device)
backbone_state = {k: v for k, v in state_dict.items()
                  if not k.startswith('projection_head')}
model.load_state_dict(backbone_state, strict=False)
model = model.to(device)
model.eval()
print("Loaded pretrained MolCLR GIN weights.")
 
# ---------------------------------------------------------------------------
# Run inference
# ---------------------------------------------------------------------------
loader = DataLoader(graphs, batch_size=16, shuffle=False)
 
all_embeddings = []
with torch.no_grad():
    for batch in loader:
        batch = batch.to(device)
        h, _ = model(batch)
        all_embeddings.append(h.cpu().numpy())
 
embeddings = np.vstack(all_embeddings)
print(f"Embeddings shape: {embeddings.shape}")  # (N, 300)
 
# ---------------------------------------------------------------------------
# Save as SMILES -> numpy dict
# ---------------------------------------------------------------------------
result = {smi: emb for smi, emb in zip(valid_smiles, embeddings)}
np.save(args.output, result)
print(f"Saved embeddings to {args.output}")