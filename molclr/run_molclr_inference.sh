#!/bin/bash
#SBATCH --job-name=molclr_inference
#SBATCH --output=/home/jmatthi2/deep_learning/COMET/molclr/logs/inference_%j.out
#SBATCH --error=/home/jmatthi2/deep_learning/COMET/molclr/logs/inference_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# run_molclr_inference.sh
# Runs MolCLR GIN embedding extraction on lance_lipid_smiles.csv.
#
# Usage:
#   sbatch run_molclr_inference.sh

WORK_DIR="/home/jmatthi2/deep_learning/COMET/molclr"
REPO_DIR="${WORK_DIR}/MolCLR"
ENV_PATH="/home/jmatthi2/deep_learning/env_molclr"

SMILES_CSV="${WORK_DIR}/lance_lipid_smiles.csv"
OUTPUT="${WORK_DIR}/molclr_lipid_embeddings.npy"

module load anaconda3/2024.02-1
conda activate "$ENV_PATH"

echo "============================================"
echo "Running MolCLR inference"
echo "============================================"

python "${WORK_DIR}/molclr_inference.py" \
    --smiles_csv "$SMILES_CSV" \
    --repo_dir   "$REPO_DIR" \
    --output     "$OUTPUT"

echo "Done. Output: $OUTPUT"
