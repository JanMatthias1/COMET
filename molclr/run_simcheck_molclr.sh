#!/bin/bash
#SBATCH --job-name=molclr_simcheck
#SBATCH --output=/home/jmatthi2/deep_learning/COMET/molclr/logs/simcheck_%j.out
#SBATCH --error=/home/jmatthi2/deep_learning/COMET/molclr/logs/simcheck_%j.err
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

# run_simcheck_molclr.sh
# Runs pairwise cosine similarity analysis on MolCLR lipid embeddings.
#
# Usage:
#   sbatch run_simcheck_molclr.sh

WORK_DIR="/home/jmatthi2/deep_learning/COMET/molclr"
ENV_PATH="/home/jmatthi2/deep_learning/env_molclr"

EMBEDDINGS="${WORK_DIR}/molclr_lipid_embeddings.npy"
OUTPUT_PLOT="${WORK_DIR}/molclr_similarity_matrix.png"

module load anaconda3/2024.02-1
conda activate "$ENV_PATH"

echo "============================================"
echo "Running MolCLR similarity check"
echo "============================================"

python "${WORK_DIR}/similarity_check_molclr.py" \
    --embeddings  "$EMBEDDINGS" \
    --output_plot "$OUTPUT_PLOT"

echo "Done."
