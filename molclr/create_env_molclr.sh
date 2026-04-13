#!/bin/bash
#SBATCH --job-name=create_env_molclr
#SBATCH --output=/home/jmatthi2/deep_learning/COMET/molclr/logs/create_env_%j.out
#SBATCH --error=/home/jmatthi2/deep_learning/COMET/molclr/logs/create_env_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# create_env_molclr.sh
# Clones MolCLR repo and builds conda environment.
# Pretrained GIN weights are included in the repo at ckpt/pretrained_gin/.
#
# Usage:
#   mkdir -p /home/jmatthi2/deep_learning/COMET/molclr/logs
#   sbatch create_env_molclr.sh

WORK_DIR="/home/jmatthi2/deep_learning/COMET/molclr"
REPO_DIR="${WORK_DIR}/MolCLR"
ENV_PATH="/home/jmatthi2/deep_learning/env_molclr"

module load anaconda3/2024.02-1

echo "============================================"
echo "Setting up MolCLR"
echo "============================================"

# Clone repo if not already present
if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning MolCLR repo..."
    git clone https://github.com/yuyangw/MolCLR.git "$REPO_DIR"
else
    echo "Repo already exists, skipping clone."
fi

# Force-delete existing env to ensure clean CPU rebuild
if [ -d "$ENV_PATH" ]; then
    echo "Removing existing environment..."
    conda env remove -y -p "$ENV_PATH"
fi

echo "Creating conda environment..."
conda create -y -p "$ENV_PATH" python=3.7

conda activate "$ENV_PATH"

echo "Installing PyTorch (CPU)..."
pip install torch==1.7.1+cpu torchvision==0.8.2+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

echo "Installing PyG dependencies (CPU)..."
pip install torch-geometric==1.6.3 \
    torch-sparse==0.6.9 \
    torch-scatter==2.0.6 \
    -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html

echo "Installing other dependencies..."
pip install PyYAML
conda install -y -c conda-forge rdkit=2020.09.1.0

echo "============================================"
echo "Done. Verify with:"
echo "  conda activate $ENV_PATH"
echo "  python -c 'import torch; import torch_geometric; import rdkit; print(\"OK\")'"
echo "============================================"
