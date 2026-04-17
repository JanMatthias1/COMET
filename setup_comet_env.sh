#!/bin/bash
#SBATCH --job-name=comet_env_setup
#SBATCH --output=logs/comet_env_setup_%j.out
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

module load anaconda3/2024.02-1

ENV_PATH="/home/jmatthi2/deep_learning/comet_matias_env"

echo "Creating environment at ${ENV_PATH}..."
conda create --prefix "${ENV_PATH}" python=3.10 -y

source activate "${ENV_PATH}"

echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "Installing requirements..."
pip install \
    numpy==1.26.4 \
    pandas==2.2.2 \
    numexpr==2.10.1 \
    scipy==1.13.0 \
    rdkit==2023.9.5 \
    lmdb==1.4.1 \
    biopython==1.83 \
    networkx==3.3 \
    scikit-learn==1.4.2 \
    tensorboard==2.16.2 \
    tqdm

echo "Checking for setup.py / pyproject.toml..."
if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
    echo "Installing package in editable mode..."
    pip install -e .
fi

echo "Done."
