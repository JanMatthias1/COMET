#!/bin/bash
#SBATCH --job-name=heatmaps
#SBATCH --output=/home/jmatthi2/deep_learning/COMET/heatmaps/logs/heatmaps_%j.out
#SBATCH --error=/home/jmatthi2/deep_learning/COMET/heatmaps/logs/heatmaps_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1

mkdir -p /home/jmatthi2/deep_learning/COMET/heatmaps/logs
mkdir -p /home/jmatthi2/deep_learning/COMET/heatmaps/figures

module load anaconda3/2024.02-1
conda activate /home/jmatthi2/deep_learning/env_schnet

python /home/jmatthi2/deep_learning/COMET/heatmaps/plot_similarity_heatmaps.py
