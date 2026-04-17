#!/bin/bash
#SBATCH --job-name=comet_infomax
#SBATCH --output=/home/jmatthi2/deep_learning/COMET_matias/experiments/logs/infomax_%j.out
#SBATCH --error=/home/jmatthi2/deep_learning/COMET_matias/experiments/logs/infomax_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=l40s,a100,h100
#SBATCH --gres=gpu:1

 
module load anaconda3/2024.02-1
conda activate /home/jmatthi2/deep_learning/comet_env
 
cd /home/jmatthi2/deep_learning/COMET_matias/experiments
 
python run_infomax.py
 