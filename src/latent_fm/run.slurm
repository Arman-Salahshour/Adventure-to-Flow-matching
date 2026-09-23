#!/bin/bash
#SBATCH --job-name=latent_fm
#SBATCH --account=ec674
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

# Submit from the repo root:  mkdir -p logs && sbatch src/latent_fm/run.slurm
set -o errexit

source ~/.venvImg/bin/activate
cd "$SLURM_SUBMIT_DIR"

make -C src/latent_fm PYTHON=$(which python)
