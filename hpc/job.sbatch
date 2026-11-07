#!/bin/bash
# One training run on one A100. Usage (from the repo folder): sbatch -J <name> hpc/job.sbatch <train.py arguments>
#SBATCH --account=cs_gy_6923-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=11
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --output=runs/%x.log
cd /scratch/$USER/project1
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
../venv/bin/python train.py "$@"
