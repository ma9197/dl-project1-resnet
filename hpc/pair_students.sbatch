#!/bin/bash
#SBATCH --account=cs_gy_6923-2026fa
#SBATCH --partition=c24m170-a100-2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=22
#SBATCH --mem=120G
#SBATCH --time=14:00:00
#SBATCH --output=runs/pair_students.slurm.log
# two runs side by side, one per GPU (single-GPU A100 nodes were not booting)
cd /scratch/$USER/project1
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
PY=../venv/bin/python
S="--channels 84 168 336 --blocks 2 2 2 --full-train --trivialaug --epochs 800 --teachers big_t1 big_t2 wide_t1 wide_t2 --kd-T 1 --kd-alpha 0.1"
CUDA_VISIBLE_DEVICES=0 $PY train.py --name kd4_B_e800 $S > runs/kd4_B_e800.log 2>&1 & p1=$!
CUDA_VISIBLE_DEVICES=1 $PY train.py --name kd5_B_e800_cm $S --cutmix 0.5 > runs/kd5_B_e800_cm.log 2>&1 & p2=$!
wait $p1 && wait $p2
