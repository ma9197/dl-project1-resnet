#!/bin/bash
#SBATCH --account=cs_gy_6923-2026fa
#SBATCH --partition=c24m170-a100-2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=22
#SBATCH --mem=120G
#SBATCH --time=14:00:00
#SBATCH --output=runs/pair_big_teachers.slurm.log
# two runs side by side, one per GPU (single-GPU A100 nodes were not booting)
cd /scratch/$USER/project1
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
PY=../venv/bin/python
B="--channels 160 320 640 --blocks 4 4 4 --full-train --trivialaug --epochs 300 --teacher-only"
CUDA_VISIBLE_DEVICES=0 $PY train.py --name big_t1 $B --seed 21 > runs/big_t1.log 2>&1 & p1=$!
CUDA_VISIBLE_DEVICES=1 $PY train.py --name big_t2 $B --seed 22 > runs/big_t2.log 2>&1 & p2=$!
wait $p1 && wait $p2
