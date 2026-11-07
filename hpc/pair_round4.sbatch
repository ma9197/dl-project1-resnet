#!/bin/bash
#SBATCH --account=cs_gy_6923-2026fa
#SBATCH --partition=c24m170-a100-2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=22
#SBATCH --mem=120G
#SBATCH --time=14:00:00
#SBATCH --output=runs/pair_round4.slurm.log
# big teachers trained with CutMix, then two students that distill from them. One run per GPU.
cd /scratch/$USER/project1
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
PY=../venv/bin/python
T="--channels 160 320 640 --blocks 4 4 4 --full-train --trivialaug --epochs 400 --teacher-only --cutmix 1.0"
CUDA_VISIBLE_DEVICES=0 $PY train.py --name big_t3 $T --seed 23 > runs/big_t3.log 2>&1 & p1=$!
CUDA_VISIBLE_DEVICES=1 $PY train.py --name big_t4 $T --seed 24 > runs/big_t4.log 2>&1 & p2=$!
wait $p1 && wait $p2 || exit 1
S="--channels 84 168 336 --blocks 2 2 2 --full-train --trivialaug --epochs 1200 --kd-T 1 --kd-alpha 0.1 --cutmix 1.0"
CUDA_VISIBLE_DEVICES=0 $PY train.py --name kd8_B_e1200_cmT --teachers big_t3 big_t4 wide_t1 wide_t2 $S --seed 1 > runs/kd8_B_e1200_cmT.log 2>&1 & p1=$!
CUDA_VISIBLE_DEVICES=1 $PY train.py --name kd9_B_e1200_cmT4 --teachers big_t1 big_t2 big_t3 big_t4 $S --seed 2 > runs/kd9_B_e1200_cmT4.log 2>&1 & p2=$!
wait $p1 && wait $p2
