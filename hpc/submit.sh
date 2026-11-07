#!/bin/bash
# 2 wide teachers in parallel, then 2 students (500 and 800 epochs) once both teachers are done.
# Needs runs/teacher_s1 and runs/final_B_ta_e200 (trained locally, see experiments.sh) copied to the cluster.
cd /scratch/$USER/project1
W="--blocks 2 2 2 --full-train --trivialaug"
a=$(sbatch --parsable -J wide_t1 hpc/job.sbatch --name wide_t1 --channels 128 256 512 $W --epochs 300 --seed 11 --teacher-only)
b=$(sbatch --parsable -J wide_t2 hpc/job.sbatch --name wide_t2 --channels 128 256 512 $W --epochs 300 --seed 12 --teacher-only)
T="--teachers wide_t1 wide_t2 teacher_s1 final_B_ta_e200 --kd-T 1 --kd-alpha 0.1"
sbatch -d afterok:$a:$b -J kd2_B_e500 hpc/job.sbatch --name kd2_B_e500 --channels 84 168 336 $W --epochs 500 $T
sbatch -d afterok:$a:$b -J kd3_B_e800 hpc/job.sbatch --name kd3_B_e800 --channels 84 168 336 $W --epochs 800 $T
squeue -u $USER
