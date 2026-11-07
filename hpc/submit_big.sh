#!/bin/bash
# Round 2: 2 big teachers (160-320-640, 4 blocks per layer, 36.7M parameters, teachers only) in parallel,
# then 2 students (800 epochs) from the 2 big + 2 wide teachers, one with CutMix. Each job uses both GPUs
# of a 2-GPU A100 node (the 1-GPU nodes were not booting).
cd /scratch/$USER/project1
a=$(sbatch --parsable hpc/pair_big_teachers.sbatch)
sbatch -d afterok:$a hpc/pair_students.sbatch
squeue -u $USER
