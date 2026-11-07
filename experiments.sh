#!/bin/bash
# Every experiment reported in the README / report, in the order they were run.
# Each run writes runs/<name>/{log.json,summary.json,model.pt}. Times are on one RTX 4090 Laptop GPU.

# 1) 15-epoch pilots: which layout to use under 5M parameters (~2 min each)
python train.py --name pilot_A --channels 64 128 256      --blocks 3 3 3   --epochs 15
python train.py --name pilot_B --channels 84 168 336      --blocks 2 2 2   --epochs 15
python train.py --name pilot_C --channels 64 128 256 512  --blocks 1 1 1 1 --epochs 15
python train.py --name pilot_D --channels 64 128 256 320  --blocks 2 2 2 1 --epochs 15

# 2) 50 epochs, layout B, with / without TrivialAugment (~7 min each)
python train.py --name e50_B_base --channels 84 168 336 --blocks 2 2 2 --epochs 50
python train.py --name e50_B_ta   --channels 84 168 336 --blocks 2 2 2 --epochs 50 --trivialaug

# 3) 200 epochs on all 50k training images, with / without TrivialAugment (~32 min each)
python train.py --name final_B_base_e200 --channels 84 168 336 --blocks 2 2 2 --epochs 200 --full-train
python train.py --name final_B_ta_e200   --channels 84 168 336 --blocks 2 2 2 --epochs 200 --full-train --trivialaug

# 4) Kaggle submission from the final model
python predict.py --run final_B_ta_e200

# 5) Distillation: 2 more teachers (new seeds), then a student of the same size trained to match the
#    average prediction of 4 teachers on the same augmented batch (T=1, since the teachers used label smoothing)
python train.py --name teacher_s1 --channels 84 168 336 --blocks 2 2 2 --epochs 200 --full-train --trivialaug --seed 1
python train.py --name teacher_s2 --channels 84 168 336 --blocks 2 2 2 --epochs 200 --full-train --trivialaug --seed 2
python train.py --name kd_B_e300 --channels 84 168 336 --blocks 2 2 2 --epochs 300 --full-train --trivialaug \
    --teachers final_B_ta_e200 final_B_base_e200 teacher_s1 teacher_s2 --kd-T 1 --kd-alpha 0.1
python predict.py --run kd_B_e300

# 6) Stronger teachers on NYU HPC (A100): 2 wider networks (11M parameters, used only as teachers, never
#    submitted), then two longer distillation runs for the 4.78M student. See hpc/ (setup.sh, then submit.sh).
bash hpc/submit.sh

# 7) Round 2 on HPC: bigger teachers (36.7M parameters, never submitted) and CutMix for the student. See hpc/submit_big.sh
bash hpc/submit_big.sh

# 8) Round 3 on HPC: longer students (1200 epochs) with more CutMix
sbatch hpc/pair_students2.sbatch

# 9) Round 4 on HPC: big teachers trained with CutMix, then students that distill from them (kd8 is one of the 2 final picks)
sbatch hpc/pair_round4.sbatch

# 10) Round 5 on HPC: the kd8 recipe with a new seed (kd10, the final model). kd9 was cancelled by the cluster both times.
sbatch hpc/pair_round5.sbatch

# 11) Final Kaggle files, and the test-time ideas I tried and did not use
python predict.py --run kd10_B_e1200_cmT_s3
python predict.py --run kd8_B_e1200_cmT
python predict.py --run kd10_B_e1200_cmT_s3 --shifts 9
python predict.py --run kd10_B_e1200_cmT_s3 --shifts 5
python predict.py --run kd8_B_e1200_cmT --shifts 9
python test_time_checks.py --run kd10_B_e1200_cmT_s3
