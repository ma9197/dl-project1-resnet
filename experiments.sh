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
