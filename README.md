# ResNet under 5M parameters on CIFAR-10

NYU Deep Learning, Fall 2026, Project 1 (Kaggle: *NYU Tandon Deep Learning F26*).
Mirsaid Abbasov (ma9197). Solo project.

A residual network trained **from scratch** on CIFAR-10 with **4,781,374 trainable parameters** (limit: 5,000,000).

| | |
|---|---|
| CIFAR-10 test accuracy (10,000 images) | **96.73%** (97.00% with horizontal-flip test-time augmentation) |
| Parameters | 4,781,374 |
| Training time | 32 min, one RTX 4090 Laptop GPU |
| Kaggle public leaderboard | to be added |

`results.ipynb` shows everything executed: the model summary and parameter count, the final test accuracy,
training curves, all experiments, per-class accuracy and the Kaggle submission.

## Model

Residual block, as in the project specification: `out = ReLU(S(x) + F(x))`, where
`F = conv3x3 -> BN -> ReLU -> conv3x3 -> BN` and `S` is the identity, or a 1x1 convolution + BN when the
shape changes.

| Part | Output size | Details |
|---|---|---|
| Stem | 84 x 32 x 32 | conv3x3, BN, ReLU |
| Residual layer 1 | 84 x 32 x 32 | 2 blocks |
| Residual layer 2 | 168 x 16 x 16 | 2 blocks, first has stride 2 |
| Residual layer 3 | 336 x 8 x 8 | 2 blocks, first has stride 2 |
| Head | 10 | global average pool (8x8), fully connected |

In the project's notation: N = 3 residual layers, B = (2, 2, 2), C = (84, 168, 336), F = 3, K = 1, P = 8.
Using 3 wide layers instead of ResNet-18's 4 narrow ones keeps the parameter count under 5M while keeping
more channels at high resolution (see the pilot experiments below).

## Training

- SGD, Nesterov momentum 0.9, batch 128, learning rate 0.1 with 2 warmup epochs then cosine decay to 0
- Weight decay 5e-4 (not on BN or bias), label smoothing 0.1, 200 epochs, bfloat16 mixed precision
- Augmentation: random crop (4-pixel padding), horizontal flip, Cutout (16x16), plus TrivialAugmentWide
- Final model trained on all 50,000 training images; the CIFAR-10 test set was used only for reporting
- Test time: average of the softmax for the image and its horizontal flip

## Experiments

| Run | Layout (channels / blocks per layer) | Params | Epochs | TrivialAugment | Test acc | + flip TTA |
|---|---|---|---|---|---|---|
| pilot_A | 64-128-256 / 3-3-3 | 4.33M | 15 | no | 91.68% | 92.21% |
| **pilot_B** | **84-168-336 / 2-2-2** | **4.78M** | 15 | no | **92.39%** | **92.59%** |
| pilot_C | 64-128-256-512 / 1-1-1-1 | 4.90M | 15 | no | 89.98% | 90.34% |
| pilot_D | 64-128-256-320 / 2-2-2-1 | 4.52M | 15 | no | 91.48% | 91.86% |
| e50_B_base | 84-168-336 / 2-2-2 | 4.78M | 50 | no | 95.41% | 95.68% |
| e50_B_ta | 84-168-336 / 2-2-2 | 4.78M | 50 | yes | 94.80% | 95.18% |
| final_B_base_e200 | 84-168-336 / 2-2-2 | 4.78M | 200 | no | 96.52% | 96.83% |
| **final_B_ta_e200** | 84-168-336 / 2-2-2 | 4.78M | 200 | yes | **96.73%** | **97.00%** |

Pilots and 50-epoch runs hold out the last 5,000 training images for validation; the 200-epoch runs use all
50,000. Test accuracy is always on the standard CIFAR-10 test set. TrivialAugment hurts at 50 epochs but
helps at 200: the stronger augmentation needs a longer schedule.

## Reproduce

```bash
pip install -r requirements.txt
python -c "from torchvision import datasets; datasets.CIFAR10('data', train=True, download=True); datasets.CIFAR10('data', train=False, download=True)"
python train.py --name final_B_ta_e200 --channels 84 168 336 --blocks 2 2 2 --epochs 200 --full-train --trivialaug
python predict.py --run final_B_ta_e200          # -> submissions/final_B_ta_e200.csv
jupyter nbconvert --to notebook --execute results.ipynb
```

`experiments.sh` lists the command for every run in the table. The trained final model is included at
`runs/final_B_ta_e200/model.pt`, so `predict.py` and the notebook work without retraining.

Kaggle data: the 2,000 evaluation images go in `kaggle_data/test_images/` with `kaggle_data/test.csv`
(from the competition page; `download_kaggle.py` fetches them file by file with the Kaggle API).
The competition's `train_images` are identical to the standard CIFAR-10 training set (same order, pixels and
labels), so training uses the torchvision copy.

## Files

| File | Purpose |
|---|---|
| `train.py` | model, GPU data augmentation, training loop, evaluation |
| `predict.py` | Kaggle submission (`id,label`) with flip test-time augmentation |
| `results.ipynb` | executed notebook with all results and plots |
| `experiments.sh` | commands for every experiment |
| `download_kaggle.py` | downloads the competition files |
| `runs/<name>/` | per-run training log, summary and (final model only) weights |
| `figures/` | plots used in the report |

## References

- K. He, X. Zhang, S. Ren, J. Sun. Deep Residual Learning for Image Recognition. CVPR 2016.
- T. DeVries, G. Taylor. Improved Regularization of Convolutional Neural Networks with Cutout. arXiv:1708.04552, 2017.
- S. Müller, F. Hutter. TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation. ICCV 2021.
- C. Szegedy et al. Rethinking the Inception Architecture for Computer Vision (label smoothing). CVPR 2016.
- I. Loshchilov, F. Hutter. SGDR: Stochastic Gradient Descent with Warm Restarts (cosine schedule). ICLR 2017.
