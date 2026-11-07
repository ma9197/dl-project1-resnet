# ResNet under 5M parameters on CIFAR-10

NYU Deep Learning, Fall 2026, Project 1 (Kaggle: *NYU Tandon Deep Learning F26*).
Mirsaid Abbasov (ma9197). Solo project.

A residual network trained **from scratch** on CIFAR-10 with **4,781,374 trainable parameters** (limit: 5,000,000).

| | |
|---|---|
| CIFAR-10 test accuracy (10,000 images) | **97.82%** (97.89% with horizontal-flip test-time augmentation) |
| Parameters | 4,781,374 |
| Kaggle public leaderboard | **0.907** (final model `kd10_B_e1200_cmT_s3`) |
| Final leaderboard (hidden half) | **0.874**, rank 28 of 81 teams (kd8, the better of the two final picks there; kd10 scored 0.871) |
| Training time | about 5 hours on one A100 GPU, plus the teachers |

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
- Weight decay 5e-4 (not on BN or bias), label smoothing 0.1, bfloat16 mixed precision
- Augmentation: random crop (4-pixel padding), horizontal flip, Cutout (16x16), TrivialAugmentWide, CutMix on every batch
- Knowledge distillation for 1200 epochs from 4 larger teachers, which are never submitted: loss = 0.1 CE + 0.9 KL
  to the average teacher softmax at temperature 1, with the teachers seeing the same augmented, mixed images
- Final model trained on all 50,000 training images; the CIFAR-10 test set was used to report and to compare final
  candidates, never to train
- Test time: average of the softmax for the image and its horizontal flip

## Experiments

Teachers (training only). All are trained on all 50,000 images.

| Name | Channels / blocks per layer | Params | Epochs | Test acc | + flip TTA |
|---|---|---|---|---|---|
| final_B_base_e200, final_B_ta_e200, teacher_s1, teacher_s2 | 84-168-336 / 2-2-2 | 4.78M | 200 | 96.52 to 96.82% | 96.83 to 97.14% |
| wide_t1, wide_t2 | 128-256-512 / 2-2-2 | 11.09M | 300 | 97.45, 97.41% | 97.66, 97.61% |
| big_t1, big_t2 | 160-320-640 / 4-4-4 | 36.69M | 300 | 97.66, 97.67% | 97.86, 97.92% |
| big_t3, big_t4 (with CutMix) | 160-320-640 / 4-4-4 | 36.69M | 400 | 97.66, 97.85% | 97.83, 97.99% |

Submitted-size models, all layout 84-168-336 / 2-2-2 (4,781,374 parameters) except the pilots.

| Run | Teachers | CutMix | Epochs | Test acc | + flip TTA | Kaggle public |
|---|---|---|---|---|---|---|
| pilot_A (64-128-256 / 3-3-3, 4.33M) | none | 0 | 15 | 91.68% | 92.21% | |
| pilot_B | none | 0 | 15 | 92.39% | 92.59% | |
| pilot_C (64-128-256-512 / 1-1-1-1, 4.90M) | none | 0 | 15 | 89.98% | 90.34% | |
| pilot_D (64-128-256-320 / 2-2-2-1, 4.52M) | none | 0 | 15 | 91.48% | 91.86% | |
| e50_B_base | none | 0 | 50 | 95.41% | 95.68% | |
| e50_B_ta | none | 0 | 50 | 94.80% | 95.18% | |
| final_B_ta_e200 | none | 0 | 200 | 96.73% | 97.00% | 0.884 |
| kd_B_e300 | 4 B | 0 | 300 | 97.10% | 97.27% | 0.885 |
| kd2_B_e500 | 2 wide + 2 B | 0 | 500 | 97.51% | 97.58% | |
| kd3_B_e800 | 2 wide + 2 B | 0 | 800 | 97.57% | 97.69% | 0.887 |
| kd4_B_e800 | 2 big + 2 wide | 0 | 800 | 97.52% | 97.68% | |
| kd5_B_e800_cm | 2 big + 2 wide | 0.5 | 800 | 97.61% | 97.73% | 0.895 |
| kd7_B_e1200_cm05_s1 | 2 big + 2 wide | 0.5 | 1200 | 97.76% | 97.86% | 0.890 |
| kd6_B_e1200_cm1 | 2 big + 2 wide | 1 | 1200 | 97.69% | 97.77% | 0.903 |
| kd8_B_e1200_cmT | big_t3, big_t4, 2 wide | 1 | 1200 | 97.67% | 97.72% | 0.907 |
| **kd10_B_e1200_cmT_s3** | big_t3, big_t4, 2 wide | 1 | 1200 | **97.82%** | **97.89%** | **0.907** |

Pilots and 50-epoch runs hold out the last 5,000 training images for validation; all others use all 50,000.

Test-time ideas I tried and did not use (`predict.py --shifts`, `test_time_checks.py`; the professor allowed BN
adaptation and class balancing if disclosed): averaging 1-pixel shifts scored 0.902 to 0.903 on Kaggle, class balancing
scored 0.895, and BN statistics from the test images lowered CIFAR-10 accuracy on 2,000-image subsets, so I did not upload it.

## Reproduce

```bash
pip install -r requirements.txt
python -c "from torchvision import datasets; datasets.CIFAR10('data', train=True, download=True); datasets.CIFAR10('data', train=False, download=True)"
# the final model is included, so this works without retraining:
python predict.py --run kd10_B_e1200_cmT_s3          # -> submissions/kd10_B_e1200_cmT_s3.csv
jupyter nbconvert --to notebook --execute results.ipynb
# retrain the teachers and the final model (one GPU per command; about 8 hours on an A100):
T="--full-train --trivialaug --teacher-only"
python train.py --name wide_t1 --channels 128 256 512 --blocks 2 2 2 --epochs 300 --seed 11 $T
python train.py --name wide_t2 --channels 128 256 512 --blocks 2 2 2 --epochs 300 --seed 12 $T
python train.py --name big_t3 --channels 160 320 640 --blocks 4 4 4 --epochs 400 --cutmix 1.0 --seed 23 $T
python train.py --name big_t4 --channels 160 320 640 --blocks 4 4 4 --epochs 400 --cutmix 1.0 --seed 24 $T
python train.py --name kd10_B_e1200_cmT_s3 --channels 84 168 336 --blocks 2 2 2 --full-train --trivialaug --epochs 1200 \
    --kd-T 1 --kd-alpha 0.1 --cutmix 1.0 --teachers big_t3 big_t4 wide_t1 wide_t2 --seed 3
```

`experiments.sh` and `hpc/` have the command for every run in the tables. The two final picks are included at
`runs/kd10_B_e1200_cmT_s3/model.pt` and `runs/kd8_B_e1200_cmT/model.pt`. Teacher weights are too large for GitHub.

Kaggle data: the 2,000 evaluation images go in `kaggle_data/test_images/` with `kaggle_data/test.csv`
(from the competition page; `download_kaggle.py` fetches them file by file with the Kaggle API).
The competition's `train_images` are identical to the standard CIFAR-10 training set (same order, pixels and
labels), so training uses the torchvision copy.

## Files

| File | Purpose |
|---|---|
| `train.py` | model, GPU data augmentation, training loop, evaluation |
| `predict.py` | Kaggle submission (`id,label`) with flip test-time augmentation |
| `test_time_checks.py` | test-time ideas I tried and did not use |
| `make_figures.py` | rebuilds the distillation figure from the logs |
| `make_npz.py` | builds the CIFAR-10 arrays from the Hugging Face copy, for machines where the Toronto download is blocked |
| `hpc/` | setup and job scripts for NYU HPC |
| `report/` | report source and PDF |
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
- G. Hinton, O. Vinyals, J. Dean. Distilling the Knowledge in a Neural Network. arXiv:1503.02531, 2015.
- L. Beyer et al. Knowledge Distillation: A Good Teacher Is Patient and Consistent. CVPR 2022.
- S. Yun et al. CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features. ICCV 2019.
