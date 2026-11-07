"""Test-time ideas I tried on the final model and did not use: BN statistics from the test images, and class balancing.

  python test_time_checks.py --run kd10_B_e1200_cmT_s3

Prints the effect on random 2,000-image CIFAR-10 test subsets (the size of the Kaggle set), and writes
submissions/<run>_bal.csv. No training or fine-tuning, and no labels from the Kaggle set are used.
"""

import argparse, copy, csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from train import load_cifar, load_run, normalize

HERE = Path(__file__).resolve().parent
BN = torch.nn.BatchNorm2d


def probs(model, x):
    """Softmax averaged over each image and its horizontal flip."""
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for k in range(0, len(x), 500):
            b = normalize(x[k:k + 500])
            out.append((model(b).float().softmax(1) + model(b.flip(3)).float().softmax(1)) / 2)
    return torch.cat(out).double().cpu().numpy()


def adapt_bn(model, x, w):
    """Copy of the model whose BN statistics are (1 - w) * training + w * statistics of the images x."""
    m = copy.deepcopy(model)
    src = {n: (b.running_mean.clone(), b.running_var.clone()) for n, b in m.named_modules() if isinstance(b, BN)}
    for b in m.modules():
        if isinstance(b, BN):
            b.reset_running_stats(); b.momentum = None; b.train()  # momentum None: plain average over batches
    with torch.no_grad():
        for k in range(0, len(x), 500):
            b = normalize(x[k:k + 500]); m(b); m(b.flip(3))
    for n, b in m.named_modules():
        if isinstance(b, BN):
            b.running_mean.mul_(w).add_((1 - w) * src[n][0]); b.running_var.mul_(w).add_((1 - w) * src[n][1])
    return m.eval()


def balance(p, tol=2, steps=5000, lr=0.3):
    """Add one offset per class to the log-probabilities until each class is predicted about equally often."""
    lp = np.log(p + 1e-12); b = np.zeros(10); t = len(p) / 10; best, best_dev = None, 1e9
    for i in range(steps):
        pred = (lp + b).argmax(1); n = np.bincount(pred, minlength=10); dev = abs(n - t).max()
        if dev < best_dev: best, best_dev = pred, dev
        if dev <= tol: break
        b -= lr * (n - t) / t / (1 + i / 200)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    model = load_run(a.run)
    _, _, xte, yte = load_cifar("cuda")
    y = yte.cpu().numpy()
    rows = list(csv.DictReader(open(HERE / "kaggle_data" / "test.csv")))
    k = np.stack([np.array(Image.open(HERE / "kaggle_data" / "test_images" / r["filename"]).convert("RGB")) for r in rows])
    k = torch.from_numpy(k).permute(0, 3, 1, 2).contiguous().cuda()
    p_k = probs(model, k)

    rng = np.random.default_rng(0)
    natural = [rng.choice(len(y), 2000, replace=False) for _ in range(10)]
    even = [np.concatenate([rng.choice(np.where(y == c)[0], 200, replace=False) for c in range(10)]) for _ in range(10)]
    acc = lambda pred, i: 100 * (pred == y[i]).mean()
    sub = lambda i: xte[torch.as_tensor(i).cuda()]

    plain = np.mean([acc(probs(model, sub(i)).argmax(1), i) for i in natural[:5]])
    print(f"plain: {plain:.2f}% on 5 random subsets")
    for w in (0.2, 0.35, 0.5, 1.0):
        adapted = np.mean([acc(probs(adapt_bn(model, sub(i), w), sub(i)).argmax(1), i) for i in natural[:5]])
        changed = (probs(adapt_bn(model, k, w), k).argmax(1) != p_k.argmax(1)).sum()
        print(f"BN mix w={w}: {adapted:.2f}% on the same subsets, changes {changed} Kaggle predictions")
    for name, subsets in (("random", natural), ("equal-class", even)):
        p = [probs(model, sub(i)) for i in subsets]
        d = [(acc(q.argmax(1), i), acc(balance(q), i)) for q, i in zip(p, subsets)]
        print(f"balance, {name} subsets: plain {np.mean(d, 0)[0]:.2f}%, balanced {np.mean(d, 0)[1]:.2f}%")
    bal = balance(p_k)
    print(f"balance changes {(bal != p_k.argmax(1)).sum()} Kaggle predictions")
    with open(HERE / "submissions" / f"{a.run}_bal.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "label"])
        for r, q in zip(rows, bal.tolist()): w.writerow([r["id"], q])


if __name__ == "__main__":
    main()
