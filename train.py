"""Project 1: modified ResNet (<= 5M parameters) on CIFAR-10, trained from scratch.

Residual block (as in the project spec):  out = ReLU( S(x) + F(x) ),
  F = conv3x3 -> BN -> ReLU -> conv3x3 -> BN,   S = identity or 1x1 conv + BN (when shape changes).

Usage examples:
  python train.py --name pilot_A --channels 64 128 256 --blocks 3 3 3 --epochs 15
  python train.py --name final --channels 84 168 336 --blocks 2 2 2 --epochs 200 --full-train

Data is kept on the GPU; augmentation (random crop with 4px padding, horizontal flip, Cutout)
runs on the GPU per batch, so training is not limited by the CPU.
Outputs go to runs/<name>/: log.json (per-epoch metrics), model.pt, summary.json.
"""

import argparse, json, math, os, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

HERE = Path(__file__).resolve().parent
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


# ----------------------------------------------------------------------------- model
class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride, kernel=3, skip_kernel=1):
        super().__init__()
        p = kernel // 2
        self.conv1 = nn.Conv2d(cin, cout, kernel, stride, p, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, kernel, 1, p, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.skip = nn.Identity()
        if stride != 1 or cin != cout:
            self.skip = nn.Sequential(nn.Conv2d(cin, cout, skip_kernel, stride, skip_kernel // 2, bias=False),
                                      nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.skip(x))


class ResNet(nn.Module):
    """N residual layers; layer i has B_i blocks with C_i channels. First layer keeps 32x32, others stride 2."""

    def __init__(self, channels=(64, 128, 256), blocks=(2, 2, 2), kernel=3, skip_kernel=1, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, channels[0], 3, 1, 1, bias=False), nn.BatchNorm2d(channels[0]),
                                  nn.ReLU(inplace=True))
        layers, cin = [], channels[0]
        for i, (c, b) in enumerate(zip(channels, blocks)):
            for j in range(b):
                layers.append(BasicBlock(cin, c, 2 if (j == 0 and i > 0) else 1, kernel, skip_kernel))
                cin = c
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(cin, num_classes)

    def forward(self, x):
        x = self.layers(self.stem(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)   # global average pool (P = final feature-map size)
        return self.fc(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ----------------------------------------------------------------------------- data
def load_cifar(device):
    tr = datasets.CIFAR10(HERE / "data", train=True, download=False)
    te = datasets.CIFAR10(HERE / "data", train=False, download=False)
    f = lambda a: torch.from_numpy(a).permute(0, 3, 1, 2).contiguous()
    return (f(tr.data).to(device), torch.tensor(tr.targets).to(device),
            f(te.data).to(device), torch.tensor(te.targets).to(device))


def normalize(x_uint8):
    x = x_uint8.float().div_(255)
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def augment(x, cutout=16):
    """x: normalized float batch (B,3,32,32). Random crop (pad 4), horizontal flip, Cutout."""
    B = x.shape[0]
    xp = F.pad(x, (4, 4, 4, 4))  # zero pad == mean pixel after normalization
    i = torch.randint(0, 9, (B,), device=x.device)
    j = torch.randint(0, 9, (B,), device=x.device)
    rows = (i[:, None] + torch.arange(32, device=x.device)[None])           # B,32
    cols = (j[:, None] + torch.arange(32, device=x.device)[None])
    x = xp[torch.arange(B, device=x.device)[:, None, None, None], torch.arange(3, device=x.device)[None, :, None, None],
           rows[:, None, :, None], cols[:, None, None, :]]
    flip = torch.rand(B, device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    if cutout:
        cy = torch.randint(0, 32, (B,), device=x.device); cx = torch.randint(0, 32, (B,), device=x.device)
        yy = torch.arange(32, device=x.device)[None, :, None]; xx = torch.arange(32, device=x.device)[None, None, :]
        mask = ((yy - cy[:, None, None]).abs() < cutout // 2) & ((xx - cx[:, None, None]).abs() < cutout // 2)
        x = x.masked_fill(mask[:, None], 0.0)
    return x


class TrivialAugDS(torch.utils.data.Dataset):
    """CPU-side TrivialAugmentWide on uint8 images (one random op of random strength per image)."""

    def __init__(self, x_uint8):
        from torchvision.transforms import v2
        self.x, self.t = x_uint8, v2.TrivialAugmentWide()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.t(self.x[i]), i


# ----------------------------------------------------------------------------- train / eval
@torch.no_grad()
def evaluate(model, x, y, tta=False, bs=1000):
    model.eval()
    correct, loss = 0, 0.0
    for k in range(0, len(x), bs):
        xb = normalize(x[k:k + bs]).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(xb).float()
            if tta:
                out = out + model(xb.flip(3)).float()
        loss += F.cross_entropy(out, y[k:k + bs], reduction="sum").item()
        correct += (out.argmax(1) == y[k:k + bs]).sum().item()
    return loss / len(x), correct / len(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--channels", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--blocks", type=int, nargs="+", default=[2, 2, 2])
    ap.add_argument("--kernel", type=int, default=3)
    ap.add_argument("--skip-kernel", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--smoothing", type=float, default=0.1)
    ap.add_argument("--cutout", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=2, help="warmup epochs")
    ap.add_argument("--trivialaug", action="store_true", help="add TrivialAugmentWide (CPU workers)")
    ap.add_argument("--full-train", action="store_true", help="train on all 50k (no held-out validation split)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = True
    dev = "cuda"
    out_dir = HERE / "runs" / a.name; out_dir.mkdir(parents=True, exist_ok=True)

    xtr, ytr, xte, yte = load_cifar(dev)
    # held-out validation: last 5k of the training set (unless --full-train); CIFAR-10 test set reported separately
    if a.full_train:
        xva, yva = xte, yte
    else:
        xva, yva = xtr[45000:], ytr[45000:]
        xtr, ytr = xtr[:45000], ytr[:45000]

    model = ResNet(a.channels, a.blocks, a.kernel, a.skip_kernel).to(dev).to(memory_format=torch.channels_last)
    n_params = count_params(model)
    assert n_params <= 5_000_000, f"{n_params:,} parameters > 5M"
    print(f"{a.name}: channels {a.channels} blocks {a.blocks} | {n_params:,} params | train {len(xtr)}", flush=True)

    decay, no_decay = [], []
    for n_, p in model.named_parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)
    opt = torch.optim.SGD([{"params": decay, "weight_decay": a.wd}, {"params": no_decay, "weight_decay": 0.0}],
                          lr=a.lr, momentum=0.9, nesterov=True)
    steps_per_epoch = math.ceil(len(xtr) / a.bs)
    total, warm = a.epochs * steps_per_epoch, a.warmup * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (total - warm))))

    loader = None
    if a.trivialaug:
        loader = torch.utils.data.DataLoader(TrivialAugDS(xtr.cpu()), batch_size=a.bs, shuffle=True, num_workers=8,
                                             persistent_workers=True, pin_memory=True, drop_last=False)

    def batches():
        if loader is not None:
            for xb_u8, idx in loader:
                yield xb_u8.to(dev, non_blocking=True), ytr[idx.to(dev)]
        else:
            perm = torch.randperm(len(xtr), device=dev)
            for k in range(0, len(xtr), a.bs):
                idx = perm[k:k + a.bs]
                yield xtr[idx], ytr[idx]

    log, t0 = [], time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        tl, tc = torch.zeros((), device=dev), torch.zeros((), device=dev)
        for xb_u8, yb in batches():
            xb = augment(normalize(xb_u8), a.cutout).contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(xb)
                loss = F.cross_entropy(out.float(), yb, label_smoothing=a.smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
            tl += loss.detach() * len(yb); tc += (out.argmax(1) == yb).sum()
        vl, va = evaluate(model, xva, yva)
        rec = {"epoch": ep, "train_loss": tl.item() / len(xtr), "train_acc": tc.item() / len(xtr),
               "val_loss": vl, "val_acc": va, "lr": sched.get_last_lr()[0], "time": time.time() - t0}
        log.append(rec)
        print(f"ep {ep:3d} train {rec['train_loss']:.3f}/{rec['train_acc']:.4f} val {vl:.3f}/{va:.4f} "
              f"| {rec['time']:.0f}s", flush=True)
        (out_dir / "log.json").write_text(json.dumps(log, indent=1))

    te_loss, te_acc = evaluate(model, xte, yte)
    _, te_acc_tta = evaluate(model, xte, yte, tta=True)
    torch.save({"model": model.state_dict(), "args": vars(a), "params": n_params}, out_dir / "model.pt")
    summary = {"name": a.name, "params": n_params, "channels": a.channels, "blocks": a.blocks,
               "epochs": a.epochs, "full_train": a.full_train, "val_acc": log[-1]["val_acc"],
               "cifar10_test_acc": te_acc, "cifar10_test_acc_flipTTA": te_acc_tta,
               "minutes": (time.time() - t0) / 60, "args": vars(a)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"FINAL {a.name}: params {n_params:,} | CIFAR-10 test acc {te_acc:.4f} (flip TTA {te_acc_tta:.4f}) "
          f"| {summary['minutes']:.1f} min", flush=True)


if __name__ == "__main__":
    main()
