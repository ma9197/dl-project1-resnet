"""Make a Kaggle submission from a trained run.

  python predict.py --run final            -> submissions/final.csv  (columns: id,label)

Reads kaggle_data/test.csv (id, filename) and kaggle_data/test_images/*.png, applies the same
normalization as training, predicts with horizontal-flip test-time augmentation, and prints the
parameter count (must be <= 5M).

  python predict.py --run final --shifts 9  -> submissions/final_shift9.csv  (also averages 1-pixel shifts)
"""

import argparse, csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F

from train import ResNet, count_params, normalize

HERE = Path(__file__).resolve().parent
SHIFTS = {1: [(0, 0)],
          5: [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)],
          9: [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--shifts", type=int, default=1, choices=sorted(SHIFTS), help="number of 1-pixel shifts to average")
    a = ap.parse_args()

    ck = torch.load(HERE / "runs" / a.run / "model.pt", map_location="cuda", weights_only=False)
    args = ck["args"]
    model = ResNet(args["channels"], args["blocks"], args["kernel"], args["skip_kernel"]).cuda()
    model.load_state_dict(ck["model"]); model.eval()
    n = count_params(model)
    assert n <= 5_000_000
    print(f"model {a.run}: {n:,} trainable parameters")

    rows = list(csv.DictReader(open(HERE / "kaggle_data" / "test.csv")))
    x = np.stack([np.array(Image.open(HERE / "kaggle_data" / "test_images" / r["filename"]).convert("RGB"))
                  for r in rows])
    x = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous().cuda()
    preds = []
    with torch.no_grad():
        for k in range(0, len(x), 500):
            pad = F.pad(normalize(x[k:k + 500]), (4, 4, 4, 4))  # zero = mean pixel, as in training
            out = 0
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for dx, dy in SHIFTS[a.shifts]:
                    xb = pad[:, :, 4 + dy:36 + dy, 4 + dx:36 + dx]
                    out = out + model(xb).float().softmax(1)
                    if not a.no_tta:
                        out = out + model(xb.flip(3)).float().softmax(1)
            preds.append(out.argmax(1).cpu())
    preds = torch.cat(preds).tolist()

    out = HERE / "submissions" / f"{a.run}{'_shift%d' % a.shifts if a.shifts > 1 else ''}.csv"; out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "label"])
        for r, p in zip(rows, preds): w.writerow([r["id"], p])
    counts = np.bincount(preds, minlength=10)
    print(f"wrote {out} | {len(preds)} rows | predicted class counts: {counts.tolist()}")


if __name__ == "__main__":
    main()
