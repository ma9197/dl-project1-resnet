"""Make a Kaggle submission from a trained run.

  python predict.py --run final            -> submissions/final.csv  (columns: id,label)

Reads kaggle_data/test.csv (id, filename) and kaggle_data/test_images/*.png, applies the same
normalization as training, predicts with horizontal-flip test-time augmentation, and prints the
parameter count (must be <= 5M).
"""

import argparse, csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from train import ResNet, count_params, normalize

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--no-tta", action="store_true")
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
            xb = normalize(x[k:k + 500])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(xb).float().softmax(1)
                if not a.no_tta:
                    out = out + model(xb.flip(3)).float().softmax(1)
            preds.append(out.argmax(1).cpu())
    preds = torch.cat(preds).tolist()

    out = HERE / "submissions" / f"{a.run}.csv"; out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "label"])
        for r, p in zip(rows, preds): w.writerow([r["id"], p])
    counts = np.bincount(preds, minlength=10)
    print(f"wrote {out} | {len(preds)} rows | predicted class counts: {counts.tolist()}")


if __name__ == "__main__":
    main()
