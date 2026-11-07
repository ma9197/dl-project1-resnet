"""Build data/cifar10.npz from the Hugging Face copy of CIFAR-10 (for machines where the Toronto download is blocked).

  python make_npz.py train.parquet test.parquet     (files from huggingface.co/datasets/uoft-cs/cifar10, plain_text/)
  python make_npz.py --check                         (prints the same checksums from the torchvision copy)

The checksums printed by both commands match, so the arrays are identical to the standard dataset.
"""

import hashlib, io, sys
import numpy as np


def digest(x, y):
    return hashlib.md5(x.tobytes()).hexdigest()[:12], hashlib.md5(np.asarray(y, np.int64).tobytes()).hexdigest()[:12]


if sys.argv[1] == "--check":
    from torchvision import datasets
    for train in (True, False):
        d = datasets.CIFAR10("data", train=train, download=False)
        print("train" if train else "test", d.data.shape, *digest(d.data, d.targets))
else:
    import pyarrow.parquet as pq
    from PIL import Image
    out = {}
    for split, path in (("tr", sys.argv[1]), ("te", sys.argv[2])):
        t = pq.read_table(path).to_pydict()
        out["x" + split] = np.stack([np.array(Image.open(io.BytesIO(im["bytes"])).convert("RGB")) for im in t["img"]])
        out["y" + split] = np.array(t["label"], np.int64)
        print("train" if split == "tr" else "test", out["x" + split].shape, *digest(out["x" + split], out["y" + split]))
    np.savez("data/cifar10.npz", **out)
