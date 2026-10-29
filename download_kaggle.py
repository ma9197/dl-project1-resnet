"""Download the competition files one by one (Kaggle's 'download all' endpoint returns 404).

Needs a Kaggle API token in ~/.kaggle/access_token. Downloads:
  kaggle_data/test.csv, sample_submission.csv, train_labels.csv
  kaggle_data/test_images/*.png        (all 2,000 evaluation images)
  kaggle_data/train_images/*.png       (a random sample of 200, only to verify they equal standard CIFAR-10)
"""

import csv, os, random, time
from kaggle.api.kaggle_api_extended import KaggleApi

COMP = "cifar-10-res-net-architecture-challenge-dl"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kaggle_data")
api = KaggleApi(); api.authenticate()

for f in ["test.csv", "sample_submission.csv", "train_labels.csv"]:
    if not os.path.exists(os.path.join(OUT, f)):
        api.competition_download_file(COMP, f, path=OUT, quiet=True)

test = [r["filename"] for r in csv.DictReader(open(os.path.join(OUT, "test.csv")))]
train = [r["filename"] for r in csv.DictReader(open(os.path.join(OUT, "train_labels.csv")))]
wanted = [f"test_images/{n}" for n in test] + [f"train_images/{n}" for n in random.Random(0).sample(train, 200)]
todo = [f for f in wanted if not os.path.exists(os.path.join(OUT, f))]
print(f"{len(todo)} files to download", flush=True)

def get(f):
    """One file at a time; on HTTP 429 (rate limit) wait and retry with growing delays."""
    wait = 30
    for attempt in range(12):
        try:
            api.competition_download_file(COMP, f, path=OUT, quiet=True)
            time.sleep(0.5)
            return None
        except Exception as e:
            err = str(e)[:100]
            if "429" in err:
                print(f"rate limited at {f}, waiting {wait}s", flush=True)
                time.sleep(wait); wait = min(wait * 2, 600)
            else:
                time.sleep(2)
    return f"{f}: {err}"

errors = []
for k, f in enumerate(todo, 1):
    e = get(f)
    if e: errors.append(e)
    if k % 100 == 0: print(f"{k}/{len(todo)}", flush=True)
n_test = len(os.listdir(os.path.join(OUT, "test_images")))
print(f"done | test images on disk: {n_test}/{len(test)} | errors: {len(errors)}", flush=True)
for e in errors[:10]: print("ERR", e)
