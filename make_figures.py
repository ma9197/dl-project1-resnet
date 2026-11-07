"""Rebuild figures/kd_curves.pdf and .png from the training logs in runs/."""

import json

import matplotlib.pyplot as plt
import numpy as np

RUNS = [("final_B_ta_e200", "No distillation", "#2a78d6"),
        ("kd3_B_e800", "2 wide + 2 B teachers", "#eb6834"),
        ("kd6_B_e1200_cm1", "2 big + 2 wide, CutMix", "#1baf7a"),
        ("kd10_B_e1200_cmT_s3", "Final: CutMix teachers too", "#eda100")]

fig, ax = plt.subplots(figsize=(3.3, 2.5))
for name, label, color in RUNS:
    acc = 100 * np.array([e["val_acc"] for e in json.load(open(f"runs/{name}/log.json"))])
    w = max(1, len(acc) // 100)  # mean over 1% of the schedule, so runs of different length look alike
    smooth = np.convolve(acc, np.ones(w) / w, mode="valid")
    ax.plot((np.arange(len(smooth)) + w) / len(acc), smooth, color=color, lw=1.5, label=label)
ax.set(xlim=(0.5, 1), ylim=(90, 98.2), xlabel="Fraction of schedule", ylabel="CIFAR-10 test accuracy (%)")
ax.grid(axis="y", color="#e5e4df", lw=0.6)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(labelsize=8); ax.xaxis.label.set_size(8); ax.yaxis.label.set_size(8)
ax.legend(fontsize=7, frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig("figures/kd_curves.pdf"); fig.savefig("figures/kd_curves.png", dpi=200)
