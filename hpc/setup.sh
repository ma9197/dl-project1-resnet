#!/bin/bash
# One-time setup on NYU HPC (Cloud Bursting): Python venv with PyTorch, then CIFAR-10 as data/cifar10.npz.
# Run from a Jupyter terminal (OnDemand, partition "interactive") with this repo at /scratch/$USER/project1.
set -e
cd /scratch/$USER
python3 -m venv venv
venv/bin/pip install -q --upgrade pip
venv/bin/pip install -q torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
venv/bin/pip install -q numpy pillow pyarrow
cd project1
# the Toronto server stalls downloads from the cluster, so use the Hugging Face copy (same images and labels)
H=https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text
mkdir -p data
curl -sL -o data/train.parquet $H/train-00000-of-00001.parquet
curl -sL -o data/test.parquet $H/test-00000-of-00001.parquet
../venv/bin/python make_npz.py data/train.parquet data/test.parquet
rm data/*.parquet
