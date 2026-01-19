# 01_classic_dl_mlp/src/stage3_preprocess.py
# This script preprocesses the Fashion-MNIST dataset by flattening the images
# from 2D (28x28) to 1D (784) tensors suitable for MLP input.

import torch
from stage1_data_load import get_dataloaders

# Function to flatten batch of images
def flatten_batch(x):
    return x.view(x.size(0), -1)  # (B, 784)

# Main function to demonstrate preprocessing
def main():
    train_loader, _, _ = get_dataloaders()
    x, y = next(iter(train_loader))
    xf = flatten_batch(x)

    print("✅ STAGE 3 — Preprocess")
    print("Before:", x.shape, "After flatten:", xf.shape)
    print("dtype:", xf.dtype, "min/max:", float(xf.min()), float(xf.max()))

if __name__ == "__main__":
    main()
