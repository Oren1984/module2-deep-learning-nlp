# 01_classic_dl_mlp/src/stage2_quick_eda.py
# This script performs quick exploratory data analysis (EDA) on the Fashion-MNIST dataset.
# It visualizes sample images and the class distribution.

import os
import torch
import matplotlib.pyplot as plt
from collections import Counter
from stage1_data_load import get_dataloaders, CLASSES

# Function to denormalize images for visualization
def denorm(x):
    # inverse of Normalize((0.5,), (0.5,)) => x*0.5 + 0.5
    return (x * 0.5) + 0.5

# Main function to perform quick EDA
def main():
    os.makedirs("outputs/figures", exist_ok=True)
    train_loader, _, _ = get_dataloaders(batch_size=64)

    images, labels = next(iter(train_loader))

    # Grid preview
    fig = plt.figure()
    for i in range(16):
        ax = fig.add_subplot(4, 4, i + 1)
        img = denorm(images[i]).squeeze(0)
        ax.imshow(img, cmap="gray")
        ax.set_title(CLASSES[labels[i].item()])
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("outputs/figures/sample_grid.png", dpi=150)
    plt.close()

    # Class distribution (approx from a few batches to keep it fast)
    counts = Counter()
    for b_idx, (_, y) in enumerate(train_loader):
        counts.update(y.tolist())
        if b_idx >= 50:
            break
        
    # Prepare data for plotting
    xs = list(range(len(CLASSES)))
    ys = [counts.get(i, 0) for i in xs]
    
    # Plot class distribution
    plt.figure()
    plt.bar(xs, ys)
    plt.xticks(xs, CLASSES, rotation=45, ha="right")
    plt.title("Approx Class Distribution (first ~51 batches)")
    plt.tight_layout()
    plt.savefig("outputs/figures/class_distribution.png", dpi=150)
    plt.close()

    print("✅ STAGE 2 — Quick EDA")
    print("Saved:")
    print("- outputs/figures/sample_grid.png")
    print("- outputs/figures/class_distribution.png")

if __name__ == "__main__":
    main()
