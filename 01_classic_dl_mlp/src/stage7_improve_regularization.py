# 01_classic_dl_mlp/src/stage7_improve_regularization.py
# This script loads the training history from JSON and plots the loss and accuracy curves.
# It saves the plots as PNG files in the outputs/figures directory.

import os
import json
import matplotlib.pyplot as plt

# Main function to plot and save curves 
def main():
    os.makedirs("outputs/figures", exist_ok=True)
    
    # Load training history from JSON file 
    with open("outputs/results/train_history.json", "r", encoding="utf-8") as f:
        payload = json.load(f)

    # Extract history 
    h = payload["history"]

    # Loss plot
    plt.figure()
    plt.plot(h["train_loss"], label="train_loss")
    plt.plot(h["val_loss"], label="val_loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/figures/loss_curves.png", dpi=150)
    plt.close()

    # Accuracy plot 
    plt.figure()
    plt.plot(h["train_acc"], label="train_acc")
    plt.plot(h["val_acc"], label="val_acc")
    plt.title("Accuracy Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/figures/acc_curves.png", dpi=150)
    plt.close()

    print("✅ STAGE 7 — Curves saved")
    print("- outputs/figures/loss_curves.png")
    print("- outputs/figures/acc_curves.png")

if __name__ == "__main__":
    main()
