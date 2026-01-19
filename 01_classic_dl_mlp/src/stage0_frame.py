# 01_classic_dl_mlp/src/stage0_frame.py
# This script sets up the project directory structure and saves project information.
# It creates necessary folders and writes a JSON file with project details.

import os
import json
import torch

# Function to create necessary directories
def ensure_dirs():
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)
    os.makedirs("outputs/results", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

# Main function to set up project frame
def main():
    ensure_dirs()
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    info = {
        "project": "01_classic_dl_mlp",
        "dataset": "Fashion-MNIST",
        "task": "Multi-class classification (10 classes)",
        "input": "28x28 grayscale images",
        "model": "MLP",
        "device": device,
    }
    
    # Save project information to JSON file
    with open("outputs/results/project_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print("✅ STAGE 0 — Frame")
    for k, v in info.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
