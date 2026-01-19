# 01_classic_dl_mlp/src/stage6_evaluate.py
# This script evaluates the trained MLP model on the test set of the Fashion-MNIST
# dataset. It computes test accuracy, generates a confusion matrix, and saves the
# results and figures to the outputs directory.

import os
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Import necessary modules for data loading and model definition
from stage1_data_load import get_dataloaders, get_device, CLASSES
from stage4_model_baseline import MLP

# Main evaluation function
@torch.no_grad()
def main():
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/results", exist_ok=True)
    
    # Load test data and model checkpoint
    device = get_device()
    _, _, test_loader = get_dataloaders(batch_size=256)
    
    # Load best model checkpoint
    ckpt = torch.load("outputs/models/best_model.pt", map_location=device)
    config = ckpt["config"]

    # Instantiate model and load state dict
    model = MLP(
        hidden1=config["hidden1"],
        hidden2=config["hidden2"],
        dropout=config["dropout"],
        use_bn=config["use_bn"]
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Evaluation on test set 
    correct = 0
    total = 0
    num_classes = len(CLASSES)
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    
    # Iterate over test data 
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)
        
        # Update overall correct predictions and total count
        correct += (preds == y).sum().item()
        total += y.size(0)
        
        # Update confusion matrix 
        for t, p in zip(y.view(-1), preds.view(-1)):
            cm[t.long(), p.long()] += 1

    # Compute test accuracy 
    acc = correct / total

    # Plot confusion matrix 
    plt.figure()
    plt.imshow(cm.cpu().numpy())
    plt.title("Confusion Matrix (Test)")
    plt.xticks(range(num_classes), CLASSES, rotation=45, ha="right")
    plt.yticks(range(num_classes), CLASSES)
    plt.tight_layout()
    plt.savefig("outputs/figures/confusion_matrix.png", dpi=150)
    plt.close()

    # Save evaluation results to JSON
    results = {"test_accuracy": acc, "confusion_matrix": cm.cpu().tolist(), "config": config}
    with open("outputs/results/eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("✅ STAGE 6 — Evaluation")
    print(f"Test accuracy: {acc:.4f}")
    print("Saved:")
    print("- outputs/figures/confusion_matrix.png")
    print("- outputs/results/eval_results.json")

if __name__ == "__main__":
    main()
