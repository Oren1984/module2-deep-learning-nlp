# 01_classic_dl_mlp/src/stage9_report.py
# This script generates a summary report of the project including configuration,
# results, and saves it as both a text file and a markdown report.

import os
import json
from datetime import datetime

# Main function to generate report
def main():
    os.makedirs("outputs/reports", exist_ok=True)
    os.makedirs("outputs/results", exist_ok=True)

    # Load project info, training history, and evaluation results
    with open("outputs/results/project_info.json", "r", encoding="utf-8") as f:
        info = json.load(f)
    
    with open("outputs/results/train_history.json", "r", encoding="utf-8") as f:
        train_payload = json.load(f)
    
    with open("outputs/results/eval_results.json", "r", encoding="utf-8") as f:
        eval_payload = json.load(f)

    # Extract relevant data for report
    config = train_payload["config"]
    h = train_payload["history"]
    best_val_acc = max(h["val_acc"]) if h["val_acc"] else None
    test_acc = eval_payload["test_accuracy"]

    # result.txt (quick) summary 
    result_lines = [
        f"Run time: {datetime.now().isoformat()}",
        f"Dataset: {info['dataset']}",
        f"Device: {info['device']}",
        f"Model: {config['model']} | hidden1={config['hidden1']} hidden2={config['hidden2']} dropout={config['dropout']} bn={config['use_bn']}",
        f"LR: {config['lr']} | epochs(max): {config['epochs']} | early_stop_patience: {config['early_stopping_patience']}",
        f"Best Val Acc: {best_val_acc:.4f}" if best_val_acc is not None else "Best Val Acc: N/A",
        f"Test Acc: {test_acc:.4f}",
        "Saved figures:",
        "- outputs/figures/sample_grid.png",
        "- outputs/figures/class_distribution.png",
        "- outputs/figures/loss_curves.png",
        "- outputs/figures/acc_curves.png",
        "- outputs/figures/confusion_matrix.png",
    ]
    
    # Save result.txt 
    with open("outputs/results/result.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(result_lines))

    # Report.md (detailed) summary
    report = f"""# Classic Deep Learning — MLP (Fashion-MNIST)

## Goal
Train an MLP for 10-class classification on Fashion-MNIST with a clean, controlled pipeline.

## Setup
- Device: **{info['device']}**
- Model: **MLP**
- Params: hidden1={config['hidden1']}, hidden2={config['hidden2']}, dropout={config['dropout']}, batchnorm={config['use_bn']}
- Optimizer: Adam (lr={config['lr']})
- Early stopping patience: {config['early_stopping_patience']}

## Results
- Best Validation Accuracy: **{best_val_acc:.4f}**
- Test Accuracy: **{test_acc:.4f}**

## Artifacts
- Curves: `outputs/figures/loss_curves.png`, `outputs/figures/acc_curves.png`
- Confusion Matrix: `outputs/figures/confusion_matrix.png`
- result.txt: `outputs/results/result.txt`
- Model checkpoint: `outputs/models/best_model.pt`
"""

    # Save report.md 
    with open("outputs/reports/report.md", "w", encoding="utf-8") as f:
        f.write(report)

    print("✅ STAGE 9 — Report done")
    print("- outputs/results/result.txt")
    print("- outputs/reports/report.md")

if __name__ == "__main__":
    main()
