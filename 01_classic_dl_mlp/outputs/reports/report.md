# Classic Deep Learning — MLP (Fashion-MNIST)

## Goal
Train an MLP for 10-class classification on Fashion-MNIST with a clean, controlled pipeline.

## Setup
- Device: **cpu**
- Model: **MLP**
- Params: hidden1=256, hidden2=128, dropout=0.2, batchnorm=True
- Optimizer: Adam (lr=0.001)
- Early stopping patience: 3

## Results
- Best Validation Accuracy: **0.8957**
- Test Accuracy: **0.8929**

## Artifacts
- Curves: `outputs/figures/loss_curves.png`, `outputs/figures/acc_curves.png`
- Confusion Matrix: `outputs/figures/confusion_matrix.png`
- result.txt: `outputs/results/result.txt`
- Model checkpoint: `outputs/models/best_model.pt`
