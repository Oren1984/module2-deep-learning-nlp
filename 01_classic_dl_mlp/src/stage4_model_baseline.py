# 01_classic_dl_mlp/src/stage4_model_baseline.py
# This script defines a baseline Multi-Layer Perceptron (MLP) model for Fashion-MNIST classification.
# It includes two hidden layers with optional dropout and batch normalization.

import torch
import torch.nn as nn

# Define the MLP model
class MLP(nn.Module):
    
    # Initialize the MLP model
    def __init__(self, input_dim=784, num_classes=10, hidden1=256, hidden2=128,
                 dropout=0.0, use_bn=False):
        
        # Initialize the MLP model with two hidden layers
        super().__init__()
        layers = []
        
        # First hidden layer
        layers.append(nn.Linear(input_dim, hidden1))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden1))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
            
        # Second hidden layer
        layers.append(nn.Linear(hidden1, hidden2))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden2))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
            
        # Output layer
        layers.append(nn.Linear(hidden2, num_classes))
        self.net = nn.Sequential(*layers)

    # Forward pass of the MLP model
    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten inside model
        return self.net(x)         # logits

# Main function to demonstrate model instantiation and forward pass
def main():
    model = MLP(dropout=0.0, use_bn=False)
    x = torch.randn(8, 1, 28, 28)
    out = model(x)
    print("✅ STAGE 4 — Model Baseline")
    print("Output shape:", out.shape)  # (8,10)

if __name__ == "__main__":
    main()
