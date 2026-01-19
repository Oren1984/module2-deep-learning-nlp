# 01_classic_dl_mlp/src/stage5_train.py
# This script trains the MLP model on the Fashion-MNIST dataset.
# It includes training loop, validation, early stopping, and saves the best model and training history.

import os
import json
import time
import torch
import torch.nn as nn

# Optimizer import 
from torch.optim import Adam
from stage1_data_load import get_dataloaders, get_device
from stage4_model_baseline import MLP

# Function to create necessary directories
def ensure_dirs():
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/results", exist_ok=True)

# Evaluation function for validation/test
@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Iterate over data loader 
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        
        # Get predictions and accuracy
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    # Return average loss and accuracy 
    return total_loss / total, correct / total 

# Main training function 
def train():
    ensure_dirs()
    device = get_device()
    train_loader, val_loader, _ = get_dataloaders(batch_size=64)

    # Baseline params for MLP model
    config = {
        "model": "MLP",
        "hidden1": 256,
        "hidden2": 128,
        "dropout": 0.2,
        "use_bn": True,
        "lr": 1e-3,
        "epochs": 20,
        "early_stopping_patience": 3
    }

    # Instantiate model, loss, optimizer
    model = MLP(
        hidden1=config["hidden1"],
        hidden2=config["hidden2"],
        dropout=config["dropout"],
        use_bn=config["use_bn"]
    ).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config["lr"])
    
    # Initialize training history
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    # Initialize early stopping variables
    best_val_loss = float("inf")
    patience = 0
    best_path = "outputs/models/best_model.pt"
    
    # Training loop 
    start = time.time()
    
    # Iterate over epochs 
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Iterate over training data
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            # Training step 
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            # Update running loss and accuracy
            running_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
        
        # Compute epoch metrics
        train_loss = running_loss / total
        train_acc = correct / total
        
        # Validate the model 
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        
        # Save metrics to history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:02d} | train loss {train_loss:.4f} acc {train_acc:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f}")

        # Early stopping check 
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience = 0
            torch.save({"model_state": model.state_dict(), "config": config}, best_path)
        else:
            patience += 1
            if patience >= config["early_stopping_patience"]:
                print("⛔ Early stopping triggered.")
                break
            
    # End of training loop
    total_time = time.time() - start

    # Save training history and time 
    with open("outputs/results/train_history.json", "w", encoding="utf-8") as f:
        json.dump({"config": config, "history": history, "train_seconds": total_time}, f, indent=2)

    print("✅ STAGE 5 — Training done")
    print("Saved:")
    print("- outputs/models/best_model.pt")
    print("- outputs/results/train_history.json")

if __name__ == "__main__":
    train()
