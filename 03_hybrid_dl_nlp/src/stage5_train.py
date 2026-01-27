# 03_hybrid_dl_nlp/src/stage5_train.py
# This module trains MLP models on TF-IDF and GloVe features.
# It saves the trained models and training metadata for evaluation.

import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.stage0_frame import Config, ensure_dirs, set_seed, get_device

# Define MLP model architecture for classification
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    # Forward pass 
    def forward(self, x):
        return self.net(x)

# Utility functions to convert data and create DataLoader
def to_dense(X):
    
    # support sparse from TF-IDF
    if hasattr(X, "toarray"):
        return X.toarray().astype(np.float32)
    return X.astype(np.float32)

# Create DataLoader from features and labels
def make_loader(X, y, batch_size, shuffle):
    X = torch.tensor(to_dense(X))
    y = torch.tensor(y).long()
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

# Train one MLP model and return best validation accuracy and model info
def train_one(experiment_name: str, train_path: str, val_path: str, model_out: str, cfg: Config):
    device = get_device()

    # Load data 
    X_train, y_train = joblib.load(train_path)
    X_val, y_val     = joblib.load(val_path)

    # Model, optimizer, loss function
    num_classes = int(np.max(y_train)) + 1
    input_dim = to_dense(X_train).shape[1]

    # Initialize model, optimizer, and loss function
    model = MLP(input_dim, num_classes, cfg.hidden_dim, cfg.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    # Data loaders 
    train_loader = make_loader(X_train, y_train, cfg.batch_size, True)
    val_loader   = make_loader(X_val, y_val, cfg.batch_size, False)

    # Training loop with early stopping
    best_val = -1.0
    bad_epochs = 0

    # Train for specified epochs 
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f"{experiment_name} train e{epoch}", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())

        # val acc calculation 
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                pred = torch.argmax(logits, dim=1)
                correct += int((pred == yb).sum().item())
                total += int(yb.size(0))
        val_acc = correct / max(total, 1)

        print(f"Epoch {epoch}: loss={total_loss/len(train_loader):.4f} val_acc={val_acc:.4f}")

        # Early stopping check 
        if val_acc > best_val:
            best_val = val_acc
            bad_epochs = 0
            torch.save(model.state_dict(), model_out)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print("⏹️ Early stopping")
                break

    return best_val, input_dim, num_classes

# Main execution to train MLP models on TF-IDF and GloVe features
if __name__ == "__main__":
    cfg = Config()
    ensure_dirs(cfg)
    set_seed(cfg.seed)

    # Experiment A: TF-IDF + MLP 
    best_val_A, inA, kA = train_one(
        "A(TFIDF+MLP)",
        str(cfg.processed_dir / "A_train_tfidf.joblib"),
        str(cfg.processed_dir / "A_val_tfidf.joblib"),
        str(cfg.models_dir / "mlp_tfidf.pt"),
        cfg
    )
    joblib.dump({"best_val_acc": float(best_val_A), "input_dim": int(inA), "num_classes": int(kA)},
                cfg.results_dir / "A_train_meta.joblib")

    # Experiment B: GloVe avg + MLP (if GloVe features exist)
    b_train = cfg.processed_dir / "B_train_glove.joblib"
    if b_train.exists():
        best_val_B, inB, kB = train_one(
            "B(GloVeAvg+MLP)",
            str(cfg.processed_dir / "B_train_glove.joblib"),
            str(cfg.processed_dir / "B_val_glove.joblib"),
            str(cfg.models_dir / "mlp_glove.pt"),
            cfg
        )
        joblib.dump({"best_val_acc": float(best_val_B), "input_dim": int(inB), "num_classes": int(kB)},
                    cfg.results_dir / "B_train_meta.joblib")
    else:
        print("⚠️ Experiment B skipped (no GloVe features found)")
