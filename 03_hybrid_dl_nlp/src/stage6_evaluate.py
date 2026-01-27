# 03_hybrid_dl_nlp/src/stage6_evaluate.py
# This module evaluates trained models on the test dataset and computes metrics.
# It supports two experiments: A (TF-IDF + MLP) and B (GloVe avg + MLP).

import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from src.stage0_frame import Config, ensure_dirs, set_seed, get_device

# Model definition (same as stage5)
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

    def forward(self, x):
        return self.net(x)


# Helpers 
def to_dense(X):
    # support sparse from TF-IDF
    if hasattr(X, "toarray"):
        return X.toarray().astype(np.float32)
    return X.astype(np.float32)

# Create DataLoader from features
def make_loader(X, batch_size):
    X = torch.tensor(to_dense(X))
    ds = TensorDataset(X)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)

# Prediction loop 
@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    for (xb,) in loader:
        xb = xb.to(device)
        logits = model(xb)
        pred = torch.argmax(logits, dim=1)
        preds.append(pred.cpu().numpy())
    return np.concatenate(preds, axis=0)

# Evaluate one experiment 
def evaluate_one(
    experiment_name: str,
    test_path: str,
    model_path: str,
    meta_path: str,
    out_path: str,
    cfg: Config,
):
    device = get_device()

    # Load test data
    X_test, y_test = joblib.load(test_path)

    # Load metadata saved from training (input_dim, num_classes)
    meta = joblib.load(meta_path)
    input_dim = int(meta["input_dim"])
    num_classes = int(meta["num_classes"])

    # Build model and load weights
    model = MLP(input_dim, num_classes, cfg.hidden_dim, cfg.dropout).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)

    # Predict
    loader = make_loader(X_test, cfg.batch_size)
    y_pred = predict(model, loader, device)

    # Metrics
    acc = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True)

    payload = {
        "experiment": experiment_name,
        "accuracy": acc,
        "confusion_matrix": cm,
        "classification_report": report,
        "n_test": int(len(y_test)),
    }
    joblib.dump(payload, out_path)

    print(f"✅ {experiment_name} TEST acc={acc:.4f} -> saved: {out_path}")
    return acc


# Main
if __name__ == "__main__":
    cfg = Config()
    ensure_dirs(cfg)
    set_seed(cfg.seed)

    # Experiment A: TF-IDF + MLP (evaluate on TEST)
    evaluate_one(
        "A(TFIDF+MLP)",
        str(cfg.processed_dir / "A_test_tfidf.joblib"),
        str(cfg.models_dir / "mlp_tfidf.pt"),
        str(cfg.results_dir / "A_train_meta.joblib"),
        str(cfg.results_dir / "A_test_metrics.joblib"),
        cfg,
    )

    # Experiment B: GloVe avg + MLP (optional)
    b_test = cfg.processed_dir / "B_test_glove.joblib"
    if b_test.exists():
        evaluate_one(
            "B(GloVeAvg+MLP)",
            str(cfg.processed_dir / "B_test_glove.joblib"),
            str(cfg.models_dir / "mlp_glove.pt"),
            str(cfg.results_dir / "B_train_meta.joblib"),
            str(cfg.results_dir / "B_test_metrics.joblib"),
            cfg,
        )
    else:
        print("⚠️ Experiment B skipped (no GloVe features found)")