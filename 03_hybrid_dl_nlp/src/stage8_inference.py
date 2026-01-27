# 03_hybrid_dl_nlp/src/stage8_inference.py
# This module performs inference using trained MLP models on TF-IDF and GloVe features
# It provides functions to predict classes for new text samples.

import joblib
import torch
import numpy as np
from src.stage0_frame import Config, get_device
from src.stage5_train import MLP, to_dense
from src.stage2_preprocess import simple_clean
from src.stage3_vectorization import load_glove, texts_to_avg_glove

# Predict classes for new texts using TF-IDF + MLP model
def predict_tfidf(texts, cfg: Config):
    le = joblib.load(cfg.processed_dir / "label_encoder.joblib")
    vec = joblib.load(cfg.processed_dir / "tfidf_vectorizer.joblib")
    meta = joblib.load(cfg.results_dir / "A_train_meta.joblib")

    # Transform texts to TF-IDF features
    X = vec.transform([simple_clean(t) for t in texts])
    device = get_device()
    model = MLP(meta["input_dim"], meta["num_classes"], cfg.hidden_dim, cfg.dropout).to(device)
    model.load_state_dict(torch.load(cfg.models_dir / "mlp_tfidf.pt", map_location=device))
    model.eval()

    # Predict classes 
    with torch.no_grad():
        logits = model(torch.tensor(to_dense(X)).to(device))
        pred = torch.argmax(logits, dim=1).cpu().numpy()

    return [le.classes_[i] for i in pred]

# Predict classes for new texts using GloVe avg + MLP model
def predict_glove(texts, cfg: Config):
    if not cfg.glove_path.exists():
        raise FileNotFoundError("GloVe not found")

    # Load label encoder and model metadata
    le = joblib.load(cfg.processed_dir / "label_encoder.joblib")
    meta = joblib.load(cfg.results_dir / "B_train_meta.joblib")

    # Load GloVe embeddings and convert texts to average GloVe features
    glove, dim = load_glove(cfg.glove_path)
    X = texts_to_avg_glove([simple_clean(t) for t in texts], glove, dim)

    # Initialize and load model 
    device = get_device()
    model = MLP(meta["input_dim"], meta["num_classes"], cfg.hidden_dim, cfg.dropout).to(device)
    model.load_state_dict(torch.load(cfg.models_dir / "mlp_glove.pt", map_location=device))
    model.eval()

    # Predict classes
    with torch.no_grad():
        logits = model(torch.tensor(X).to(device))
        pred = torch.argmax(logits, dim=1).cpu().numpy()

    return [le.classes_[i] for i in pred]

# Main execution for testing inference functions
if __name__ == "__main__":
    cfg = Config()
    samples = [
        "This was amazing and I would buy again",
        "Worst experience ever, totally disappointed"
    ]

    print("TF-IDF+MLP:", predict_tfidf(samples, cfg))

    # GloVeAvg+MLP prediction (if model exists)
    if (cfg.models_dir / "mlp_glove.pt").exists():
        print("GloVeAvg+MLP:", predict_glove(samples, cfg))
