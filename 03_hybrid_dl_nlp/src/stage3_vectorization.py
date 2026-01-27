# 03_hybrid_dl_nlp/src/stage3_vectorization.py
# This module vectorizes the preprocessed text data using TF-IDF and GloVe embeddings.
# It saves the vectorized features and label encodings for downstream model training.

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path
from src.stage0_frame import Config, ensure_dirs
from src.stage2_preprocess import simple_tokenize

# Load preprocessed splits from CSV files
def load_splits(cfg: Config):
    train = pd.read_csv(cfg.processed_dir / "train_clean.csv")
    val   = pd.read_csv(cfg.processed_dir / "val_clean.csv")
    test  = pd.read_csv(cfg.processed_dir / "test_clean.csv")
    return train, val, test

# Encode labels to integers and save the encoder object
def encode_labels(train, val, test, cfg: Config):
    le = LabelEncoder()
    y_train = le.fit_transform(train["label"].astype(str))
    y_val   = le.transform(val["label"].astype(str))
    y_test  = le.transform(test["label"].astype(str))
    joblib.dump(le, cfg.processed_dir / "label_encoder.joblib")
    return y_train, y_val, y_test, le

# Build TF-IDF vectorizer and transform texts to feature matrices
def build_tfidf(train_texts, cfg: Config):
    vec = TfidfVectorizer(
    max_features=5000,
    min_df=2,
    max_df=0.95,
    ngram_range=(1, 1),
    )
    X = vec.fit_transform(train_texts)
    return vec, X

# Load GloVe embeddings from file and return as a dictionary
def load_glove(glove_path: Path):
    
    # expects: word val1 val2 ...
    embeddings = {}
    with glove_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip().split()
            if len(parts) < 10:
                continue
            word = parts[0]
            vec = np.asarray(parts[1:], dtype=np.float32)
            embeddings[word] = vec
    dim = len(next(iter(embeddings.values())))
    return embeddings, dim

# Convert texts to average GloVe embeddings feature matrix
def texts_to_avg_glove(texts, glove, dim):
    X = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        toks = simple_tokenize(t)
        vecs = [glove[w] for w in toks if w in glove]
        if vecs:
            X[i] = np.mean(vecs, axis=0)
    return X

# Main execution to vectorize data and save features and labels
if __name__ == "__main__":
    cfg = Config()
    ensure_dirs(cfg)
    
    # Load splits and encode labels 
    train, val, test = load_splits(cfg)
    y_train, y_val, y_test, le = encode_labels(train, val, test, cfg)

    # --- Experiment A features: TF-IDF
    tfidf_vec, X_train_tfidf = build_tfidf(train["text_clean"].tolist(), cfg)
    X_val_tfidf  = tfidf_vec.transform(val["text_clean"].tolist())
    X_test_tfidf = tfidf_vec.transform(test["text_clean"].tolist())

    # Save TF-IDF vectorizer and features
    joblib.dump(tfidf_vec, cfg.processed_dir / "tfidf_vectorizer.joblib")
    joblib.dump((X_train_tfidf, y_train), cfg.processed_dir / "A_train_tfidf.joblib")
    joblib.dump((X_val_tfidf, y_val),     cfg.processed_dir / "A_val_tfidf.joblib")
    joblib.dump((X_test_tfidf, y_test),   cfg.processed_dir / "A_test_tfidf.joblib")
    print("✅ Saved TF-IDF features for Experiment A")

    # --- Experiment B features: GloVe avg
    if cfg.glove_path.exists():
        glove, dim = load_glove(cfg.glove_path)
        X_train_g = texts_to_avg_glove(train["text_clean"].tolist(), glove, dim)
        X_val_g   = texts_to_avg_glove(val["text_clean"].tolist(), glove, dim)
        X_test_g  = texts_to_avg_glove(test["text_clean"].tolist(), glove, dim)

        # Save GloVe features and dimension
        joblib.dump(dim, cfg.processed_dir / "glove_dim.joblib")
        joblib.dump((X_train_g, y_train), cfg.processed_dir / "B_train_glove.joblib")
        joblib.dump((X_val_g, y_val),     cfg.processed_dir / "B_val_glove.joblib")
        joblib.dump((X_test_g, y_test),   cfg.processed_dir / "B_test_glove.joblib")
        print("✅ Saved GloVe-avg features for Experiment B")
    else:
        print("⚠️ GloVe file not found -> skipping Experiment B feature build")
        print("Expected:", cfg.glove_path)
