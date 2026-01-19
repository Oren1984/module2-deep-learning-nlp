# 02_classic_nlp/src/stage4_model_baseline.py
# This script trains a baseline model using TF-IDF vectorization and Multinomial Naive
# Bayes classifier on the preprocessed SMS spam dataset.

from __future__ import annotations # Ensure forward compatibility

from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

# Define project directories
PROJECT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT / "data" / "processed"
MODELS_DIR = PROJECT / "outputs" / "models"
RESULTS_DIR = PROJECT / "outputs" / "results"

# Main function to train and evaluate the baseline model
def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load train and test splits
    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Run stage3_preprocess.py first.")

    # Load datasets 
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Define baseline pipeline: TF-IDF + MultinomialNB
    baseline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                stop_words="english",
                ngram_range=(1, 1),
                min_df=2
            )),
            ("clf", MultinomialNB()),
        ]
    )

    # Train and evaluate baseline
    baseline.fit(train_df["text"], train_df["y"])
    preds = baseline.predict(test_df["text"])
    acc = accuracy_score(test_df["y"], preds)
    
    # Save model and metrics
    out_model = MODELS_DIR / "baseline_tfidf_nb.joblib"
    joblib.dump(baseline, out_model)

    # Save metrics 
    out_metrics = RESULTS_DIR / "baseline_metrics.json"
    pd.Series({"accuracy": float(acc)}).to_json(out_metrics, indent=2)

    print("✅ Baseline trained: TF-IDF + MultinomialNB")
    print(f"Accuracy: {acc:.4f}")
    print(f"Saved model: {out_model}")
    print(f"Saved metrics: {out_metrics}")

if __name__ == "__main__":
    main()
