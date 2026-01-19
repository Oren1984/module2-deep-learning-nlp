# 02_classic_nlp/src/stage5_train.py
# This script trains a better model using TF-IDF vectorization and Linear SVM
# classifier on the preprocessed SMS spam dataset with hyperparameter tuning.

from __future__ import annotations # Ensure forward compatibility

from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
import joblib

# Define project directories 
PROJECT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT / "data" / "processed"
MODELS_DIR = PROJECT / "outputs" / "models"
RESULTS_DIR = PROJECT / "outputs" / "results"

# Main function to train and evaluate the best model
def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load train split 
    train_path = PROCESSED_DIR / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError("Run stage3_preprocess.py first.")

    # Load dataset 
    train_df = pd.read_csv(train_path)

    # Define pipeline: TF-IDF + LinearSVC
    pipe = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                stop_words="english",
            )),
            ("clf", LinearSVC()),
        ]
    )

    # Small grid (fast + educational) for hyperparameter tuning
    param_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [1, 2],
        "clf__C": [0.5, 1.0, 2.0],
    }
    
    # Perform grid search with cross-validation
    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="f1",
        cv=3,
        n_jobs=-1,
        verbose=0,
    )

    # Fit grid search 
    grid.fit(train_df["text"], train_df["y"])

    # Save best model and parameters 
    best_model = grid.best_estimator_
    out_model = MODELS_DIR / "best_tfidf_linearsvc.joblib"
    joblib.dump(best_model, out_model)

    # Save best parameters 
    out_params = RESULTS_DIR / "best_params.json"
    pd.Series(grid.best_params_).to_json(out_params, indent=2)

    print("✅ Trained best model: TF-IDF + LinearSVC (GridSearch)")
    print("Best params:", grid.best_params_)
    print(f"Saved model: {out_model}")
    print(f"Saved params: {out_params}")

if __name__ == "__main__":
    main()
