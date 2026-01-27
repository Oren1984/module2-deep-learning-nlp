# 03_hybrid_dl_nlp/src/stage4_model_baseline.py
# This module trains a baseline Logistic Regression model using TF-IDF features.
# It evaluates the model on the validation set and saves the model and results.

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from src.stage0_frame import Config, ensure_dirs

# Main execution to train and evaluate baseline model 
if __name__ == "__main__":
    cfg = Config()
    ensure_dirs(cfg)

    # Load TF-IDF features and labels 
    X_train, y_train = joblib.load(cfg.processed_dir / "A_train_tfidf.joblib")
    X_val, y_val     = joblib.load(cfg.processed_dir / "A_val_tfidf.joblib")

    # Train Logistic Regression model
    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)

    # Evaluate on validation set
    pred = clf.predict(X_val)
    acc = accuracy_score(y_val, pred)

    # Save model and results
    joblib.dump(clf, cfg.models_dir / "baseline_logreg_tfidf.joblib")
    joblib.dump({"val_accuracy": float(acc)}, cfg.results_dir / "baseline_results.joblib")
    print(f"✅ Baseline LogReg (TF-IDF) val_acc={acc:.4f}")
