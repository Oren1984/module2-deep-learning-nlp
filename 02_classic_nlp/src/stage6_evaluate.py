# 02_classic_nlp/src/stage6_evaluate.py
# This script evaluates the best trained model on the test dataset.
# It computes accuracy, precision, recall, F1 score, and

from __future__ import annotations # Ensure forward compatibility

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

import joblib
import json

# Define project directories 
PROJECT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT / "data" / "processed"
MODELS_DIR = PROJECT / "outputs" / "models"
FIG_DIR = PROJECT / "outputs" / "figures"
REPORTS_DIR = PROJECT / "outputs" / "reports"
RESULTS_DIR = PROJECT / "outputs" / "results"

# Function to save confusion matrix as an image
def save_confusion_matrix(cm, out_path: Path) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(cm)
    
    # Set titles and labels
    ax.set_title("Confusion Matrix (rows=true, cols=pred)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    # label ticks for binary classification: 0=ham, 1=spam
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["ham", "spam"])
    ax.set_yticklabels(["ham", "spam"])
    
    # Annotate cells with counts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
            
    # Adjust layout and save figure
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

# Main evaluation function
def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load test split and best model
    test_path = PROCESSED_DIR / "test.csv"
    model_path = MODELS_DIR / "best_tfidf_linearsvc.joblib"
    
    # Check files exist 
    if not test_path.exists():
        raise FileNotFoundError("Run stage3_preprocess.py first.")
    if not model_path.exists():
        raise FileNotFoundError("Run stage5_train.py first (best model).")

    # Load data and model 
    test_df = pd.read_csv(test_path)
    model = joblib.load(model_path)
    
    # Make predictions 
    y_true = test_df["y"].values
    y_pred = model.predict(test_df["text"])

    # Compute metrics 
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    # Save confusion matrix and classification report 
    cm = confusion_matrix(y_true, y_pred)
    cm_path = FIG_DIR / "confusion_matrix.png"
    save_confusion_matrix(cm, cm_path)
    
    report_txt = classification_report(y_true, y_pred, target_names=["ham", "spam"], zero_division=0)
    report_path = REPORTS_DIR / "classification_report.txt"
    report_path.write_text(report_txt, encoding="utf-8")
    
    # Save metrics as JSON 
    metrics_path = RESULTS_DIR / "final_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("✅ Evaluation complete")
    print(metrics)
    print(f"Saved confusion matrix: {cm_path}")
    print(f"Saved classification report: {report_path}")
    print(f"Saved metrics: {metrics_path}")

if __name__ == "__main__":
    main()
