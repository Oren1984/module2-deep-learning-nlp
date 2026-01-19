# 02_classic_nlp/src/stage0_frame.py
# This script outlines the project structure and goals for a classic NLP task.
# It specifies the data location and expected outputs.

from pathlib import Path

# Define the project root directory
PROJECT = Path(__file__).resolve().parents[1]

# Set up paths for data and outputs
def main() -> None:
    print("✅ Project 2 — Classic NLP (Statistical Pipeline)")
    print("Goal: classify SMS messages as SPAM vs HAM using classic NLP.")
    print("\nPipeline:")
    print("1) Load raw data")
    print("2) Quick EDA")
    print("3) Clean & preprocess text")
    print("4) Baseline model (TF-IDF + MultinomialNB)")
    print("5) Train better model (TF-IDF + Linear SVM)")
    print("6) Evaluate (Accuracy, Precision, Recall, F1 + Confusion Matrix)")
    print("7) Improve (n-grams / regularization)")
    print("8) Inference demo")
    print("9) Final report\n")

    print("📌 Data expected at: data/raw/sms_spam.csv")
    print("📦 Outputs will be written to: outputs/*")

if __name__ == "__main__":
    main()
