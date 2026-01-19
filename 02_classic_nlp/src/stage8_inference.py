# 02_classic_nlp/src/stage8_inference.py
# This script demonstrates inference using the trained SMS spam classification model.
# It loads a saved model and predicts labels for sample SMS messages.

from __future__ import annotations # Ensure forward compatibility

from pathlib import Path
import joblib # Import joblib for loading the model

# Define project directories
PROJECT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT / "outputs" / "models"

# Sample SMS messages for inference
SAMPLES = [
    "Congrats! You've won a free prize. Click here to claim now!",
    "Hey, are we still meeting tomorrow at 10?",
    "URGENT: Your account has been compromised. Verify your password now.",
    "Can you send me the notes from class?",
]

# Main function to perform inference 
def main() -> None:
    
    # Prefer improved if exists, otherwise best
    model_path = MODELS_DIR / "improved_tfidf_linearsvc.joblib"
    if not model_path.exists():
        model_path = MODELS_DIR / "best_tfidf_linearsvc.joblib"
        
    # Check model file exists
    if not model_path.exists():
        raise FileNotFoundError("Train a model first (stage5_train or stage7_improve).")
    
    # Load the trained model
    model = joblib.load(model_path)

    print(f"✅ Loaded model: {model_path.name}\n")
    for t in SAMPLES:
        pred = int(model.predict([t])[0])  # 1=spam, 0=ham
        label = "SPAM" if pred == 1 else "HAM"
        print(f"[{label}] {t}")

if __name__ == "__main__":
    main()
