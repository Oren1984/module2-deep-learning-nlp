# 02_classic_nlp/src/stage3_preprocess.py
# This script cleans and preprocesses the SMS spam dataset.
# It removes unwanted patterns and splits the data into training and test sets.

from __future__ import annotations # Ensure forward compatibility

from pathlib import Path
import re
import pandas as pd
from sklearn.model_selection import train_test_split 

# Define project directories 
PROJECT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT / "data" / "processed"

# Regular expressions for cleaning text 
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MULTI_SPACE = re.compile(r"\s+")

# Text cleaning function 
def clean_text(s: str) -> str:
    s = str(s)
    s = s.strip()
    s = _URL_RE.sub(" <URL> ", s)
    s = s.replace("\n", " ").replace("\r", " ")
    s = _MULTI_SPACE.sub(" ", s)
    return s

# Main function to preprocess and split the dataset
def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Load cleaned dataset
    in_path = PROCESSED_DIR / "dataset_clean_base.csv"
    if not in_path.exists():
        raise FileNotFoundError("Run stage1_data_load.py first.")

    # Read dataset and clean text
    df = pd.read_csv(in_path)
    df["text"] = df["text"].map(clean_text)

    # Split once and persist (important for reproducibility)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["y"],
        test_size=0.2,
        random_state=42,
        stratify=df["y"],
    )

    # Save train and test splits
    train_df = pd.DataFrame({"text": X_train, "y": y_train})
    test_df = pd.DataFrame({"text": X_test, "y": y_test})

    # Save train and test splits
    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("✅ Preprocess + split done")
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")
    print(f"Saved: {train_path}")
    print(f"Saved: {test_path}")

if __name__ == "__main__":
    main()
