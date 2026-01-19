# 02_classic_nlp/src/stage2_quick_eda.py
# This script performs quick exploratory data analysis (EDA) on the cleaned SMS spam dataset.
# It computes basic statistics and saves a summary report.

from __future__ import annotations # Ensure forward compatibility

from pathlib import Path
import pandas as pd

# Define project directories
PROJECT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT / "data" / "processed"
OUTPUTS_DIR = PROJECT / "outputs" / "results"

# Main function to perform quick EDA
def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load cleaned dataset
    in_path = PROCESSED_DIR / "dataset_clean_base.csv"
    if not in_path.exists():
        raise FileNotFoundError("Run stage1_data_load.py first.")

    # Read dataset 
    df = pd.read_csv(in_path)
    
    # Compute basic statistics
    df["len_chars"] = df["text"].astype(str).str.len()
    df["len_words"] = df["text"].astype(str).str.split().map(len)

    # Prepare summary report
    summary = {
        "rows": int(len(df)),
        "spam_rate": float(df["y"].mean()),
        "avg_len_chars": float(df["len_chars"].mean()),
        "avg_len_words": float(df["len_words"].mean()),
        "median_len_words": float(df["len_words"].median()),
        "ham_rows": int((df["y"] == 0).sum()),
        "spam_rows": int((df["y"] == 1).sum()),
    }

    # Save summary as JSON 
    out_path = OUTPUTS_DIR / "eda_quick_summary.json"
    pd.Series(summary).to_json(out_path, indent=2)

    print("✅ Quick EDA")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
