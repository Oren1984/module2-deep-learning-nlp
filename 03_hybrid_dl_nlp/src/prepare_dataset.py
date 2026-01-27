# 03_hybrid_dl_nlp/src/prepare_dataset.py
# This module prepares the raw dataset into a standardized format.
# It handles different input formats and cleans the data.

from pathlib import Path
import pandas as pd

def main():
    root = Path(__file__).resolve().parents[1]  # 03_hybrid_dl_nlp

    src_path = root / "data" / "raw" / "dataset.csv"
    out_path = root / "data" / "raw" / "dataset_prepared.csv"

    if not src_path.exists():
        raise FileNotFoundError(f"Missing: {src_path}")

    df = pd.read_csv(src_path, encoding="latin-1")

    # UCI spam dataset: v1 = label, v2 = text
    if {"v1", "v2"}.issubset(df.columns):
        df2 = pd.DataFrame({
            "text": df["v2"].astype(str),
            "label": df["v1"].astype(str),
        })
    # Already standardized
    elif {"text", "label"}.issubset(df.columns):
        df2 = df[["text", "label"]].copy()
        df2["text"] = df2["text"].astype(str)
        df2["label"] = df2["label"].astype(str)
    else:
        raise ValueError(f"Unexpected columns: {list(df.columns)}")

    # Clean empties
    df2 = df2.dropna()
    df2 = df2[df2["text"].str.strip().ne("")]
    df2 = df2[df2["label"].str.strip().ne("")]

    # Sanity checks
    print("✅ Prepared dataset shape:", df2.shape)
    print("✅ Labels:", df2["label"].value_counts().to_dict())

    df2.to_csv(out_path, index=False, encoding="utf-8")
    print("✅ Saved:", out_path)

if __name__ == "__main__":
    main()
