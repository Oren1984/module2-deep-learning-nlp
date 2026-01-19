# 02_classic_nlp/src/stage1_data_load.py
# This script loads and normalizes the SMS spam dataset.
# It handles various common dataset formats and encodings,

from __future__ import annotations # Ensure forward compatibility

from pathlib import Path
import pandas as pd

# Define project directories
PROJECT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT / "data" / "raw"
PROCESSED_DIR = PROJECT / "data" / "processed"

# Candidate dataset file names
CANDIDATES = [
    RAW_DIR / "sms_spam.csv",
    RAW_DIR / "spam.csv",
    RAW_DIR / "SMSSpamCollection",
    RAW_DIR / "SMSSpamCollection.csv",
    RAW_DIR / "SMSSpamCollection.tsv",
    RAW_DIR / "SMSSpamCollection.txt",
]

# Robust CSV/TSV reader with multiple encoding and delimiter support
def _try_read(path: Path) -> pd.DataFrame:
    """
    Robust loader:
    - tries multiple encodings
    - auto-detects delimiter (comma/semicolon/tab/pipe)
    - handles TSV saved as .csv
    """
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    seps = [",", ";", "\t", "|"]

    last_err: Exception | None = None
    
    # Try all combinations
    for enc in encodings:
        # 1) Try pandas auto delimiter (engine=python can infer)
        try:
            df = pd.read_csv(path, encoding=enc, sep=None, engine="python")
            if df.shape[1] >= 2:
                return df
        except Exception as e:
            last_err = e

        # 2) Try common delimiters explicitly
        for sep in seps:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep, engine="python")
                if df.shape[1] >= 2:
                    return df
            except Exception as e:
                last_err = e

    raise RuntimeError(f"Found {path.name} but failed to read it. Last error: {last_err}")

# Normalize columns to standard names: "label", "text"
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [str(c).lower().strip() for c in df.columns]
    df.columns = cols

    # Kaggle spam.csv style: v1 (label), v2 (text)
    if "v1" in cols and "v2" in cols:
        out = df[["v1", "v2"]].copy()
        out.columns = ["label", "text"]
        return out

    # Standard style
    if "label" in cols and "text" in cols:
        return df[["label", "text"]].copy()

    # If exactly 2 columns, assume label + text
    if df.shape[1] >= 2:
        out = df.iloc[:, :2].copy()
        out.columns = ["label", "text"]
        return out

    raise ValueError(f"Unsupported dataset format. Columns: {df.columns.tolist()}")

# Normalize labels to binary: ham=0, spam=1
def _normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    
    mapping = {
        "spam": 1,
        "ham": 0,
        "1": 1,
        "0": 0,
        "true": 1,
        "false": 0,
    }
    df["y"] = df["label"].map(mapping)
    
    # Check for unknown labels
    if df["y"].isna().any():
        unknown = sorted(df.loc[df["y"].isna(), "label"].unique().tolist())
        raise ValueError(f"Unknown labels found: {unknown}. Expected ham/spam or 0/1.")
    
    # Drop original label column
    df["text"] = df["text"].astype(str).fillna("")
    df = df[df["text"].str.len() > 0].reset_index(drop=True)

    return df[["text", "y"]]

# Load dataset from any candidate file
def _load_any() -> pd.DataFrame:
    for p in CANDIDATES:
        if p.exists():
            df = _try_read(p)
            df = _normalize_columns(df)
            return df
    raise FileNotFoundError("No dataset found. Put your file under data/raw/ (e.g., sms_spam.csv).")

# Main function to load and normalize dataset
def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load and normalize dataset
    df = _load_any()
    df = _normalize_labels(df)
    
    # Save normalized dataset
    out_path = PROCESSED_DIR / "dataset_clean_base.csv"
    df.to_csv(out_path, index=False)

    print("✅ Loaded & normalized dataset")
    print(f"Rows: {len(df)}")
    print(df["y"].value_counts().rename(index={0: "ham", 1: "spam"}))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
