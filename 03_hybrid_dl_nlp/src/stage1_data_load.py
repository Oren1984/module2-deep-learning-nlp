# 03_hybrid_dl_nlp/src/stage1_data_load.py
# This module handles loading and splitting the dataset into train, validation, and test sets.
# It saves the splits into the processed data directory.

import pandas as pd
from sklearn.model_selection import train_test_split
from src.stage0_frame import Config, ensure_dirs, set_seed

# Function to load dataset from CSV
def load_dataset(cfg: Config) -> pd.DataFrame:
    if not cfg.raw_csv.exists():
        raise FileNotFoundError(f"Missing dataset file: {cfg.raw_csv}")
    df = pd.read_csv(cfg.raw_csv)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("dataset.csv must contain columns: text,label")
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)
    return df

# Function to split dataset and save to processed directory
def split_save(df: pd.DataFrame, cfg: Config):
    # Test split
    train_df, test_df = train_test_split(
        df, test_size=cfg.test_size, random_state=cfg.seed, stratify=df["label"]
    )

    # Val split from train
    val_rel = cfg.val_size / (1.0 - cfg.test_size)
    train_df, val_df = train_test_split(
        train_df, test_size=val_rel, random_state=cfg.seed, stratify=train_df["label"]
    )

    # Save splits to CSV
    out_train = cfg.processed_dir / "train.csv"
    out_val = cfg.processed_dir / "val.csv"
    out_test = cfg.processed_dir / "test.csv"
    train_df.to_csv(out_train, index=False)
    val_df.to_csv(out_val, index=False)
    test_df.to_csv(out_test, index=False)

    print("✅ Saved splits:")
    print("train:", train_df.shape, "->", out_train)
    print("val  :", val_df.shape, "->", out_val)
    print("test :", test_df.shape, "->", out_test)
    print("labels:", sorted(df["label"].unique())[:10], "..." if df["label"].nunique() > 10 else "")

# Main execution to load, split, and save dataset
if __name__ == "__main__":
    cfg = Config()
    ensure_dirs(cfg)
    set_seed(cfg.seed)

    df = load_dataset(cfg)
    print("✅ Loaded:", df.shape)
    split_save(df, cfg)
