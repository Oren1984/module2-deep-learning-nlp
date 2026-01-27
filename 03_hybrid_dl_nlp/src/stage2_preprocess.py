# 03_hybrid_dl_nlp/src/stage2_preprocess.py
# This module preprocesses the text data by cleaning and tokenizing.
# It saves the cleaned data into new CSV files in the processed data directory.

import re
import pandas as pd
from src.stage0_frame import Config, ensure_dirs

# Simple regex-based tokenizer
_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")

# Basic text cleaning function
def simple_clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Tokenization function using regex
def simple_tokenize(text: str):
    return _TOKEN_RE.findall(text.lower())

# Preprocess and save cleaned data
def preprocess_split(path_in, path_out):
    df = pd.read_csv(path_in)
    df["text_clean"] = df["text"].astype(str).apply(simple_clean)
    df.to_csv(path_out, index=False)

# Main execution to preprocess train, val, and test splits
if __name__ == "__main__":
    cfg = Config()
    ensure_dirs(cfg)

    for name in ["train", "val", "test"]:
        inp = cfg.processed_dir / f"{name}.csv"
        out = cfg.processed_dir / f"{name}_clean.csv"
        preprocess_split(inp, out)
        print("✅", name, "->", out)
