# 03_hybrid_dl_nlp/src/stage0_frame.py
# This module sets up the project structure, configuration, and utility functions
# for a hybrid deep learning NLP project.

from dataclasses import dataclass
from pathlib import Path
import random
import numpy as np
import torch

# Configuration dataclass to hold project settings
@dataclass
class Config:
    project_root: Path = Path(__file__).resolve().parents[1]
    raw_csv = project_root / "data" / "raw" / "dataset_prepared.csv"
    glove_path: Path = project_root / "data" / "raw" / "glove" / "glove.6B.100d.txt"

    # Directories for data and outputs
    processed_dir: Path = project_root / "data" / "processed"
    outputs_dir: Path = project_root / "outputs"
    models_dir: Path = outputs_dir / "models"
    results_dir: Path = outputs_dir / "results"
    figures_dir: Path = outputs_dir / "figures"
    reports_dir: Path = outputs_dir / "reports"
    
    # Hyperparameters and settings
    seed: int = 42
    test_size: float = 0.15
    val_size: float = 0.15  # מתוך ה-train (אחרי הפרדת test)

    # TF-IDF Vectorizer parameters
    max_features_tfidf: int = 30000
    min_df: int = 2
    max_df: float = 0.95
    ngram_range: tuple = (1, 2)

    # MLP parameters for TF-IDF features
    batch_size: int = 64
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.2
    hidden_dim: int = 256
    patience: int = 2  # early stopping

# Utility functions for setting seed, ensuring directories, and getting device
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Ensure necessary directories exist for the project
def ensure_dirs(cfg: Config):
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)

# Get the appropriate device (GPU if available, else CPU)
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

# Main execution to set up the project frame
if __name__ == "__main__":
    cfg = Config()
    ensure_dirs(cfg)
    set_seed(cfg.seed)
    print("✅ Project frame ready")
    print("Root:", cfg.project_root)
    print("Device:", get_device())
    print("Raw CSV exists:", cfg.raw_csv.exists())
