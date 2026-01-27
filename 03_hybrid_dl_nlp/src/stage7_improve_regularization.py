# 03_hybrid_dl_nlp/src/stage7_improve_regularization.py
# This module improves regularization for MLP models on TF-IDF features.
# It tunes dropout and weight decay hyperparameters.

from copy import deepcopy
from src.stage0_frame import Config
from src.stage5_train import train_one

# Main execution to tune regularization hyperparameters
if __name__ == "__main__":
    cfg = Config()

    # Experiment: TF-IDF + MLP with improved regularization
    cfg2 = deepcopy(cfg)
    cfg2.dropout = 0.4
    cfg2.weight_decay = 5e-4

    # Train with new regularization settings
    best_val, _, _ = train_one(
        "A(TFIDF+MLP)_REG_TUNE",
        str(cfg2.processed_dir / "A_train_tfidf.joblib"),
        str(cfg2.processed_dir / "A_val_tfidf.joblib"),
        str(cfg2.models_dir / "mlp_tfidf_reg_tune.pt"),
        cfg2
    )
    print("✅ Reg-tune best val acc:", best_val)
