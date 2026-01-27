# 03_hybrid_dl_nlp/src/stage11_sql_db.py
# This module stores evaluation results into a SQL database (MySQL or PostgreSQL).
# It reads the results from Stage 6 and writes them into appropriate tables.

import joblib
import pandas as pd
from sqlalchemy import create_engine
from src.stage0_frame import Config

# Function to create a SQLAlchemy engine
def get_engine(cfg: Config):
    """
    Expected env / cfg values:
    cfg.sql_url examples:
    - mysql+mysqlconnector://user:password@localhost:3306/nlp_db
    - postgresql+psycopg2://user:password@localhost:5432/nlp_db
    """
    return create_engine(cfg.sql_url, echo=False, future=True)

# Main execution block
if __name__ == "__main__":
    cfg = Config()

    # Load evaluation results (Stage 6)
    metrics = joblib.load(cfg.results_dir / "A_test_metrics.joblib")

    # Convert metrics to flat table
    rows = []
    for label, vals in metrics["classification_report"].items():
        if isinstance(vals, dict):
            rows.append({
                "label": label,
                "precision": vals.get("precision"),
                "recall": vals.get("recall"),
                "f1_score": vals.get("f1-score"),
                "support": vals.get("support"),
            })

    # Create DataFrames
    df_metrics = pd.DataFrame(rows)
    df_summary = pd.DataFrame([{
        "experiment": metrics["experiment"],
        "accuracy": metrics["accuracy"],
        "n_test": metrics["n_test"],
    }])

    engine = get_engine(cfg)

    # Write to DB
    df_metrics.to_sql("nlp_classification_metrics", engine, if_exists="append", index=False)
    df_summary.to_sql("nlp_run_summary", engine, if_exists="append", index=False)

    print("✅ SQL metrics saved (MySQL / PostgreSQL)")
