# 03_hybrid_dl_nlp/src/stage12_mongo_db.py
# This module stores evaluation results into a MongoDB database.
# It reads the results from Stage 6 and writes them into a MongoDB collection.

import joblib
from pymongo import MongoClient
from datetime import datetime, timezone
from src.stage0_frame import Config

# Function to get MongoDB collection
def get_collection(cfg: Config):
    """
    Expected env / cfg values:
    cfg.mongo_uri example:
    - mongodb://localhost:27017
    """
    client = MongoClient(cfg.mongo_uri)
    db = client[cfg.mongo_db]
    return db[cfg.mongo_collection]

# Main execution block
if __name__ == "__main__":
    cfg = Config()

    metrics = joblib.load(cfg.results_dir / "A_test_metrics.joblib")

    doc = {
        "experiment": metrics["experiment"],
        "accuracy": metrics["accuracy"],
        "n_test": metrics["n_test"],
        "confusion_matrix": metrics["confusion_matrix"],
        "classification_report": metrics["classification_report"],
        "created_at": datetime.now(timezone.utc),
    }

    collection = get_collection(cfg)
    collection.insert_one(doc)

    print("✅ MongoDB document inserted")
