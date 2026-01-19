# 03 — Hybrid DL + NLP (TF-IDF + MLP) — SMS Spam Detection

Goal: Build a **realistic, end-to-end NLP classification pipeline** using classic TF-IDF features with a small **PyTorch MLP**.
The pipeline is script-driven (reproducible), while the notebook is for **inference/inspection only**.

---

## Project Structure

- `src/` — stages (scripts) for the full pipeline  
- `data/raw/` — raw dataset (prepared CSV) + optional GloVe folder  
- `data/processed/` — splits + cleaned CSVs + vectorized features (joblib)  
- `outputs/models/` — trained model weights (`.pt`)  
- `outputs/results/` — saved metrics (`.joblib`)  
- `outputs/reports/` — final markdown report  
- `notebooks/` — demo notebook (inference + evaluation sanity checks)  
- `doc/` — final summaries (EN/HE)

---

## Dataset

Recommended: **SMS Spam Collection** (Spam vs Ham).

This project uses a prepared dataset file:

- Output of preparation step:
  - `data/raw/dataset_prepared.csv`

Expected columns:
- `label` (e.g., `ham` / `spam`)
- `text`

---

## How to Run (Step-by-Step)

From the project root:

```bash
# 0) Prepare dataset (creates data/raw/dataset_prepared.csv)
python -m src.prepare_dataset

# 1) Split data
python -m src.stage1_data_load

# 2) Clean text
python -m src.stage2_preprocess

# 3) Vectorization (TF-IDF + optional GloVe)
python -m src.stage3_vectorization

# 4) Baseline model (LogReg on TF-IDF)
python -m src.stage4_model_baseline

# 5) Train MLP (TF-IDF)
python -m src.stage5_train

# 6) Evaluate on TEST (official pipeline evaluation)
python -m src.stage6_evaluate

# 7) Improve / regularization tuning (optional but included)
python -m src.stage7_improve_regularization

# 8) Inference examples
python -m src.stage8_inference

# 9) Final report
python -m src.stage9_report

# 10) Run all optional
python -m src.stage10_run_all_optional