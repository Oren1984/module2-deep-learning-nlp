# Hybrid DL + NLP — Final Project Summary  
**SMS Spam Detection (TF-IDF + MLP)**

## Overview
This project implements a **realistic end-to-end NLP classification pipeline** for detecting spam messages.
The focus is on **engineering clarity and reproducibility**, not experimentation.

All heavy processing (data preparation, training, evaluation) is executed via Python scripts.
This repository exposes **results and artifacts**, while the notebook is used for **inference and inspection only**.

---

## Task
- **Problem:** Binary text classification (Spam vs Ham)
- **Dataset:** SMS Spam Collection (cleaned & split)
- **Type:** Supervised classification

---

## Approach
- **Text preprocessing:** normalization and cleaning
- **Feature engineering:** TF-IDF (Bag-of-Words)
- **Models:**
  - Baseline: Logistic Regression (TF-IDF)
  - Final: MLP (PyTorch) over TF-IDF vectors
- **Evaluation:** Hold-out TEST set

---

## Pipeline Stages
1. Dataset preparation
2. Train / validation / test split
3. Text preprocessing
4. TF-IDF vectorization
5. Baseline model (LogReg)
6. MLP training with early stopping
7. Regularization tuning
8. Inference
9. TEST evaluation
10. Final report generation

All stages are executed via scripts under `src/`.

---

## Results (TEST)
- **Accuracy:** ~0.98
- **Metrics:** Precision, Recall, F1-score
- **Artifacts:**
  - Confusion Matrix
  - Classification Report
  - Serialized metrics (`joblib`)

---

## Notebook Usage
The notebook (`notebooks/demo.ipynb`) is intended for:
- Inference examples
- Sanity checks
- Result inspection

🚫 Not for training or hyperparameter tuning.

---

## Design Philosophy
- Minimal but complete
- Script-driven pipeline
- Reproducible results
- No unnecessary components

This project reflects a **production-oriented NLP workflow**, not a research experiment.
