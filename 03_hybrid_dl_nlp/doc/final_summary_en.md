# 🧠 Hybrid DL + NLP — Final Project Summary  
### 📩 SMS Spam Detection (TF-IDF + MLP)

---

## 🔍 Overview

This project implements a **realistic end-to-end NLP classification pipeline** for detecting spam messages.  
The focus is on **engineering clarity, architectural reasoning, and reproducibility** — not experimentation.

The pipeline intentionally combines **classical NLP techniques** with a **deep learning model**, reflecting how real-world applied systems are often designed:

> **Start simple → validate assumptions → add deep learning only where it adds value**

All heavy processing (data preparation, training, evaluation) is executed via **Python scripts**.  
The repository exposes **results and artifacts**, while the notebook is used **only for inference and inspection**.

---

## 🎯 Task Definition

| Aspect | Description |
|------|-------------|
| **Problem** | Binary text classification (Spam vs Ham) |
| **Dataset** | SMS Spam Collection (cleaned & split) |
| **Learning Type** | Supervised classification |
| **Domain** | Applied NLP |

---

## 🧩 Why This Hybrid Approach (Design Rationale)

This project was **deliberately designed** as a **Hybrid NLP + Deep Learning pipeline**,  
not as a pure deep learning or transformer-based solution.

---

### 🟦 Why TF-IDF?

TF-IDF was chosen intentionally due to the **nature of the data and the problem**:

- The dataset is **small, structured, and domain-specific**
- TF-IDF provides:
  - Strong performance for classic NLP problems
  - Full interpretability
  - Fast training and low infrastructure cost
- Establishes a **clear and explainable baseline**, critical for production systems

> TF-IDF acts as a *controlled feature space*, not a shortcut.

---

### 🟩 Why Logistic Regression as Baseline?

The baseline model serves multiple engineering purposes:

- Validates:
  - Data quality
  - Preprocessing logic
  - Feature extraction correctness
- Provides a **performance reference point**
- Confirms the task is solvable **before introducing deep learning**

> No deep model should be trusted without a solid baseline.

---

### 🟨 Why MLP (and not Transformers)?

The choice of an **MLP over TF-IDF vectors** is intentional and pragmatic:

- Focus on **applied engineering**, not model complexity
- MLP enables:
  - Non-linear decision boundaries
  - Demonstration of DL fundamentals:
    - Training loops
    - Regularization
    - Early stopping
- Avoids:
  - High computational overhead
  - Black-box behavior
  - Overengineering

> This mirrors real-world systems where **DL is added incrementally**, not by default.

---

### 🟥 Why Not End-to-End Embeddings?

Embedding-based pipelines were **intentionally avoided**:

- While powerful, they are:
  - Less interpretable
  - Harder to debug
  - More opaque in failure modes
- This project prioritizes:
  - Control
  - Transparency
  - Reproducibility

> The goal is **engineering confidence**, not leaderboard chasing.

---

## ⚙️ Approach Summary

- **Text preprocessing:** normalization and cleaning  
- **Feature engineering:** TF-IDF (Bag-of-Words)  
- **Models:**
  - Baseline → Logistic Regression (TF-IDF)
  - Final → MLP (PyTorch) over TF-IDF vectors  
- **Evaluation strategy:** Hold-out TEST set  

---

## 🔄 Pipeline Stages

1. Dataset preparation  
2. Train / validation / test split  
3. Text preprocessing  
4. TF-IDF vectorization  
5. Baseline model (Logistic Regression)  
6. MLP training with early stopping  
7. Regularization tuning  
8. Inference  
9. TEST evaluation  
10. Final report generation  

📁 All stages are executed via scripts under `src/`.

---

## 📊 Results (TEST Set)

- **Accuracy:** ~0.98  
- **Metrics:** Precision, Recall, F1-score  

### 📦 Generated Artifacts
- Confusion Matrix  
- Classification Report  
- Serialized metrics (joblib)  

---

## 📓 Notebook Usage Policy

The notebook (`notebooks/demo.ipynb`) is intended **only** for:

- Inference examples  
- Sanity checks  
- Result inspection  

🚫 **Not used for training or hyperparameter tuning.**

---

## 🧠 Design Philosophy

- Minimal but complete  
- Script-driven pipeline  
- Fully reproducible results  
- Clear separation between:
  - Training
  - Evaluation
  - Inference
- Deep learning used **intentionally**, not automatically  

---

## 🏁 Final Note

This project reflects a **production-oriented hybrid NLP workflow**, demonstrating how:

- Classical NLP
- Deep learning
- Engineering discipline  

can coexist in a **clean, controlled, and scalable system**.

It emphasizes **decision-making**, not just implementation.
