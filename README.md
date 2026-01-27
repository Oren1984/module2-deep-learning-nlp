# 📦 Module 2 — Deep Learning & NLP

A structured, hands-on module covering **Deep Learning**, **Classic NLP**, and a **Hybrid DL + NLP** project —  
built with a clean, professional repository layout and a unified project template.

---

## What is this repository?

This repository contains **three focused projects** that together form a complete learning path for  
**Deep Learning & Natural Language Processing**, from fundamentals to a realistic hybrid setup.

The emphasis is on:
- Clear structure
- Practical experimentation
- Reproducible results
- Production-oriented thinking (without overengineering)

---

## 📁 Repository Structure


module2-deep-learning-nlp/
├── 00_environment_checks/ # PyTorch & optional GPU validation
├── 01_classic_dl_mlp/ # Deep Learning (MLP)
├── 02_classic_nlp/ # Statistical NLP pipeline
├── 03_hybrid_dl_nlp/ # Final hybrid project (DL + NLP)


Each project follows **the same template**:
- `src/` — step-by-step pipeline (stages)
- `data/` — raw & processed datasets
- `outputs/` — figures, results, reports
- `docs/` — final summaries
- `notebooks/` — demo & exploration

---

## 🧠 Projects Overview

### ✅ Project 1 — Classic Deep Learning (MLP)
**Focus:** Neural network fundamentals using PyTorch  
- Model architecture (MLP)
- Training loop, loss & optimization
- Regularization & early stopping
- Inference and performance analysis

📊 Dataset: *Fashion-MNIST*

---

### ✅ Project 2 — Classic NLP (Statistical Pipeline)
**Focus:** Traditional NLP before deep transformers  
- Text preprocessing & tokenization
- TF-IDF & n-grams
- Naive Bayes / SVM classifiers
- Proper evaluation (Precision / Recall / F1)

📊 Dataset: *SMS Spam Detection*

---

### ✅ Project 3 — Hybrid Deep Learning + NLP (Final)
**Focus:** Bridging classic NLP with deep learning  
- Text → vector representations
- MLP on top of text features
- Multiple experiments & comparison
- Optional database logging (SQL & MongoDB)

🚫 No transformers — by design.

---

## 🧪 Optional: Database Integration (Project 3 only)

The final project includes **optional experiment logging**:
- Relational DB (MySQL / PostgreSQL via SQLAlchemy)
- NoSQL DB (MongoDB)

Same logical experiment data — different storage paradigms.

---

## ⚙️ Environment Setup

```bash
python -m venv .module2
.module2/Scripts/Activate.ps1
pip install -r requirements.txt

Test PyTorch:

python -c "import torch; print(torch.__version__)"

---

Why this structure works

✔ One repository — multiple clean projects
✔ Consistent pipeline stages
✔ Clear separation of concerns
✔ Presentation-ready outputs
✔ Real-world mindset without unnecessary complexity

🏁 Final Note

This repository represents Module 2 — Deep Learning & NLP as a complete, coherent unit:

Deep Learning fundamentals

Classic NLP foundations

A realistic hybrid project

Simple. Structured. Professional.


---
