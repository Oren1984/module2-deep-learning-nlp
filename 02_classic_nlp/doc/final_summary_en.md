## 📄 Project 2 — Classic NLP (Statistical Pipeline)

---

## Binary text classification of SMS messages into:

* HAM (0) — legitimate messages

* SPAM (1) — unwanted / promotional / phishing messages

The project demonstrates a classic NLP pipeline using statistical feature extraction and linear models.


---


## 📊 Dataset

* Name: SMS Spam Collection

* Total samples: 5,572

* Class distribution:

    * Ham: 4,825

    * Spam: 747

* Spam rate: ~13.4%


# Text Statistics (Quick EDA)

* Average message length: ~15.5 words

* Median length: 12 words

* Messages are short, informal, and noisy — ideal for classic NLP baselines.


---

## 🧠 Approach (Classic NLP)

This project intentionally avoids deep learning and focuses on interpretable, fast, and strong baselines.

# Feature Engineering

* TF-IDF (Term Frequency–Inverse Document Frequency)

    * Converts text into numeric vectors

    * Bag-of-Words representation

    * Stopword removal

    * Optional n-grams (tested)


# Models

1. Baseline Model

    * TF-IDF + Multinomial Naive Bayes

2. Final Model

    * TF-IDF + Linear SVM (LinearSVC)

    * Hyperparameter tuning via GridSearch

---

## 🧪 Pipeline Stages

1. Data loading & normalization

2. Quick EDA (dataset statistics)

3. Text preprocessing & cleaning

4. Baseline model training

5. Improved model training (Linear SVM)

6. Evaluation on held-out test set

7. Regularization & parameter tuning

8.  Inference examples

9.  Automated report generation

✔️ All stages executed via Python scripts (src/)
✔️ Notebook used for evaluation & demonstration only

---

## 📈 Evaluation Results (Test Set)
Metric	Value
Accuracy	~0.986
Precision	~0.993
Recall	~0.899
F1-score	~0.944

# Interpretation

* Very high accuracy due to dataset simplicity

* Recall < Precision indicates the model is conservative:

    * Fewer false positives

    * Some spam messages may still be missed

---

## 📊 Confusion Matrix

1. Clear separation between ham and spam

2. Most errors are false negatives (spam → ham)

3. In real-world spam filtering, recall may be prioritized depending on business requirements

---

## 🔍 Inference Demo

Example predictions:

* “Congrats! You’ve won a free prize!” → SPAM

* “Hey, are we still meeting tomorrow?” → HAM

Demonstrates strong generalization on unseen messages.

---

## 📦 Output Artifacts

1. outputs/models/

    * baseline_tfidf_nb.joblib

    * best_tfidf_linearsvc.joblib

    *   improved_tfidf_linearsvc.joblib

2. outputs/figures/

    * confusion_matrix.png

3.  outputs/reports/

    * classification_report.txt

4.  outputs/results/

    * Metrics & parameter JSON files

---

## 🧠 Key Takeaways

* Classic NLP + linear models remain extremely effective for short-text classification

* TF-IDF is a strong baseline before deep learning

* Linear SVM outperforms Naive Bayes on this dataset

* High accuracy does not guarantee optimal recall — metrics must be interpreted in context


✔️ Project Status: Complete
✔️ Pipeline: End-to-End verified
✔️ Notebook: Presentation & inspection ready