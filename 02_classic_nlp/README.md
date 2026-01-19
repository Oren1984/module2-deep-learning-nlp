# Project 2 — Classic NLP (Statistical Pipeline)

Goal: Classic text classification pipeline ("old-school NLP") using TF-IDF + linear models.

## Structure
- `src/` stages 0–10
- `data/raw/` put the dataset CSV here
- `data/processed/` cleaned dataset + splits
- `outputs/figures/` confusion matrix, charts
- `outputs/models/` saved model/vectorizer
- `outputs/reports/` markdown report
- `outputs/results/` metrics JSON/CSV

## Dataset
Recommended: SMS Spam Detection dataset.
Place file as:
`data/raw/sms_spam.csv`

Expected columns:
- `label` (ham/spam or 0/1)
- `text`

## Run
```bash
python src/stage0_frame.py
python src/stage1_data_load.py
python src/stage2_quick_eda.py
python src/stage3_preprocess.py
python src/stage4_model_baseline.py
python src/stage5_train.py
python src/stage6_evaluate.py
python src/stage7_improve_regularization.py
python src/stage8_inference.py
python src/stage9_report.py

stage10 _run_all_optional.py is provided for educational convenience only; production workflows run training/eval/inference as separate jobs.


## 2) `requirements.txt`

```txt
torch
torchvision
numpy
matplotlib
scikit-learn

3) .gitignore
__pycache__/
*.pyc
.venv/
venv/
.env

# keep folder structure but allow outputs (you WANT images/reports)
# so do NOT ignore outputs entirely
# If you ever want to ignore large model files, uncomment:
# outputs/models/*.pt