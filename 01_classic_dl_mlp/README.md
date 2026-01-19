# Project 1 — Classic Deep Learning (MLP) — Fashion-MNIST

## Goal
Train a controlled, simple MLP on Fashion-MNIST:
- tensors & device (CPU/GPU)
- MLP architecture + activations
- training loop + validation
- regularization (Dropout, BatchNorm)
- early stopping
- inference timing
- outputs: plots + result.txt + report

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


Outputs

outputs/figures/: plots (loss/acc, confusion matrix, sample predictions)

outputs/models/: saved models

outputs/results/result.txt: key metrics summary

outputs/reports/report.md: final report


---

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