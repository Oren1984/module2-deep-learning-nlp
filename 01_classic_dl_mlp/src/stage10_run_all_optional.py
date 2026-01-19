# 01_classic_dl_mlp/src/stage10_run_all_optional.py

"""
STAGE 10 — Optional Educational Orchestrator
===========================================

⚠️ NOTE (Industry Standard):
In real-world Deep Learning projects, we usually do NOT run the entire pipeline
as one script. We run training/evaluation/inference as separate commands/jobs.

This file exists only for learning/demo convenience.
"""

import subprocess
import sys

# Define the stages to run in order
STAGES = [
    "stage0_frame.py",
    "stage1_data_load.py",
    "stage2_quick_eda.py",
    "stage3_preprocess.py",
    "stage4_model_baseline.py",
    "stage5_train.py",
    "stage6_evaluate.py",
    "stage7_improve_regularization.py",
    "stage8_inference.py",
    "stage9_report.py",
]

# Function to run all stages sequentially
def run():
    for s in STAGES:
        print(f"\n▶ Running: {s}")
        code = subprocess.call([sys.executable, f"src/{s}"])
        if code != 0:
            print(f"\n⛔ Stopped at: {s} (exit code {code})")
            return
    print("\n✅ STAGE 11 completed — all stages executed")

if __name__ == "__main__":
    run()
