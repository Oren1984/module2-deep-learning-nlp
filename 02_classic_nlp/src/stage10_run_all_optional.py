# 02 _classic_nlp/src/stage10_run_all_optional.py
# This script runs all stages of the classic NLP SMS spam classification pipeline sequentially. 
# It is intended for educational convenience to execute the entire workflow in one go.

from __future__ import annotations # Ensure forward compatibility

import subprocess
import sys
from pathlib import Path

# Define project directories
PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"

# List of all stage scripts to run in order
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

# Function to run a single stage script
def run(pyfile: str) -> None:
    path = SRC / pyfile
    print("\n" + "=" * 60)
    print(f"▶ Running: {pyfile}")
    print("=" * 60)
    subprocess.check_call([sys.executable, str(path)])

# Main function to run all stages sequentially
def main() -> None:
    for s in STAGES:
        run(s)
    print("\n✅ All stages completed (educational convenience).")

if __name__ == "__main__":
    main()
