# 03_hybrid_dl_nlp/src/stage10_run_all_optional.py
# This module runs all optional stages of the pipeline sequentially.
# It is intended for educational convenience to execute the entire workflow in one go.

import subprocess
import sys

# List of optional modules to run
MODULES = [
    "src.prepare_dataset",
    "src.stage1_data_load",
    "src.stage2_preprocess",
    "src.stage3_vectorization",
    "src.stage4_model_baseline",
    "src.stage5_train",
    "src.stage6_evaluate",
    "src.stage9_report",
]

# Add optional database storage modules
if __name__ == "__main__":
    for m in MODULES:
        print("\n" + "=" * 60)
        print("▶ Running:", m)
        r = subprocess.run([sys.executable, "-m", m])
        if r.returncode != 0:
            print("❌ Failed at:", m)
            sys.exit(r.returncode)
    print("\n✅ ALL DONE")
