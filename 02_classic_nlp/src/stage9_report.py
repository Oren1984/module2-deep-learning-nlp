# 02_classic_nlp/src/stage9_report.py
# This script generates a final summary report of the SMS spam classification project.
# It compiles dataset statistics, model details, and evaluation metrics into a markdown file.

from __future__ import annotations

from pathlib import Path
import json
from unicodedata import name

PROJECT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT / "outputs" / "results"
REPORTS_DIR = PROJECT / "outputs" / "reports"
DOCS_DIR = PROJECT / "doc"


def load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = load_json(RESULTS_DIR / "final_metrics.json")
    eda = load_json(RESULTS_DIR / "eda_quick_summary.json")
    best_params = load_json(RESULTS_DIR / "best_params.json")
    improved_params = load_json(RESULTS_DIR / "improved_params.json")

    report_md = f"""# 📄 Final Summary — Classic NLP

## Goal
Binary SMS classification: **SPAM (1)** vs **HAM (0)** using classic NLP.

---

## 📊 Dataset — Quick EDA
- Total rows: {eda.get("rows", "N/A")}
- Spam rate: {eda.get("spam_rate", "N/A")}
- Avg words per message: {eda.get("avg_len_words", "N/A")}
- Median words per message: {eda.get("median_len_words", "N/A")}

---

## 🧠 Models
### Baseline
- TF-IDF (unigrams)
- Multinomial Naive Bayes

### Best Model
- TF-IDF + Linear SVM (GridSearch)

**Best parameters:**
```json
{json.dumps(best_params, indent=2)}

Improved parameters (optional stage):

{json.dumps(improved_params, indent=2)}

📈 Final Metrics (Test Set)
{json.dumps(metrics, indent=2)}

📦 Outputs

Confusion Matrix: outputs/figures/confusion_matrix.png

Classification Report: outputs/reports/classification_report.txt

Metrics JSON: outputs/results/final_metrics.json
"""

    out_path = DOCS_DIR / "final_summary.md"
    out_path.write_text(report_md, encoding="utf-8") 

    print("✅ Final report generated")
    print(f"Saved to: {out_path}")

if __name__ == "__main__":
    main()


