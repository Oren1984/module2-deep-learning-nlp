# 03_hybrid_dl_nlp/src/stage9_report.py
# This module generates a final report summarizing experiment results.
# It compiles results from baseline and advanced models into a markdown file.

import joblib
from datetime import datetime
from src.stage0_frame import Config, ensure_dirs

# Helper to load results if file exists
def load_if_exists(path):
    return joblib.load(path) if path.exists() else None

# Main execution to generate final report
if __name__ == "__main__":
    cfg = Config()
    ensure_dirs(cfg)

    # Load results from baseline and experiments
    baseline = load_if_exists(cfg.results_dir / "baseline_results.joblib")
    A = load_if_exists(cfg.results_dir / "A_test_results.joblib")
    B = load_if_exists(cfg.results_dir / "B_test_results.joblib")

    # Compile report content 
    lines = []
    lines.append("# Hybrid DL + NLP — Final Report")
    lines.append(f"- Generated: {datetime.utcnow().isoformat()}Z")
    lines.append("")
    lines.append("## Experiments")
    lines.append("- A: TF-IDF + MLP (PyTorch)")
    lines.append("- B: GloVe Avg Embeddings + MLP (PyTorch)")
    lines.append("")
    
    # Summarize results 
    if baseline:
        lines.append("## Baseline (LogReg on TF-IDF)")
        lines.append(f"- Val accuracy: **{baseline['val_accuracy']:.4f}**")
        lines.append("")
        
    # Summarize Experiment A and B results
    if A:
        lines.append("## Experiment A نتائج")
        lines.append(f"- Test accuracy: **{A['test_acc']:.4f}**")
        lines.append("")
        lines.append("### Classification Report")
        lines.append("```")
        lines.append(A["report"])
        lines.append("```")
        lines.append("")
    if B:
        lines.append("## Experiment B نتائج")
        lines.append(f"- Test accuracy: **{B['test_acc']:.4f}**")
        lines.append("")
        lines.append("### Classification Report")
        lines.append("```")
        lines.append(B["report"])
        lines.append("```")
        lines.append("")
    
    # Handle missing experiments
    else:
        lines.append("## Experiment B")
        lines.append("- Not executed (missing GloVe or features)")
        lines.append("")

    # Save report to markdown file
    out = cfg.reports_dir / "final_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("✅ Report saved:", out)
