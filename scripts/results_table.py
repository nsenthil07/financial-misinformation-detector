"""Print the test-set results as Markdown tables (pasted into the README / model card)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmd import config  # noqa: E402

COLS = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "latency_ms_per_item"]

for task in config.TASKS:
    rep = json.loads((config.REPORTS_DIR / f"{task}_report.json").read_text())
    print(f"\n### {task}  (n_test={rep['n_test']}, selected by CV {rep['selection_metric']}, "
          f"best = {rep['best_model']}, embeddings = {rep['embeddings']})\n")
    print("| model | features | " + " | ".join(COLS) + " |")
    print("|---|---|" + "---|" * len(COLS))
    for r in sorted(rep["test_results"], key=lambda r: -r["cv_tuned"]):
        star = " **(selected)**" if r["model"] == rep["best_model"] else ""
        print(f"| {r['model']}{star} | {r['feature_set']} | " + " | ".join(f"{r[c]:.3f}" for c in COLS) + " |")
    for name, m in rep["baselines"].items():
        print(f"| *baseline: {name}* | - | " + " | ".join(f"{m.get(c, float('nan')):.3f}" for c in COLS[:-1]) + " | - |")
    print("\nFeature-set CV (mean):")
    import pandas as pd

    cv = pd.DataFrame(rep["feature_set_cv"]).pivot(index="model", columns="feature_set", values="cv_mean").round(3)
    print(cv.to_markdown())
