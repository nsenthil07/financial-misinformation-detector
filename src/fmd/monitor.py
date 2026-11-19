"""Step 7 - Observability: log every prediction and watch for drift.

Why? A model that was 97% accurate at launch can quietly get worse when
scammers change their wording ("data drift"). We can't notice unless we
record what the model sees and does.

Privacy by design: the log stores a SHA-256 hash of the text, never the
text itself.

Drift metric - Population Stability Index (PSI), widely used in banking:
    PSI < 0.10   stable
    0.10 - 0.25  moderate shift, investigate
    > 0.25       significant shift, consider retraining
"""
import hashlib
import json
from datetime import datetime, timezone

import numpy as np

from . import config

LOG_PATH = config.LOGS_DIR / "predictions.jsonl"

# PSI is noisy on small samples: even a random sample of the reference data
# can exceed 0.25 at n=60. At n=200 the 95th percentile is about 0.08, so we
# wait for 200 predictions before judging.
MIN_DRIFT_SAMPLES = 200


def log_prediction(task, text, score, label, needs_review, latency_ms, model,
                   llm_provider=None, llm_error=None, path=LOG_PATH):
    """Append one prediction to the log file (one JSON object per line)."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task": task,
        "model": model,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "text_length": len(text),
        "score": round(float(score), 4),
        "label": label,
        "needs_review": needs_review,
        "latency_ms": round(latency_ms, 2),
        "llm_provider": llm_provider,
        "llm_error": llm_error,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record


def read_log(path=LOG_PATH):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def psi(reference, current, bins=10):
    """Population Stability Index between two lists of numbers."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if len(reference) == 0 or len(current) == 0:
        return float("nan")
    # bin edges = deciles of the reference data
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    n_bins = len(edges) - 1
    # +0.5 per bin (smoothing): an empty bin would otherwise give log(0)
    ref_share = (np.histogram(reference, edges)[0] + 0.5) / (len(reference) + 0.5 * n_bins)
    cur_share = (np.histogram(current, edges)[0] + 0.5) / (len(current) + 0.5 * n_bins)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def drift_status(value):
    if np.isnan(value):
        return "not enough data"
    if value < 0.10:
        return "stable"
    if value < 0.25:
        return "moderate shift"
    return "significant shift"


def drift_report(task, logs):
    """Compare logged predictions for one task with the test-set reference
    saved at training time."""
    report_file = config.REPORTS_DIR / f"{task}_report.json"
    reference = json.loads(report_file.read_text())["reference"]
    recent = [r for r in logs if r["task"] == task]
    out = {"task": task, "n_logged": len(recent)}
    enough = len(recent) >= MIN_DRIFT_SAMPLES
    if not enough:
        out["note"] = f"need at least {MIN_DRIFT_SAMPLES} logged predictions for a meaningful drift estimate"

    checks = {
        "score": (reference["scores"], [r["score"] for r in recent]),
        "text_length": (reference["lengths"], [r["text_length"] for r in recent]),
    }
    for name, (ref_values, cur_values) in checks.items():
        value = psi(ref_values, cur_values) if enough else float("nan")
        out[f"psi_{name}"] = None if np.isnan(value) else round(value, 4)
        out[f"status_{name}"] = drift_status(value)

    if recent:
        out["flag_rate"] = round(float(np.mean([r["label"] == "deceptive" for r in recent])), 4)
        out["review_rate"] = round(float(np.mean([r["needs_review"] for r in recent])), 4)
        out["p95_latency_ms"] = round(float(np.percentile([r["latency_ms"] for r in recent], 95)), 2)
        out["llm_error_rate"] = round(float(np.mean([bool(r["llm_error"]) for r in recent])), 4)
    return out
