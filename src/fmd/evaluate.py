"""Step 4 - Measure how good the models are, and where they fail.

Why several metrics instead of just accuracy?
  * accuracy   - share of all predictions that are right. Misleading when one
                 class is rare: a model that says "legit" to every SMS is 87%
                 accurate and catches zero scams.
  * precision  - of the items we flagged as deceptive, how many really were?
                 (Low precision = we annoy members with false alarms.)
  * recall     - of all the deceptive items, how many did we catch?
                 (Low recall = scams slip through.)
  * F1         - one number that balances precision and recall.
  * balanced accuracy - average of recall on each class; 0.5 = no skill,
                 even when the classes are imbalanced.
  * ROC-AUC    - how well the scores rank deceptive above legit, across all
                 thresholds (0.5 = coin flip, 1.0 = perfect).
"""
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, average_precision_score,
    balanced_accuracy_score,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.tree import export_text

from . import config
from .features import feature_names


def metrics(y_true, proba, threshold=config.DECISION_THRESHOLD) -> dict:
    pred = (np.asarray(proba) >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
    }


def latency_ms_per_item(pipeline, texts, repeats=3) -> float:
    texts = list(texts)
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        pipeline.predict_proba(texts)
        best = min(best, time.perf_counter() - t0)
    return 1000 * best / len(texts)


def slice_metrics(test_df: pd.DataFrame, proba, column: str, min_size: int = 25,
                  threshold: float = config.DECISION_THRESHOLD) -> pd.DataFrame:
    """Governance check: does the model work equally well for every group?
    A big gap means the model may be unfair to (or blind to) one group."""
    df = test_df.assign(proba=proba, pred=(proba >= threshold).astype(int))
    rows = []
    for value, g in df.groupby(column):
        if len(g) < min_size:
            continue
        rows.append({
            column: value, "n": len(g), "share_deceptive": g["label"].mean(),
            "accuracy": accuracy_score(g["label"], g["pred"]),
            "false_positive_rate": ((g["pred"] == 1) & (g["label"] == 0)).sum() / max((g["label"] == 0).sum(), 1),
            "false_negative_rate": ((g["pred"] == 0) & (g["label"] == 1)).sum() / max((g["label"] == 1).sum(), 1),
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False)


# --- Explainability ---------------------------------------------------------
def top_importances(pipeline, k=20) -> pd.DataFrame:
    """Global feature importance from the tree model: how much each feature
    reduced impurity / loss across all splits."""
    importance = np.asarray(pipeline.named_steps["model"].feature_importances_, dtype=float)
    importance = importance / max(importance.sum(), 1e-12)
    names = feature_names(pipeline)
    top = np.argsort(importance)[::-1][:k]
    return pd.DataFrame({"feature": [names[i] for i in top], "importance": importance[top]})


def tree_rules(pipeline, max_depth=4) -> str:
    """Print a decision tree as human-readable if/then rules."""
    return export_text(pipeline.named_steps["model"], feature_names=feature_names(pipeline),
                       max_depth=max_depth, decimals=2)


# --- Plots ------------------------------------------------------------------
def plot_confusion(y_true, proba, title, path, threshold=config.DECISION_THRESHOLD):
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, (proba >= threshold).astype(int),
        display_labels=["legitimate", "deceptive"], cmap="Blues", ax=ax, colorbar=False,
    )
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(y_true, probas: dict, title, path):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for name, p in probas.items():
        RocCurveDisplay.from_predictions(y_true, p, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="random guess")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_importances(imp_df: pd.DataFrame, title, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    d = imp_df.iloc[::-1]
    ax.barh(d["feature"], d["importance"], color="#3b6fb6")
    ax.set_xlabel("relative importance")
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_feature_set_comparison(cv_df: pd.DataFrame, title, path, metric_label="F1"):
    pivot = cv_df.pivot(index="model", columns="feature_set", values="cv_mean")
    fig, ax = plt.subplots(figsize=(7, 4))
    pivot.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylabel(f"cross-validated {metric_label}")
    ax.set_xlabel("")
    ax.set_ylim(max(0, pivot.values.min() - 0.1), min(1, pivot.values.max() + 0.05))
    ax.set_title(title, fontsize=10)
    ax.legend(title="features", fontsize=7)
    plt.xticks(rotation=20, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
