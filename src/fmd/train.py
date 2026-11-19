"""Train, compare and save all models.

Run:  python -m fmd.train            (full run, about 15-25 min on a laptop)
      python -m fmd.train --quick    (fewer tuning rounds, about 5 min)

For each task ("claims" and "messages") the steps are:

  A. Feature experiment: every model x every feature set is scored with
     3-fold cross-validation on the TRAINING data only.
  B. Tuning: each model is tuned with RandomizedSearchCV, using the feature
     set that worked best for it in step A.
  C. Test: each tuned model is scored ONCE on the held-out test set. The model
     with the best CROSS-VALIDATED score is chosen (not the best test score,
     so the test result stays honest).
  D. Threshold: the probability cut-off is tuned on training data.
  E. Reports: tables, charts, readable tree rules, fairness slices, and a
     reference score distribution for drift monitoring.
  F. Save the chosen model and the vector store used by the LLM assistant.

Before the tasks, a small "router" model is trained to tell claims from messages.
"""
import argparse
import json
import platform
import time
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline

from . import config, data, evaluate, models
from .features import FEATURE_SETS, build_features
from .rag import VectorStore, load_scam_patterns

warnings.filterwarnings("ignore", category=UserWarning)

CV = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.RANDOM_STATE)


def make_model_pipeline(model_name, feature_set, pos_weight, embeddings):
    """Pipeline = features step + model step. Returns it with the model's
    search space (renamed to 'model__<setting>' as scikit-learn expects)."""
    model, search_space = models.build(model_name, pos_weight)
    pipeline = Pipeline([("features", build_features(feature_set, embeddings)), ("model", model)])
    search_space = {f"model__{name}": values for name, values in search_space.items()}
    return pipeline, search_space


# ---------------------------------------------------------------------------
# A. Which features help each model?
# ---------------------------------------------------------------------------
def compare_feature_sets(X, y, metric, pos_weight, embeddings):
    rows = []
    for feature_set in FEATURE_SETS:
        for model_name in models.MODEL_NAMES:
            pipeline, _ = make_model_pipeline(model_name, feature_set, pos_weight, embeddings)
            scores = cross_val_score(pipeline, X, y, cv=CV, scoring=metric)
            rows.append({"model": model_name, "feature_set": feature_set,
                         "cv_mean": scores.mean(), "cv_std": scores.std()})
            print(f"  [A] {model_name:18s} {feature_set:12s} {metric} {scores.mean():.3f} ± {scores.std():.3f}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# B + C. Tune each model, then test it once
# ---------------------------------------------------------------------------
def tune_and_test(X_train, y_train, X_test, y_test, best_feature_set, metric, pos_weight, embeddings, quick):
    results, fitted, test_scores = [], {}, {}
    for model_name in models.MODEL_NAMES:
        feature_set = best_feature_set[model_name]
        pipeline, search_space = make_model_pipeline(model_name, feature_set, pos_weight, embeddings)
        search = RandomizedSearchCV(pipeline, search_space, n_iter=4 if quick else 12, scoring=metric,
                                    cv=CV, random_state=config.RANDOM_STATE)
        start = time.perf_counter()
        search.fit(X_train, y_train)
        seconds = time.perf_counter() - start

        best = search.best_estimator_
        proba = best.predict_proba(X_test)[:, 1]
        row = evaluate.metrics(y_test, proba)
        row.update({
            "model": model_name,
            "feature_set": feature_set,
            "cv_tuned": search.best_score_,
            "tuning_seconds": seconds,
            "latency_ms_per_item": evaluate.latency_ms_per_item(best, X_test[:500]),
            "best_params": {k.replace("model__", ""): (v.item() if hasattr(v, "item") else v)
                            for k, v in search.best_params_.items()},
        })
        results.append(row)
        fitted[model_name] = best
        test_scores[model_name] = proba
        print(f"  [C] {model_name:18s} ({feature_set}) test F1 {row['f1']:.3f}  AUC {row['roc_auc']:.3f}  "
              f"bal-acc {row['balanced_accuracy']:.3f}  ({seconds:.0f}s)")
    results = pd.DataFrame(results).sort_values("cv_tuned", ascending=False)
    return results, fitted, test_scores


# ---------------------------------------------------------------------------
# D. Threshold tuning
# ---------------------------------------------------------------------------
def tune_threshold(task, pipeline, X_train, y_train):
    """Pick the probability cut-off using out-of-fold predictions on TRAINING data.

    0.5 is only a default. With 71% false claims the scores sit high, so 0.5
    flags almost everything. We pick the cut-off that maximises the task's
    goal: balanced accuracy for claims, F1 for scams."""
    oof_proba = cross_val_predict(clone(pipeline), X_train, y_train, cv=CV, method="predict_proba")[:, 1]
    score_fn = balanced_accuracy_score if task == "claims" else f1_score
    candidates = np.round(np.arange(0.10, 0.91, 0.01), 2)
    scores = [score_fn(y_train, (oof_proba >= t).astype(int)) for t in candidates]
    best = int(np.argmax(scores))
    return {"threshold": float(candidates[best]), "objective": score_fn.__name__,
            "oof_objective": float(scores[best])}


# ---------------------------------------------------------------------------
# E. Readable explanations and fairness slices
# ---------------------------------------------------------------------------
def interpretable_models(task, train_df, test_df, pos_weight):
    """Embedding columns ("emb:17") can't be explained to a person, so we also
    fit two models on words + red flags only:
      * a depth-4 decision tree  -> if/then rules anyone can read
      * a random forest          -> a meaningful importance chart"""
    X_train, y_train = train_df["text"].tolist(), train_df["label"].to_numpy()
    X_test, y_test = test_df["text"].tolist(), test_df["label"].to_numpy()

    tree, _ = make_model_pipeline("decision_tree", "interpretable", pos_weight, "lsa")
    tree.set_params(model__max_depth=4, model__min_samples_leaf=10).fit(X_train, y_train)
    forest, _ = make_model_pipeline("random_forest", "interpretable", pos_weight, "lsa")
    forest.fit(X_train, y_train)

    scores = {
        "shallow_tree_depth4": evaluate.metrics(y_test, tree.predict_proba(X_test)[:, 1]),
        "random_forest_words_flags": evaluate.metrics(y_test, forest.predict_proba(X_test)[:, 1]),
    }
    importance = evaluate.top_importances(forest)
    evaluate.plot_importances(importance, f"{task}: most important words & red flags (random forest)",
                              config.FIGURES_DIR / f"{task}_importances_interpretable.png")
    header = (f"# {task}: depth-4 decision tree on words + red flags (readable, not the production model)\n"
              f"# 'word:x' = TF-IDF weight of word x; 'flag:x' = standardised red-flag count "
              f"(> 0 means above average)\n# class 1 = deceptive, class 0 = legitimate\n"
              f"# test ROC-AUC of this tree: {scores['shallow_tree_depth4']['roc_auc']:.3f}\n\n")
    (config.REPORTS_DIR / f"{task}_decision_tree_rules.txt").write_text(header + evaluate.tree_rules(tree))
    return {"metrics": scores, "top_features": importance.to_dict(orient="records")}


def slice_tables(test_df, proba, threshold):
    """Does the model work equally well for every topic / length of text?"""
    tables = {}
    for column in ("source", "topic"):
        if test_df[column].nunique() > 1:
            tables[column] = evaluate.slice_metrics(test_df, proba, column, threshold=threshold)
    length = pd.cut(test_df["text"].str.len(), [0, 60, 120, 200, 10_000],
                    labels=["<60 chars", "60-120", "120-200", ">200"]).astype(str)
    tables["length"] = evaluate.slice_metrics(test_df.assign(length=length), proba, "length",
                                              min_size=10, threshold=threshold)
    return {name: table.to_dict(orient="records") for name, table in tables.items()}


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
def train_router():
    """Decides which model a text goes to: 'claims' (statements about the
    world) or 'messages' (texts sent to a person). Logistic regression on
    character n-grams is fast and very good at telling writing styles apart."""
    claims = data.load_claims(sources=None)["text"]  # both fact-checkers: more claim styles
    messages = data.load_messages()["text"]
    df = pd.DataFrame({"text": pd.concat([claims, messages]),
                       "label": [0] * len(claims) + [1] * len(messages)})
    train_df, test_df = data.split(df)
    router = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3, sublinear_tf=True)),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", C=3.0)),
    ])
    router.fit(train_df["text"], train_df["label"])
    accuracy = float((router.predict(test_df["text"]) == test_df["label"]).mean())
    joblib.dump(router, config.MODELS_DIR / "router.joblib", compress=3)
    print(f"router test accuracy (claim vs message): {accuracy:.3f}")
    return {"test_accuracy": accuracy, "n_test": len(test_df)}


# ---------------------------------------------------------------------------
# One full task
# ---------------------------------------------------------------------------
def run_task(task, quick, embeddings):
    print(f"\n{'=' * 70}\nTASK: {task}\n{'=' * 70}")
    df = data.load_task(task)
    train_df, test_df = data.split(df)
    X_train, y_train = train_df["text"].tolist(), train_df["label"].to_numpy()
    X_test, y_test = test_df["text"].tolist(), test_df["label"].to_numpy()
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()  # how much rarer the deceptive class is
    metric = config.SELECTION_METRIC[task]
    print(f"rows {len(df)}, train {len(train_df)}, test {len(test_df)}, "
          f"deceptive {df['label'].mean():.1%}, selection metric {metric}")

    # A. feature experiment
    cv_table = compare_feature_sets(X_train, y_train, metric, pos_weight, embeddings)
    best_rows = cv_table.loc[cv_table.groupby("model")["cv_mean"].idxmax()]
    best_feature_set = dict(zip(best_rows["model"], best_rows["feature_set"]))

    # B + C. tune and test
    results, fitted, test_scores = tune_and_test(X_train, y_train, X_test, y_test, best_feature_set,
                                                 metric, pos_weight, embeddings, quick)
    best_name = results.iloc[0]["model"]  # best CROSS-VALIDATED score
    best_pipeline = fitted[best_name]
    print(f"  chosen model: {best_name}")

    # Baselines every model must beat
    majority_class = int(y_train.mean() >= 0.5)  # always predict the most common label
    baselines = {"majority_class": evaluate.metrics(y_test, np.full(len(y_test), 0.99 if majority_class else 0.01))}
    baselines["majority_class"]["roc_auc"] = 0.5

    # F (part 1). vector store for the LLM assistant
    if task == "claims":
        store = VectorStore(train_df[["text", "label", "evidence", "url"]], embeddings)
        # Retrieval-only baseline: copy the verdict of the 10 most similar past fact-checks.
        knn_scores = store.knn_vote(X_test, k=10)
        baselines["retrieval_knn_vote"] = evaluate.metrics(y_test, knn_scores)
        test_scores["retrieval kNN (RAG only)"] = knn_scores
        print(f"  [baseline] retrieval kNN vote: AUC {baselines['retrieval_knn_vote']['roc_auc']:.3f}")
    else:
        store = VectorStore(load_scam_patterns(), embeddings, extra_texts=X_train)

    # D. threshold
    threshold_info = tune_threshold(task, best_pipeline, X_train, y_train)
    threshold = threshold_info["threshold"]
    best_proba = test_scores[best_name]
    at_threshold = evaluate.metrics(y_test, best_proba, threshold)
    print(f"  tuned threshold {threshold:.2f} -> test "
          + ", ".join(f"{k} {v:.3f}" for k, v in at_threshold.items()))

    # E. reports and charts
    figures = config.FIGURES_DIR
    evaluate.plot_feature_set_comparison(cv_table, f"{task}: which features help each model? (3-fold CV {metric})",
                                         figures / f"{task}_feature_sets.png", metric)
    evaluate.plot_roc(y_test, test_scores, f"{task}: ROC curves on held-out test set", figures / f"{task}_roc.png")
    evaluate.plot_confusion(y_test, best_proba, f"{task}: {best_name} (threshold {threshold:.2f})",
                            figures / f"{task}_confusion.png", threshold)
    top_features = evaluate.top_importances(best_pipeline)
    evaluate.plot_importances(top_features, f"{task}: top features ({best_name})", figures / f"{task}_importances.png")

    report = {
        "task": task,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_rows": len(df), "n_train": len(train_df), "n_test": len(test_df),
        "deceptive_share": float(df["label"].mean()),
        "selection_metric": metric,
        "best_model": best_name,
        "best_feature_set": results.iloc[0]["feature_set"],
        "embeddings": embeddings,
        "threshold": threshold,
        "threshold_tuning": threshold_info,
        "test_metrics_at_threshold": at_threshold,
        "test_results": results.to_dict(orient="records"),
        "feature_set_cv": cv_table.to_dict(orient="records"),
        "baselines": baselines,
        "slices": slice_tables(test_df, best_proba, threshold),
        "top_features": top_features.to_dict(orient="records"),
        "interpretable": interpretable_models(task, train_df, test_df, pos_weight),
        # reference distributions used later for drift monitoring
        "reference": {"scores": np.round(best_proba, 4).tolist(),
                      "lengths": test_df["text"].str.len().tolist()},
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__},
    }
    (config.REPORTS_DIR / f"{task}_report.json").write_text(json.dumps(report, indent=2, default=str))

    # F (part 2). save
    joblib.dump(best_pipeline, config.MODELS_DIR / f"{task}_best.joblib", compress=3)
    store.save(config.MODELS_DIR / f"{task}_vector_store.joblib")
    print(f"  saved models/{task}_best.joblib and models/{task}_vector_store.joblib")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="all", choices=["all", *config.TASKS])
    parser.add_argument("--quick", action="store_true", help="fewer tuning rounds")
    parser.add_argument("--embeddings", default="lsa", choices=["lsa", "sentence-transformer"])
    args = parser.parse_args()

    for folder in (config.MODELS_DIR, config.REPORTS_DIR, config.FIGURES_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    router_info = train_router()
    (config.REPORTS_DIR / "router_report.json").write_text(json.dumps(router_info, indent=2))
    tasks = config.TASKS if args.task == "all" else [args.task]
    for task in tasks:
        run_task(task, args.quick, args.embeddings)


if __name__ == "__main__":
    main()
