"""Step 8 - The two-stage workflow that ties everything together.

    text --> router (claim or message?) --> Stage 1: tree model --> score
                     |
                     |-- clearly legitimate (score < threshold - 0.15):
                     |       return straight away, no LLM cost
                     |
                     '-- flagged or uncertain:
                           retrieve evidence from the vector store
                           Stage 2: LLM writes an explanation with citations
                           within 0.15 of the threshold -> "needs human review"

The threshold is tuned for each task during training (train.tune_threshold).
"Cheap model first, expensive model only when needed" keeps cost and
latency low while every flag still gets an explanation.
"""
import json
import time

import joblib
import numpy as np

from . import config, llm, monitor
from .features import RED_FLAG_NAMES, feature_names, red_flag_row

# Plain-English names for the red flags shown to users.
FLAG_LABELS = {
    "has_url": "contains a link",
    "has_long_number": "contains a long phone/short-code number",
    "n_currency": "mentions money amounts",
    "n_urgency": "urgent / time-pressure language",
    "n_money_words": "prize, profit or 'free money' wording",
    "n_action_words": "asks you to call, text or click",
    "n_account_words": "mentions your account, card or security",
}


def red_flags(text):
    """List the warning signs present in one text, in plain English."""
    values = dict(zip(RED_FLAG_NAMES, red_flag_row(text)))
    flags = [label for key, label in FLAG_LABELS.items() if values[key] > 0]
    if values["n_exclaims"] > 1:
        flags.append("excessive exclamation marks")
    if values["caps_ratio"] > 0.5 and values["n_chars"] > 15:
        flags.append("mostly CAPITAL letters")
    return flags


def influential_words(pipeline, text, k=5):
    """Words in THIS text that the model found most important overall."""
    model = pipeline.named_steps["model"]
    parts = dict(pipeline.named_steps["features"].transformer_list)
    if "words" not in parts or not hasattr(model, "feature_importances_"):
        return []
    vectorizer = parts["words"]
    word_columns = vectorizer.transform([text]).indices  # which known words appear
    importance = np.asarray(model.feature_importances_)[: len(vectorizer.vocabulary_)]  # words come first
    names = feature_names(pipeline)
    ranked = sorted(word_columns, key=lambda i: importance[i], reverse=True)[:k]
    return [names[i].removeprefix("word:") for i in ranked if importance[i] > 0]


class Detector:
    def __init__(self, log=True):
        """Load everything that training saved in models/ and reports/."""
        if not (config.MODELS_DIR / "router.joblib").exists():
            raise FileNotFoundError("models missing - run `python -m fmd.train` first")
        self.log = log
        self.router = joblib.load(config.MODELS_DIR / "router.joblib")
        self.models, self.stores, self.info = {}, {}, {}
        for task in config.TASKS:
            self.models[task] = joblib.load(config.MODELS_DIR / f"{task}_best.joblib")
            self.stores[task] = joblib.load(config.MODELS_DIR / f"{task}_vector_store.joblib")
            report = json.loads((config.REPORTS_DIR / f"{task}_report.json").read_text())
            test = report["test_metrics_at_threshold"]
            self.info[task] = {
                "best_model": report["best_model"],
                "best_feature_set": report["best_feature_set"],
                "embeddings": report["embeddings"],
                "trained_at": report["trained_at"],
                "threshold": report["threshold"],
                "test_metrics": {m: round(test[m], 4) for m in
                                 ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc")},
            }

    def guess_task(self, text):
        """The router: a small classifier trained to tell claims from messages."""
        return "messages" if self.router.predict([text])[0] == 1 else "claims"

    def analyze(self, text, task="auto", explain=True, provider=None):
        start = time.perf_counter()
        text = text.strip()
        if task == "auto":
            task = self.guess_task(text)

        # Stage 1: tree model
        pipeline = self.models[task]
        score = float(pipeline.predict_proba([text])[0, 1])
        threshold = self.info[task]["threshold"]
        low = threshold - config.REVIEW_MARGIN
        high = threshold + config.REVIEW_MARGIN
        label = "deceptive" if score >= threshold else "legitimate"
        needs_review = low <= score <= high
        flags = red_flags(text)

        result = {
            "task": task,
            "label": label,
            "score": round(score, 4),
            "threshold": threshold,
            "needs_human_review": needs_review,
            "model": self.info[task]["best_model"],
            "red_flags": flags,
            "influential_terms": influential_words(pipeline, text),
        }

        # Stage 2: retrieval + LLM, only for flagged or uncertain text
        assistant = None
        if explain and score >= low:
            evidence = self.stores[task].search(text, k=4)
            assistant = llm.assess(text, task, score, flags, evidence, provider, threshold)
            result["assistant"] = assistant
            result["evidence"] = [
                {
                    "id": n,
                    "text": doc["text"][:300],
                    "rating": {1: "false / scam", 0: "true / legitimate"}.get(doc["label"]),
                    "similarity": round(doc["similarity"], 3),
                    "url": doc["url"] or None,
                }
                for n, doc in enumerate(evidence, start=1)
            ]

        result["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
        if self.log:
            monitor.log_prediction(task, text, score, label, needs_review, result["latency_ms"], result["model"],
                                   llm_provider=assistant["provider"] if assistant else None,
                                   llm_error=assistant["error"] if assistant else None)
        return result
