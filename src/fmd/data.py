"""Step 1 - Load and clean the two datasets.

Each loader returns a pandas DataFrame with the same columns:

    text    the claim or message the model reads
    label   1 = deceptive, 0 = legitimate
    source  where the row came from (used later for fairness / slice checks)
    topic   a coarse topic tag (claims only; "sms" for messages)
    evidence  the fact-checker's explanation (claims only). NEVER used as a
              model feature, because it contains the answer. It is only used
              to build the knowledge base that the LLM assistant searches.
"""
import json
import re

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def _clean(text: str) -> str:
    """Light cleaning: fix whitespace. We deliberately keep punctuation,
    capital letters and symbols like '$' or '!' because scams use them a lot."""
    return re.sub(r"\s+", " ", str(text)).strip()


def load_claims(path=config.FINFACT_PATH, sources=config.CLAIM_SOURCES) -> pd.DataFrame:
    """Fin-Fact: financial claims checked by Snopes / PolitiFact.
    `sources=None` keeps both fact-checkers (used only to demonstrate the
    source shortcut)."""
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)

    records = []
    for r in rows:
        if r.get("label") not in ("true", "false"):
            continue  # drop "NEI" (not enough info): there is no ground truth
        issues = [i.lower() for i in (r.get("issues") or [])]
        records.append(
            {
                "text": _clean(r["claim"]),
                "label": 1 if r["label"] == "false" else 0,
                "source": "snopes" if "snopes.com" in r["url"] else "politifact",
                "topic": issues[0] if issues else "unknown",
                "evidence": _clean(" ".join(r.get("sci_digest") or []) or r.get("justification", "")[:600]),
                "url": r["url"],
            }
        )
    df = pd.DataFrame(records).drop_duplicates(subset="text").reset_index(drop=True)
    if sources:
        df = df[df["source"].isin(sources)].reset_index(drop=True)
    return df


def load_messages(path=config.SMS_PATH) -> pd.DataFrame:
    """UCI SMS Spam Collection: real text messages, spam vs ham."""
    df = pd.read_csv(path, sep="\t", header=None, names=["raw_label", "text"], quoting=3)
    df["text"] = df["text"].map(_clean)
    df["label"] = (df["raw_label"] == "spam").astype(int)
    df["source"] = "sms"
    df["topic"] = "sms"
    df["evidence"] = ""
    df["url"] = ""
    df = df.drop(columns="raw_label").drop_duplicates(subset="text").reset_index(drop=True)
    return df


def load_task(task: str) -> pd.DataFrame:
    if task == "claims":
        return load_claims()
    if task == "messages":
        return load_messages()
    raise ValueError(f"Unknown task {task!r}; choose from {config.TASKS}")


def split(df: pd.DataFrame):
    """Hold out a test set. `stratify` keeps the same share of deceptive rows
    in train and test, which matters because scams are only ~13% of messages."""
    return train_test_split(
        df, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=df["label"]
    )
