"""Step 2 - Turn text into numbers (feature engineering).

Tree models can't read words, so every text becomes a row of numbers.
We build three kinds of features and later compare them:

1. Words (TF-IDF)  - one column per word or two-word phrase, weighted by how
                     distinctive it is ("claim prize" high, "the" near zero).
2. Red flags       - 15 counts a fraud analyst would look for: links, phone
                     numbers, money amounts, urgency words, ALL CAPS, ...
3. Embeddings      - a short list of numbers that captures the MEANING of the
                     text, so "you won cash" and "claim your reward" end up close.
                     Default: LSA (works offline). Optional: a small neural
                     sentence-transformer model (all-MiniLM-L6-v2).

`build_features()` glues the chosen parts together with scikit-learn's
FeatureUnion. Because it is a normal scikit-learn step, it can sit inside a
Pipeline, and cross-validation re-learns the vocabulary inside every fold
(so no information leaks from validation data into training).
"""
import re

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import FunctionTransformer, Normalizer, StandardScaler

from . import config

# ---------------------------------------------------------------------------
# 2. Red-flag features
# ---------------------------------------------------------------------------
URGENCY_WORDS = r"\b(urgent|immediately|now|today|expire[sd]?|final|last chance|act fast|asap|limited|hurry|within \d+)\b"
MONEY_WORDS = r"\b(cash|prize|won|win|winner|reward|free|bonus|claim|guaranteed|profit|returns?|invest\w*|crypto|bitcoin|loan|debt|credit|refund)\b"
ACTION_WORDS = r"\b(call|text|reply|click|visit|send|txt|verify|confirm|login|log in|update)\b"
ACCOUNT_WORDS = r"\b(account|bank|card|pin|password|ssn|suspend\w*|locked|security|unauthori[sz]ed)\b"
SUPERLATIVES = r"\b(never|always|every|all|nobody|everyone|biggest|largest|record|only|first time|ever)\b"

RED_FLAG_NAMES = [
    "n_chars", "n_words", "caps_ratio", "n_exclaims", "is_question", "has_url",
    "has_long_number", "n_currency", "n_numbers", "has_percent",
    "n_urgency", "n_money_words", "n_action_words", "n_account_words", "n_superlatives",
]


def red_flag_row(text):
    """Compute the 15 red-flag numbers for one text (same order as RED_FLAG_NAMES)."""
    low = text.lower()
    letters = [c for c in text if c.isalpha()]
    caps_ratio = sum(c.isupper() for c in letters) / max(len(letters), 1)
    return [
        len(text),
        len(text.split()),
        caps_ratio,
        text.count("!"),
        int(text.rstrip().endswith("?")),
        int(bool(re.search(r"https?://|www\.|\.com\b|\.co\.uk\b", low))),
        int(bool(re.search(r"\d{5,}", text))),  # phone numbers / short codes
        len(re.findall(r"[$£€]|\busd\b|\bdollars?\b|\bpounds?\b", low)),
        len(re.findall(r"\d+(?:[.,]\d+)?", text)),
        int("%" in text or "percent" in low),
        len(re.findall(URGENCY_WORDS, low)),
        len(re.findall(MONEY_WORDS, low)),
        len(re.findall(ACTION_WORDS, low)),
        len(re.findall(ACCOUNT_WORDS, low)),
        len(re.findall(SUPERLATIVES, low)),
    ]


def red_flag_features(texts):
    """Red-flag numbers for many texts -> 2-D array (rows = texts)."""
    return np.array([red_flag_row(t) for t in texts], dtype=float)


# ---------------------------------------------------------------------------
# 3. Embeddings
# ---------------------------------------------------------------------------
_neural_model = None


def neural_embeddings(texts):
    """Encode texts with a small transformer model (needs `sentence-transformers`
    and an internet connection the first time, to download ~90 MB)."""
    global _neural_model
    if _neural_model is None:
        from sentence_transformers import SentenceTransformer

        _neural_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _neural_model.encode(list(texts), normalize_embeddings=True)


def make_embeddings(kind="lsa", n_dims=config.EMBEDDING_DIM):
    """kind="lsa": TF-IDF -> TruncatedSVD (Latent Semantic Analysis) -> unit length.
    kind="sentence-transformer": the neural model above."""
    if kind == "sentence-transformer":
        return FunctionTransformer(neural_embeddings)
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        TruncatedSVD(n_components=n_dims, random_state=config.RANDOM_STATE),
        Normalizer(),
    )


# ---------------------------------------------------------------------------
# Putting it together
# ---------------------------------------------------------------------------
# The four feature sets compared in training. "interpretable" (words + red
# flags) is used only for readable rules/charts, because a column called
# "emb:17" means nothing to a person.
FEATURE_SETS = ["tfidf", "handcrafted", "embeddings", "all"]
PARTS = {
    "tfidf": ["words"],
    "handcrafted": ["red_flags"],
    "embeddings": ["embeddings"],
    "all": ["words", "red_flags", "embeddings"],
    "interpretable": ["words", "red_flags"],
}


def build_features(feature_set="all", embeddings="lsa", n_dims=config.EMBEDDING_DIM):
    """Return a scikit-learn FeatureUnion for the chosen feature set."""
    available = {
        "words": TfidfVectorizer(ngram_range=(1, 2), max_features=config.TFIDF_MAX_FEATURES,
                                 min_df=2, sublinear_tf=True),
        "red_flags": make_pipeline(FunctionTransformer(red_flag_features), StandardScaler()),
        "embeddings": make_embeddings(embeddings, n_dims),
    }
    return FeatureUnion([(name, available[name]) for name in PARTS[feature_set]])


def feature_names(pipeline):
    """Human-readable column names ("word:prize", "flag:has_url", "emb:3")
    for a fitted Pipeline whose first step is called "features"."""
    names = []
    for part_name, part in pipeline.named_steps["features"].transformer_list:
        if part_name == "words":
            names += ["word:" + w for w in part.get_feature_names_out()]
        elif part_name == "red_flags":
            names += ["flag:" + n for n in RED_FLAG_NAMES]
        else:
            n_emb = part.transform(["x"]).shape[1]
            names += [f"emb:{i}" for i in range(n_emb)]
    return names
