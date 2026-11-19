"""Governance finding: the 'which fact-checker wrote this?' shortcut.

Reproduces why the claims model is trained on Snopes only. On the mixed
dataset, a rule that ignores the claim entirely and only looks at the
source scores about the same F1 as real models - so high F1 there would
prove nothing.

Run: python scripts/check_shortcut.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sklearn.pipeline import Pipeline  # noqa: E402

from fmd import data, evaluate, models  # noqa: E402
from fmd.features import build_features  # noqa: E402

df = data.load_claims(sources=None)
print(df.groupby("source")["label"].agg(share_false="mean", n="size").round(3), "\n")
tr, te = data.split(df)

rate = tr.groupby("source")["label"].mean()
shortcut = te["source"].map(rate).to_numpy()
print("source-only rule  :", {k: round(v, 3) for k, v in evaluate.metrics(te["label"], shortcut).items()})

est, _ = models.build("random_forest")
pipe = Pipeline([("features", build_features("tfidf")), ("model", est)]).fit(tr["text"], tr["label"])
p = pipe.predict_proba(te["text"])[:, 1]
print("random forest     :", {k: round(v, 3) for k, v in evaluate.metrics(te["label"], p).items()})

# Can the model tell the sources apart from wording alone? If yes, it can use the shortcut.
pipe_src = Pipeline([("features", build_features("tfidf")), ("model", models.build("random_forest")[0])])
pipe_src.fit(tr["text"], (tr["source"] == "snopes").astype(int))
acc = np.mean(pipe_src.predict(te["text"]) == (te["source"] == "snopes").astype(int))
print(f"\nwording alone predicts the fact-checker with {acc:.1%} accuracy -> the shortcut is learnable")
