"""Step 3 - The classifiers (the HarvardX part).

Five tree-based models, from simplest to most powerful:

  decision_tree      one tree of if/then rules. Easy to read, but tends to
                     memorise the training data (overfit).
  random_forest      hundreds of trees, each trained on a random sample of
                     rows and features; they vote. Bagging reduces overfitting.
  gradient_boosting  scikit-learn's classic boosting: trees are added one at a
                     time, each fixing the errors of the ones before.
  xgboost            a fast, regularised boosting library (Kaggle favourite).
  lightgbm           Microsoft's boosting library; grows trees leaf-wise, very fast.

Each entry has a sensible default model plus a small search space used for
hyper-parameter tuning.
"""
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from . import config

SEED = config.RANDOM_STATE


def build(name: str, pos_weight: float = 1.0):
    """Return (estimator, search_space). `pos_weight` = (#legit / #deceptive)
    tells boosting models to pay more attention to the rarer class."""
    if name == "decision_tree":
        return (
            DecisionTreeClassifier(class_weight="balanced", max_depth=20, min_samples_leaf=2, random_state=SEED),
            {"max_depth": [5, 10, 20, 40, None], "min_samples_leaf": randint(1, 20),
             "criterion": ["gini", "entropy"]},
        )
    if name == "random_forest":
        return (
            RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                   n_jobs=-1, random_state=SEED),
            {"n_estimators": randint(200, 600), "max_depth": [None, 20, 40, 80],
             "max_features": ["sqrt", "log2", 0.05], "min_samples_leaf": randint(1, 5)},
        )
    if name == "gradient_boosting":
        return (
            GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3,
                                       max_features="sqrt", subsample=0.8, random_state=SEED),
            {"n_estimators": randint(100, 400), "learning_rate": loguniform(0.02, 0.3),
             "max_depth": randint(2, 6), "subsample": uniform(0.6, 0.4)},
        )
    if name == "xgboost":
        from xgboost import XGBClassifier

        return (
            XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6, subsample=0.8,
                          colsample_bytree=0.5, tree_method="hist", scale_pos_weight=pos_weight,
                          eval_metric="logloss", n_jobs=-1, random_state=SEED),
            {"n_estimators": randint(150, 600), "learning_rate": loguniform(0.02, 0.3),
             "max_depth": randint(3, 10), "subsample": uniform(0.6, 0.4),
             "colsample_bytree": uniform(0.2, 0.6), "reg_lambda": loguniform(0.1, 10)},
        )
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return (
            LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.5, class_weight="balanced",
                           n_jobs=-1, random_state=SEED, verbose=-1),
            {"n_estimators": randint(150, 600), "learning_rate": loguniform(0.02, 0.2),
             "num_leaves": randint(8, 64), "min_child_samples": randint(5, 40),
             "colsample_bytree": uniform(0.2, 0.6)},
        )
    raise ValueError(name)


MODEL_NAMES = ("decision_tree", "random_forest", "gradient_boosting", "xgboost", "lightgbm")
