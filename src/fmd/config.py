"""Central settings for the project.

Everything that you might want to change (file paths, random seed, model
settings) lives here, so the rest of the code never hard-codes them.
"""
from pathlib import Path

# Folder layout -------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
KNOWLEDGE_BASE = ROOT / "data" / "knowledge_base"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
LOGS_DIR = ROOT / "logs"

FINFACT_PATH = DATA_RAW / "finfact.json"
SMS_PATH = DATA_RAW / "sms_spam.tsv"

# The two problems this project solves -------------------------------------
#   "claims"   -> is a financial claim true or false?  (misinformation)
#   "messages" -> is a text message a scam or legit?    (fraud / scams)
TASKS = ("claims", "messages")

# Shortcut fix: in Fin-Fact, 94% of PolitiFact claims are "true" and 71% of
# Snopes claims are "false". A model can score F1 0.82 just by recognising
# which website wrote the claim. Training on one fact-checker removes that
# shortcut. See scripts/check_shortcut.py and docs/HOW_IT_WAS_BUILT.md.
CLAIM_SOURCES = ("snopes",)

# Metric used to pick / tune models. For claims, 71% are false, so a model
# that always says "false" already gets F1 0.83 - F1 is misleading there.
# ROC-AUC measures whether the model actually ranks false claims above true ones.
SELECTION_METRIC = {"claims": "roc_auc", "messages": "f1"}

# Reproducibility: the same seed gives the same split and the same models.
RANDOM_STATE = 42
TEST_SIZE = 0.20  # 20% of the data is held back and never used for training

# Label meaning (same for both tasks so the API is consistent)
#   1 = deceptive (false claim / scam message)
#   0 = legitimate (true claim / normal message)
LABEL_NAMES = {0: "legitimate", 1: "deceptive"}

# Text -> numbers settings
TFIDF_MAX_FEATURES = 5000
EMBEDDING_DIM = 200  # size of the LSA "embedding" used when no neural model is available

# Default probability cut-off. Training replaces it with a tuned, per-task
# value (stored in reports/<task>_report.json). Scores within REVIEW_MARGIN of
# the cut-off are marked "needs human review".
DECISION_THRESHOLD = 0.5
REVIEW_MARGIN = 0.15
