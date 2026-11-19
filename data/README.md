# Data

`data/raw/` is filled by `python scripts/download_data.py` (it is git-ignored):

| File | Source | License |
|---|---|---|
| `finfact.json` | [Fin-Fact](https://github.com/IIT-DM/Fin-Fact) (Rangapur et al., 2023) | MIT |
| `sms_spam.tsv` | [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) (Almeida & Gómez Hidalgo, 2011), via a GitHub mirror | CC BY 4.0 |

`knowledge_base/scam_patterns.json` is a hand-written list of 10 scam / legitimate
message patterns, summarized from FTC consumer guidance and used as the RAG
knowledge base for the scam assistant. Add your own patterns to extend it.
