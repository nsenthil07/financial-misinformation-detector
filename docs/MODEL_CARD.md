# Model Card: Financial Misinformation & Scam Detector

*Format follows Mitchell et al., "Model Cards for Model Reporting" (2019).*

## Model details
| | Claims model (misinformation) | Messages model (scams) |
|---|---|---|
| Algorithm | Random forest | XGBoost (gradient boosting) |
| Features | LSA text embeddings (200-d) | TF-IDF + 15 red-flag features + LSA embeddings |
| Selected by | 3-fold CV ROC-AUC on training data | 3-fold CV F1 on training data |
| Decision threshold | 0.69 (tuned for balanced accuracy, out-of-fold) | 0.37 (tuned for F1, out-of-fold) |
| Human-review band | threshold ± 0.15 | threshold ± 0.15 |
| Explanation layer | RAG over 1,626 training-split fact-checks + LLM | RAG over 10 FTC-based scam patterns + LLM |

A character n-gram logistic regression **router** (97.0% test accuracy) sends
input to the right model when the caller doesn't specify one.

Version 1.0.0, trained October 2026. Python 3.13, scikit-learn 1.9, XGBoost, LightGBM.
Features are built with scikit-learn `FeatureUnion` inside a `Pipeline` (`src/fmd/features.py`).

## Intended use
* **Intended:** research and portfolio demonstration; decision *support* for
  fraud or member-protection analysts, who see a score, red flags, evidence and
  an explanation and then decide.
* **Not intended:** automatically blocking, deleting or penalizing a member's
  messages or accounts; judging the truth of claims without human review;
  financial advice; content moderation of political speech.

## Training data
| | Fin-Fact (Rangapur et al., 2023, MIT) | UCI SMS Spam Collection (Almeida & Gómez Hidalgo, 2011, CC BY 4.0) |
|---|---|---|
| Rows used | 2,033 Snopes claims (NEI rows and PolitiFact excluded, see below) | 5,160 unique messages |
| Deceptive share | 71.1% false | 12.4% spam |
| Split | 80/20 stratified, seed 42 | 80/20 stratified, seed 42 |
| Era / locale | Mostly US, 2010s–2023 | UK and Singapore SMS, ~2011 |

## Evaluation (held-out test set, never used for training or selection)

**Messages** (n = 1,032), threshold 0.37

| Metric | Value | Majority baseline |
|---|---|---|
| F1 | **0.972** | 0.000 |
| Precision | 0.992 | n/a |
| Recall | 0.953 | 0 |
| Balanced accuracy | 0.976 | 0.500 |
| ROC-AUC | 0.999 | 0.500 |

**Claims** (n = 407), threshold 0.69

| Metric | Value | Majority baseline | Retrieval-only kNN |
|---|---|---|---|
| ROC-AUC | **0.664** | 0.500 | 0.645 |
| Balanced accuracy | 0.616 | 0.500 | 0.587 (at 0.5) |
| Precision | 0.799 | 0.710 | 0.752 |
| Recall | 0.606 | 1.000 | 0.903 |

**How to read this:** the scam model is strong. The claims model is only
**modestly** better than chance. Wording alone is weak evidence of whether a
financial claim is true, which is why claims are always routed to the
evidence-based assistant and frequently to human review.

## Governance findings

1. **Source shortcut (found and mitigated).** In full Fin-Fact, 94% of
   PolitiFact claims are true and 71% of Snopes claims are false. Wording
   alone identifies the fact-checker with 94.9% accuracy, and a rule that uses
   *only* the source gets F1 0.818, the same as a real random forest (0.795).
   Mitigation: train on a single fact-checker; report ROC-AUC and balanced
   accuracy instead of F1. Reproduce: `python scripts/check_shortcut.py`.
2. **Slice performance, claims (at threshold 0.69).** False-positive rates vary
   widely by topic: "share" claims 0.62, "profit" 0.53, "credit" 0.00. Topic
   groups are small (26–57 claims), so these estimates are noisy, but the
   model shouldn't be trusted equally across topics.
3. **Slice performance, messages.** Very short scam messages (< 60 characters)
   had a false-negative rate of 0.20, but only ~5 such scams are in the test
   set. Short scams are a known blind spot to monitor.
4. **Geographic and era bias.** The SMS data is from 2011 UK/Singapore
   networks. Words like "uk", "mobile" and premium short codes are among the
   strongest signals. Modern US scams (crypto, impersonation, delivery,
   "toll" texts) are under-represented, so the model needs fresh data before
   any real deployment.
5. **Routing errors.** Some legitimate notices (e.g. "Your payment of $45 was
   received") read like statements and are routed to the claims model, which
   flags them for review. This fails safe (human review), but adds workload.

## Explainability
* Readable depth-4 decision-tree rules: `reports/*_decision_tree_rules.txt`
* Importance of words and red flags: `reports/figures/*_importances_interpretable.png`
* Per-request: triggered red flags, influential words, retrieved evidence with
  similarity scores and source URLs, and an LLM explanation with citations.

## LLM component risks and controls
| Risk | Control |
|---|---|
| Hallucinated facts or sources | Answer only from numbered evidence; must cite; "uncertain" allowed; temperature 0 |
| Prompt injection in member text | Text isolated in `<content>` tags and declared untrusted; tree score is computed independently of the LLM |
| Malformed output | JSON schema validated and clamped (`parse_response`); unit-tested |
| Provider outage | Automatic fallback to template explanation; error logged and surfaced in `/v1/monitoring` |
| Data leaving the institution | Default `auto` → local Ollama or offline when no API key; hosted APIs are opt-in |

*The LLM's accuracy has **not** been benchmarked in the published results
(no LLM was available in the build environment). Run
`scripts/evaluate_llm.py` to measure the hybrid vs tree-only vs LLM-only.*

## Monitoring
Every prediction is logged with a SHA-256 hash instead of raw text. The
Population Stability Index on score and text-length distributions is reported
at `/v1/monitoring` once ≥ 200 predictions exist (at n = 200 the 95th
percentile PSI of same-distribution samples is ≈ 0.08; a simulated 40% scam
wave gives PSI ≈ 0.41). Also reported: flag rate, review rate, p95 latency and
LLM error rate.

## Caveats
* Embeddings in the published run are LSA. Neural sentence embeddings
  (MiniLM) are supported (`--embeddings sentence-transformer`) and may change
  the ranking of models.
* A random forest on words + red flags scored a higher test ROC-AUC on claims
  (0.73) than the CV-selected model (0.66). It was *not* swapped in, because
  choosing by test score would bias the result; it is a candidate for the
  next CV round.
* Scores are not calibrated probabilities.
