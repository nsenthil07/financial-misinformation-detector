"""Benchmark: tree model vs LLM-with-RAG vs LLM-without-RAG.

Answers the questions an AI governance review would ask:
  * Does the LLM actually beat the cheap classifier?
  * Does retrieval (RAG) help the LLM, or is it just decoration?
  * How often does the LLM refuse ("uncertain"), and what does it cost in latency?

Requires a real LLM provider, e.g.
    ollama pull llama3.2 && FMD_LLM_PROVIDER=ollama python scripts/evaluate_llm.py --n 100
    ANTHROPIC_API_KEY=... python scripts/evaluate_llm.py --n 100

Results are written to reports/llm_benchmark_<task>.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmd import config, data, evaluate, llm  # noqa: E402
from fmd.detector import Detector, red_flags  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="claims", choices=config.TASKS)
ap.add_argument("--n", type=int, default=100, help="test items to evaluate (LLM calls cost money/time)")
ap.add_argument("--provider", default=None)
args = ap.parse_args()

provider = llm.choose_provider(args.provider)
if provider == "offline":
    sys.exit("No LLM configured. Set FMD_LLM_PROVIDER=ollama (free, local) or an API key - see README.")

det = Detector(log=False)
_, test_df = data.split(data.load_task(args.task))
sample = test_df.groupby("label", group_keys=False).apply(
    lambda g: g.sample(min(len(g), args.n // 2), random_state=config.RANDOM_STATE))
store = det.stores[args.task]
y = sample["label"].to_numpy()

# hybrid      = what the API does: tree score + red flags + retrieved evidence -> LLM
# llm_no_rag  = same LLM and prompt, but no evidence and no tree score (LLM alone)
rows = {"tree_model": [], "hybrid": [], "llm_no_rag": []}
lat = {"hybrid": [], "llm_no_rag": []}
to_score = {"deceptive": 1.0, "legitimate": 0.0, "uncertain": 0.5}
for i, text in enumerate(sample["text"], 1):
    p = float(det.models[args.task].predict_proba([text])[0, 1])
    rows["tree_model"].append(p)
    flags = red_flags(text)
    for name, ev, score in (("hybrid", store.search(text, k=4), p), ("llm_no_rag", [], None)):
        t0 = time.perf_counter()
        a = llm.assess(text, args.task, score, flags, ev, provider)
        lat[name].append(time.perf_counter() - t0)
        rows[name].append(to_score[a["verdict"]] if not a["error"] else np.nan)
    print(f"\r{i}/{len(sample)}", end="", flush=True)
print()

out = {"task": args.task, "provider": provider, "n": len(sample)}
for name, scores in rows.items():
    s = np.asarray(scores, float)
    ok = ~np.isnan(s)
    out[name] = {k: round(v, 4) for k, v in evaluate.metrics(y[ok], s[ok]).items()}
    if name != "tree_model":
        out[name]["uncertain_rate"] = round(float(np.mean(s[ok] == 0.5)), 4)
        out[name]["error_rate"] = round(float(1 - ok.mean()), 4)
        out[name]["mean_latency_s"] = round(float(np.mean(lat[name])), 3)
path = config.REPORTS_DIR / f"llm_benchmark_{args.task}.json"
path.write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
