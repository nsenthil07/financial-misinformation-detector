"""Step 10 - A clickable demo for recruiters and reviewers.

Run:  streamlit run app/streamlit_app.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fmd import config, monitor  # noqa: E402
from fmd.detector import Detector  # noqa: E402

st.set_page_config(page_title="Financial Misinformation & Scam Detector", page_icon="🛡️", layout="wide")


@st.cache_resource
def load():
    return Detector()


det = load()
st.title("Financial Misinformation & Scam Detector")
st.caption("Stage 1: tree-based classifier (decision tree / random forest / boosting).  "
           "Stage 2: retrieval-augmented LLM assistant that explains the result with evidence.  "
           "Research prototype - not financial advice.")

tab_try, tab_models, tab_monitor = st.tabs(["Try it", "Model comparison", "Monitoring"])

EXAMPLES = {
    "Scam text": "URGENT! Your credit union account has been suspended. Verify your card now at www.secure-cu-verify.com or call 08001234567",
    "Normal text": "Hey, are we still on for dinner tonight? I can pick you up at 7",
    "Financial claim": "Did the IRS announce a new $5,000 stimulus payment for all taxpayers this month?",
    "Investment hype": "Guaranteed 40% monthly returns with our new crypto trading bot, act now before spots run out!",
}

with tab_try:
    c1, c2 = st.columns([2, 1])
    with c2:
        ex = st.radio("Load an example", list(EXAMPLES))
        task = st.selectbox("Task", ["auto", "messages", "claims"])
        provider = st.selectbox("LLM provider", ["auto", "offline", "ollama", "openai", "anthropic"])
    with c1:
        text = st.text_area("Paste a message or financial claim", EXAMPLES[ex], height=140)
        go = st.button("Analyze", type="primary")
    if go and text.strip():
        r = det.analyze(text, task, True, provider)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Verdict", r["label"].upper())
        m2.metric("Deceptiveness score", f"{r['score']:.2f}", help=f"flagging threshold {r['threshold']:.2f}")
        m3.metric("Needs human review", "yes" if r["needs_human_review"] else "no")
        m4.metric("Latency", f"{r['latency_ms']:.0f} ms")
        st.write(f"**Model:** {r['model']} ({r['task']} task)")
        if r["red_flags"]:
            st.warning("Red flags: " + ", ".join(r["red_flags"]))
        if r["influential_terms"]:
            st.write("**Influential words:** " + ", ".join(f"`{t}`" for t in r["influential_terms"]))
        if "assistant" in r:
            a = r["assistant"]
            st.subheader(f"Assistant ({a['provider']})")
            st.write(a["explanation"])
            if a.get("recommended_action"):
                st.info(a["recommended_action"])
            if a.get("error"):
                st.caption(f"LLM fell back to offline mode: {a['error']}")
            st.dataframe(pd.DataFrame(r["evidence"]), use_container_width=True, hide_index=True)
        else:
            st.success("Low risk - the LLM stage was skipped to save cost.")

with tab_models:
    for task in config.TASKS:
        rep = json.loads((config.REPORTS_DIR / f"{task}_report.json").read_text())
        st.subheader(f"{task}: test-set results (best = {rep['best_model']})")
        cols = ["model", "feature_set", "accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc",
                "latency_ms_per_item"]
        st.dataframe(pd.DataFrame(rep["test_results"])[cols].round(3), hide_index=True, use_container_width=True)
        img = st.columns(2)
        img[0].image(str(config.FIGURES_DIR / f"{task}_roc.png"))
        img[1].image(str(config.FIGURES_DIR / f"{task}_importances_interpretable.png"))

with tab_monitor:
    logs = monitor.read_log()
    st.write(f"{len(logs)} predictions logged (text stored only as a SHA-256 hash).")
    st.json({t: monitor.drift_report(t, logs) for t in config.TASKS})
