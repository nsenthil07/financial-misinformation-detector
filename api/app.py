"""Step 9 - Serve the detector as a REST API with FastAPI.

Run:   uvicorn api.app:app --reload
Docs:  http://localhost:8000/docs   (interactive page, generated automatically)

Endpoints
  GET  /health            is the service up, are the models loaded?
  GET  /v1/model-info     which models are live and how well they tested
  POST /v1/analyze        score one text, optionally with an LLM explanation
  GET  /v1/monitoring     drift and operational stats from the prediction log
"""
import sys
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fmd import config, monitor  # noqa: E402
from fmd.detector import Detector  # noqa: E402

app = FastAPI(
    title="Financial Misinformation & Scam Detector",
    version="1.0.0",
    description="Tree-based classifiers (decision tree / random forest / boosting) with an LLM + RAG "
                "fact-check assistant. Research prototype - not for automated decisions about members.",
)

_detector = None


def get_detector():
    """Load the models the first time they are needed, then reuse them."""
    global _detector
    if _detector is None:
        _detector = Detector()
    return _detector


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=5000,
                      examples=["URGENT: your account is locked. Verify now at secure-bank-login.com"])
    task: Literal["auto", "claims", "messages"] = "auto"
    explain: bool = Field(True, description="run the RAG + LLM assistant for flagged/uncertain text")
    llm_provider: Literal["auto", "offline", "ollama", "openai", "anthropic"] | None = None


@app.get("/health")
def health():
    try:
        get_detector()
        return {"status": "ok", "models_loaded": list(config.TASKS)}
    except FileNotFoundError as error:
        return {"status": "degraded", "detail": str(error)}


@app.get("/v1/model-info")
def model_info():
    return get_detector().info


@app.post("/v1/analyze")
def analyze(request: AnalyzeRequest):
    try:
        detector = get_detector()
    except FileNotFoundError as error:
        raise HTTPException(503, str(error))
    return detector.analyze(request.text, request.task, request.explain, request.llm_provider)


@app.get("/v1/monitoring")
def monitoring():
    logs = monitor.read_log()
    return {task: monitor.drift_report(task, logs) for task in config.TASKS}
