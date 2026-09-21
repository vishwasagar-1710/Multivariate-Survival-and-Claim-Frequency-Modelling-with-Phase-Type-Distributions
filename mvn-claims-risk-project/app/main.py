"""
main.py
========
Risk-scoring API. Flags policies as high/medium/low risk pre-renewal using
the trained Gradient Boosting classifier (src/ml_classifier.py) fed by
MAP/PH-inspired rating-cell features (src/ph_features.py).

Run locally:
    pip install -r requirements.txt
    python scripts/run_pipeline.py        # trains & persists model artifacts
    uvicorn app.main:app --reload --port 8000

Then visit http://localhost:8000/docs for interactive Swagger UI, or:
    curl -X POST http://localhost:8000/score -H "Content-Type: application/json" \
         -d '{"Area":"C","VehPower":7,"VehAge":4,"DrivAge":28,"BonusMalus":95,
              "Density":3200,"Region":"R3","VehGas":"Diesel","VehBrand":"B4","Exposure":0.85}'

Deployment: this app is designed to run in a GitHub Codespace (see
README.md "Deployment" section) or any container platform via the included
Dockerfile.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.schemas import BatchPolicyInput, BatchRiskScoreOutput, PolicyInput, RiskScoreOutput
from app.scoring import score_policy

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT / "frontend"
OUTPUTS_DIR = ROOT / "outputs"

app = FastAPI(
    title="Bivariate Frequency-Severity Risk Scoring API",
    description=(
        "Flags motor policies as high-risk pre-renewal using a Gradient "
        "Boosting classifier trained on rating factors plus MAP/PH-inspired "
        "phase-type severity features. Backed by a bivariate-MVN / MANOVA / "
        "QDA analysis of the joint (frequency, severity) distribution -- "
        "see REPORT.md for full methodology."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# Serve generated plots + results.json so the frontend can render live
# figures and findings without duplicating any numbers in JS.
if OUTPUTS_DIR.exists():
    app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api")
def api_info():
    return {
        "service": "Bivariate Survival & Claim Frequency Risk Scoring API",
        "endpoints": ["/health", "/score (POST)", "/score/batch (POST)", "/docs"],
    }


@app.get("/")
def root():
    """Serves the React frontend (frontend/index.html) at the site root.
    Falls back to a plain JSON status if the frontend hasn't been built
    into the image for some reason."""
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"status": "ok", "note": "frontend/index.html not found; see /api and /docs"}


@app.post("/score", response_model=RiskScoreOutput)
def score(policy: PolicyInput):
    try:
        return score_policy(policy.model_dump())
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=400, detail=f"Scoring failed: {e}")


@app.post("/score/batch", response_model=BatchRiskScoreOutput)
def score_batch(batch: BatchPolicyInput):
    try:
        results = [score_policy(p.model_dump()) for p in batch.policies]
        return {"results": results}
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
