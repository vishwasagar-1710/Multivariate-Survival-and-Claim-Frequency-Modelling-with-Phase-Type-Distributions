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

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import BatchPolicyInput, BatchRiskScoreOutput, PolicyInput, RiskScoreOutput
from app.scoring import score_policy

app = FastAPI(
    title="Bivariate Frequency-Severity Risk Scoring API",
    description=(
        "Flags motor policies as high-risk pre-renewal using a Gradient "
        "Boosting classifier trained on rating factors plus MAP/PH-inspired "
        "phase-type severity features. Backed by a bivariate-MVN / MANOVA / "
        "QDA analysis of the joint (frequency, severity) distribution -- "
        "see /mnt/user-data/outputs REPORT.md for full methodology."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {
        "service": "Bivariate Survival & Claim Frequency Risk Scoring API",
        "endpoints": ["/health", "/score (POST)", "/score/batch (POST)", "/docs"],
    }


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
