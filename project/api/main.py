"""
Risk Scoring API
=================
Serves the trained bivariate-informed risk classifier as a REST API.

Endpoints
---------
GET  /health
GET  /schema                - expected input fields & categorical levels
POST /score                 - flag a policy Low/Medium/High risk before
                               renewal, using rating factors only
                               (QDA + Gradient Boosting ensemble)
POST /claim-anomaly         - given an observed (frequency, severity)
                               pair for a policy mid-term, return its
                               Mahalanobis distance from the fitted
                               bivariate normal and flag it as a
                               statistical outlier (chi2, df=2)
POST /explain                - natural-language explanation + suggested
                               underwriting actions for a scored policy,
                               grounded in real feature importances/
                               percentiles/historical segment stats and
                               narrated by a local Ollama model
GET  /ollama/models          - list locally available Ollama models

Run locally:
    uvicorn api.main:app --reload --port 8000
"""
from __future__ import annotations
import json
import os
import urllib.request
import urllib.error
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from scipy import stats

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")

app = FastAPI(
    title="Multivariate Claim Risk Scoring API",
    description=(
        "Bivariate survival/claim-frequency-severity risk scoring for "
        "motor insurance policies, backed by MVN theory (Mahalanobis "
        "outlier detection), MANOVA-validated segmentation, and a "
        "QDA + Gradient Boosting classifier ensemble."
    ),
    version="1.0.0",
)

# Allow the companion dashboard (ui/index.html, opened as a local file
# or served from any origin) to call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------- load once ---
_preprocessor = joblib.load(os.path.join(MODEL_DIR, "preprocessor.joblib"))
_qda = joblib.load(os.path.join(MODEL_DIR, "qda_model.joblib"))
_gbm = joblib.load(os.path.join(MODEL_DIR, "gbm_model.joblib"))
_mvn = joblib.load(os.path.join(MODEL_DIR, "mvn_params.joblib"))
with open(os.path.join(MODEL_DIR, "feature_schema.json")) as f:
    _schema = json.load(f)

# Explainability artifacts (optional - only present if train_model.py
# was run after the explainability feature was added; guard so /score
# and /claim-anomaly still work fine without them).
def _load_json_optional(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None

_feature_importance = _load_json_optional(os.path.join(MODEL_DIR, "feature_importance.json"))
_feature_distributions = _load_json_optional(os.path.join(MODEL_DIR, "feature_distributions.json"))
_categorical_risk_stats = _load_json_optional(os.path.join(MODEL_DIR, "categorical_risk_stats.json"))

_MU = np.array(_mvn["mu"])
_SIGMA = np.array(_mvn["sigma"])
_SIGMA_INV = np.linalg.inv(_SIGMA)
_CHI2_THRESHOLD_99 = stats.chi2.ppf(0.99, df=2)


# ------------------------------------------------------------- schemas ----
class PolicyInput(BaseModel):
    VehPower: int = Field(..., ge=1, le=15, description="Vehicle power, ordinal scale")
    VehAge: int = Field(..., ge=0, le=50, description="Vehicle age in years")
    DrivAge: int = Field(..., ge=18, le=100, description="Driver age in years")
    BonusMalus: int = Field(..., ge=50, le=350, description="French bonus-malus factor (100 = neutral)")
    Density: int = Field(..., ge=1, description="Inhabitants per km2 in driver's city")
    Exposure: float = Field(1.0, gt=0, le=1.0, description="Policy exposure fraction of year")
    Area: str = Field(..., description="Area code, e.g. 'A'..'F'")
    VehBrand: str = Field(..., description="Vehicle brand code, e.g. 'B12'")
    VehGas: str = Field(..., description="'Diesel' or 'Regular'")
    Region: str = Field(..., description="French region code, e.g. 'R82'")

    class Config:
        json_schema_extra = {
            "example": {
                "VehPower": 7, "VehAge": 3, "DrivAge": 34, "BonusMalus": 90,
                "Density": 4500, "Exposure": 1.0, "Area": "D",
                "VehBrand": "B12", "VehGas": "Regular", "Region": "R82",
            }
        }


class RiskScoreResponse(BaseModel):
    risk_tier_qda: str
    risk_tier_gbm: str
    probabilities_qda: dict
    probabilities_gbm: dict
    combined_high_risk_score: float
    flag_for_review: bool
    notes: str


class ClaimObservation(BaseModel):
    annual_frequency: float = Field(..., gt=0, description="Observed claim count / exposure so far")
    severity: float = Field(..., gt=0, description="Observed average claim severity (EUR)")


class ClaimAnomalyResponse(BaseModel):
    mahalanobis_distance_sq: float
    chi2_threshold_99pct: float
    is_statistical_outlier: bool
    log_frequency: float
    log_severity: float
    interpretation: str


class KeyFactor(BaseModel):
    feature: str
    policy_value: str
    direction: str          # "increases_risk" | "decreases_risk" | "neutral"
    explanation: str


class ExplainResponse(BaseModel):
    risk_tier_qda: str
    risk_tier_gbm: str
    combined_high_risk_score: float
    summary: str
    key_factors: list[KeyFactor]
    recommended_actions: list[str]
    confidence_note: str
    model_used: str


# --------------------------------------------------------------- helpers ---
def _engineer(policy: PolicyInput) -> pd.DataFrame:
    row = policy.model_dump()
    df = pd.DataFrame([row])
    df["LogDensity"] = np.log1p(df["Density"])
    df["RiskLoadIndex"] = (df["BonusMalus"] - 50) / 300.0
    df["VehicleRiskIndex"] = (df["VehPower"] / 12.0) * (1 / (1 + df["VehAge"]))
    df["DriverExperienceProxy"] = np.clip((df["DrivAge"] - 18) / 72.0, 0, 1)
    df["UrbanicityIndex"] = df["LogDensity"] / np.log1p(30000)  # ~ portfolio max density
    return df


# ------------------------------------------------------------- endpoints --
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/schema")
def schema():
    return _schema


def _score_policy(policy: PolicyInput):
    """Shared scoring logic used by both /score and /explain."""
    feats = _engineer(policy)
    cols = _schema["numeric_features"] + _schema["categorical_features"]
    X = feats[cols]
    Xt = _preprocessor.transform(X)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()

    qda_proba = _qda.predict_proba(Xt)[0]
    gbm_proba = _gbm.predict_proba(Xt)[0]
    qda_classes = list(_qda.classes_)
    gbm_classes = list(_gbm.classes_)

    qda_pred = qda_classes[int(np.argmax(qda_proba))]
    gbm_pred = gbm_classes[int(np.argmax(gbm_proba))]

    qda_dict = {c: float(p) for c, p in zip(qda_classes, qda_proba)}
    gbm_dict = {c: float(p) for c, p in zip(gbm_classes, gbm_proba)}

    combined = float(np.mean([
        qda_dict.get("Medium", 0) + qda_dict.get("High", 0),
        gbm_dict.get("Medium", 0) + gbm_dict.get("High", 0),
    ]))

    return {
        "feats": feats,
        "risk_tier_qda": qda_pred,
        "risk_tier_gbm": gbm_pred,
        "probabilities_qda": qda_dict,
        "probabilities_gbm": gbm_dict,
        "combined_high_risk_score": combined,
        "flag_for_review": combined > 0.5,
    }


@app.post("/score", response_model=RiskScoreResponse)
def score(policy: PolicyInput):
    try:
        result = _score_policy(policy)
        return RiskScoreResponse(
            risk_tier_qda=result["risk_tier_qda"],
            risk_tier_gbm=result["risk_tier_gbm"],
            probabilities_qda=result["probabilities_qda"],
            probabilities_gbm=result["probabilities_gbm"],
            combined_high_risk_score=result["combined_high_risk_score"],
            flag_for_review=result["flag_for_review"],
            notes=(
                "QDA uses balanced class priors (sensitive, higher false-"
                "positive rate) per the course MVN classification theory; "
                "Gradient Boosting is calibrated to the true ~96% no-claim "
                "base rate (higher precision, lower recall). Use "
                "combined_high_risk_score to rank policies for underwriting "
                "review rather than relying on either hard label alone."
            ),
        )
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Unknown/invalid field: {e}")


@app.post("/claim-anomaly", response_model=ClaimAnomalyResponse)
def claim_anomaly(obs: ClaimObservation):
    """
    Flags whether an observed mid-term (frequency, severity) pair for a
    policy is a multivariate statistical outlier relative to the fitted
    bivariate normal on the claimant population (Mahalanobis distance /
    chi-square test, df=2, alpha=0.01). Useful for triaging claims for
    fraud review or reserve adequacy checks.
    """
    log_f = float(np.log(obs.annual_frequency))
    log_s = float(np.log(obs.severity))
    x = np.array([log_f, log_s])
    diff = x - _MU
    d2 = float(diff @ _SIGMA_INV @ diff)
    is_outlier = d2 > _CHI2_THRESHOLD_99

    if is_outlier:
        interp = ("This claim profile falls outside the 99% confidence "
                   "region of typical claimant behaviour - recommend "
                   "manual review.")
    else:
        interp = "Claim profile is consistent with typical claimant behaviour."

    return ClaimAnomalyResponse(
        mahalanobis_distance_sq=d2,
        chi2_threshold_99pct=float(_CHI2_THRESHOLD_99),
        is_statistical_outlier=bool(is_outlier),
        log_frequency=log_f,
        log_severity=log_s,
        interpretation=interp,
    )


# ---------------------------------------------------- explainability -----
def _percentile_label(value: float, dist: dict) -> str:
    """Cheap percentile estimate from stored quantile checkpoints."""
    checkpoints = [("p5", 5), ("p25", 25), ("p50", 50), ("p75", 75), ("p95", 95)]
    if value <= dist["p5"]:
        return "bottom 5%"
    if value >= dist["p95"]:
        return "top 5%"
    for (key, pct), (next_key, next_pct) in zip(checkpoints, checkpoints[1:]):
        if dist[key] <= value <= dist[next_key]:
            # linear-interpolate within this bracket for a friendlier number
            lo, hi = dist[key], dist[next_key]
            frac = 0 if hi == lo else (value - lo) / (hi - lo)
            approx_pct = pct + frac * (next_pct - pct)
            return f"~{approx_pct:.0f}th percentile"
    return "near the median"


def _build_explain_context(policy: PolicyInput, feats: pd.DataFrame, score_result: dict) -> dict:
    """
    Assemble the grounded, numeric facts an LLM should reason over -
    top globally-important features (from permutation importance),
    this policy's own values and percentile standing on them, and
    historical risk stats for its specific categorical levels. This
    is what keeps the LLM's narrative tied to the actual model rather
    than inventing plausible-sounding but unfounded reasons.
    """
    context = {
        "risk_tier_qda": score_result["risk_tier_qda"],
        "risk_tier_gbm": score_result["risk_tier_gbm"],
        "combined_high_risk_score": round(score_result["combined_high_risk_score"], 4),
        "probabilities_qda": score_result["probabilities_qda"],
        "probabilities_gbm": score_result["probabilities_gbm"],
        "policy_input": policy.model_dump(),
    }

    if _feature_importance:
        # Collapse one-hot categorical importances back to their parent
        # feature for a cleaner top-N list, keeping the strongest level.
        collapsed = {}
        for row in _feature_importance:
            name = row["feature"]
            if name.startswith("num__"):
                key = name.replace("num__", "")
            elif name.startswith("cat__"):
                key = name.replace("cat__", "").rsplit("_", 1)[0]
            else:
                key = name
            if key not in collapsed or row["importance"] > collapsed[key]:
                collapsed[key] = row["importance"]
        top_features = sorted(collapsed.items(), key=lambda kv: -kv[1])[:6]
        context["top_important_features"] = [{"feature": k, "importance": round(v, 5)}
                                              for k, v in top_features]

        factor_details = []
        for feat_name, _ in top_features:
            if _feature_distributions and feat_name in _feature_distributions:
                value = float(feats[feat_name].iloc[0])
                dist = _feature_distributions[feat_name]
                factor_details.append({
                    "feature": feat_name,
                    "policy_value": value,
                    "portfolio_median": dist["p50"],
                    "percentile": _percentile_label(value, dist),
                })
            elif _categorical_risk_stats and feat_name in _categorical_risk_stats:
                level = str(policy.model_dump().get(feat_name, ""))
                stats_for_level = _categorical_risk_stats[feat_name].get(level)
                if stats_for_level:
                    factor_details.append({
                        "feature": feat_name,
                        "policy_value": level,
                        "level_n_policies": stats_for_level["n_policies"],
                        "level_mean_pure_premium_eur": round(stats_for_level["mean_pure_premium"], 2),
                        "level_pct_medium_or_high": round(stats_for_level["pct_medium_or_high"], 2),
                    })
        context["factor_details"] = factor_details

    return context


def _call_ollama(prompt: str, model: str) -> dict:
    """POST to a local Ollama server's /api/generate, forcing JSON output."""
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.2},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Could not reach Ollama at {OLLAMA_URL} ({e}). Make sure "
                f"Ollama is running (`ollama serve`) and the model is "
                f"pulled (`ollama pull {model}`)."
            ),
        )
    raw = payload.get("response", "")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama returned non-JSON output that couldn't be parsed: {raw[:500]}",
        )


EXPLAIN_PROMPT_TEMPLATE = """You are an underwriting assistant explaining a motor insurance risk score to a human underwriter. You are given the model's actual output and grounded statistics - only reason from these facts, do not invent numbers.

RISK ASSESSMENT DATA (JSON):
{context_json}

Notes on the fields: "top_important_features" ranks features by global permutation importance from the deployed model (how much shuffling that feature hurts prediction quality across all policies) - it is NOT specific to this one policy. "factor_details" gives this specific policy's own value on each of those features, its percentile standing in the portfolio (for numeric features) or its historical claim stats (for categorical features like Area/Region/VehBrand/VehGas).

Write a response as JSON with exactly these keys:
- "summary": 2-3 sentences, plain language, explaining why this policy landed in its risk tier, referencing the combined_high_risk_score and the risk tier.
- "key_factors": a list of 3-5 objects, each with "feature", "policy_value" (as a short string), "direction" (one of "increases_risk", "decreases_risk", "neutral"), and "explanation" (one sentence grounded in the factor_details numbers given - reference the percentile or historical stat explicitly).
- "recommended_actions": a list of 2-4 short, concrete underwriting actions (e.g. request additional documentation, apply a rating surcharge, refer for manual review, approve as standard risk). Actions should match the risk tier - do not recommend aggressive action for a Low risk tier.
- "confidence_note": one sentence noting that this is a statistical/ML estimate, not a certainty, and that QDA and Gradient Boosting may disagree (mention if they do here).

Respond with ONLY the JSON object, no markdown fences, no extra commentary."""


@app.post("/explain", response_model=ExplainResponse)
def explain(policy: PolicyInput, model: str = DEFAULT_OLLAMA_MODEL):
    """
    Score the policy, then ask a local Ollama model to turn the
    grounded statistical context into a plain-language explanation and
    underwriting suggestions. Requires Ollama running locally
    (`ollama serve`) with `model` already pulled.
    """
    if _feature_importance is None:
        raise HTTPException(
            status_code=409,
            detail=(
                "Explainability artifacts not found (models/feature_importance.json "
                "etc). Re-run `python src/train_model.py` to generate them."
            ),
        )
    score_result = _score_policy(policy)
    context = _build_explain_context(policy, score_result["feats"], score_result)
    prompt = EXPLAIN_PROMPT_TEMPLATE.format(context_json=json.dumps(context, indent=2))

    llm_json = _call_ollama(prompt, model)

    try:
        key_factors = [KeyFactor(**kf) for kf in llm_json.get("key_factors", [])]
    except Exception:
        key_factors = []

    return ExplainResponse(
        risk_tier_qda=score_result["risk_tier_qda"],
        risk_tier_gbm=score_result["risk_tier_gbm"],
        combined_high_risk_score=score_result["combined_high_risk_score"],
        summary=llm_json.get("summary", ""),
        key_factors=key_factors,
        recommended_actions=llm_json.get("recommended_actions", []),
        confidence_note=llm_json.get("confidence_note", ""),
        model_used=model,
    )


@app.get("/ollama/models")
def ollama_models():
    """Proxy to Ollama's /api/tags, so the UI can populate a model picker."""
    req = urllib.request.Request(f"{OLLAMA_URL}/api/tags", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Could not reach Ollama at {OLLAMA_URL} ({e}). Is `ollama serve` running?",
        )
    names = [m["name"] for m in payload.get("models", [])]
    return {"models": names, "default": DEFAULT_OLLAMA_MODEL}
