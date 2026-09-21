"""
scoring.py
===========
Loads the persisted pipeline artifacts (ML model, PH cell lookup, MVN
params) produced by scripts/run_pipeline.py and exposes a single
`score_policy(...)` function used by BOTH:
  - app/main.py       (production FastAPI service)
  - app/demo_server.py (dependency-free stdlib http.server, used to prove
                         the scoring logic works end-to-end inside this
                         sandbox, where fastapi/uvicorn cannot be installed
                         due to no network access -- see README)

Keeping the scoring logic in one module means the two servers are
guaranteed to behave identically; only the HTTP transport differs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
ART = ROOT / "outputs" / "artifacts"

MODEL_VERSION = "high-risk-gbc-v1"

_model = None
_ph_lookup = None
_global_ph = None


def _load():
    global _model, _ph_lookup, _global_ph
    if _model is None:
        model_path = ART / "ml_high_risk_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                "Model artifact not found. Run `python scripts/run_pipeline.py` first."
            )
        _model = joblib.load(model_path)
        ph_records = json.loads((ART / "ph_cell_lookup.json").read_text())
        _ph_lookup = pd.DataFrame(ph_records)
        # global fallback = the modal / median row (used for unseen cells)
        _global_ph = _ph_lookup.select_dtypes(include=[np.number]).median().to_dict()
    return _model, _ph_lookup, _global_ph


def _drivage_bucket(age: int) -> str:
    bins = [17, 25, 35, 50, 65, 200]
    labels = ["18-25", "26-35", "36-50", "51-65", "65+"]
    for lo, hi, lab in zip(bins[:-1], bins[1:], labels):
        if lo < age <= hi:
            return lab
    return labels[-1]


def _lookup_ph(area: str, veh_gas: str, driv_age_bucket: str, ph_lookup: pd.DataFrame, global_ph: dict) -> dict:
    match = ph_lookup[
        (ph_lookup.Area == area) & (ph_lookup.VehGas == veh_gas) & (ph_lookup.DrivAgeBucket == driv_age_bucket)
    ]
    if len(match) == 0:
        return global_ph
    row = match.iloc[0]
    return {
        "cell_mean_severity": row.cell_mean_severity, "cell_var_severity": row.cell_var_severity,
        "ph_p": row.ph_p, "ph_rate1": row.ph_rate1, "ph_rate2": row.ph_rate2, "ph_scv": row.ph_scv,
    }


def score_policy(policy: dict) -> dict:
    """policy: dict with keys Area, VehPower, VehAge, DrivAge, BonusMalus,
    Density, Region, VehGas, VehBrand, Exposure. Returns a scoring result
    dict matching app/schemas.RiskScoreOutput."""
    model, ph_lookup, global_ph = _load()

    driv_bucket = _drivage_bucket(policy["DrivAge"])
    ph_feats = _lookup_ph(policy["Area"], policy["VehGas"], driv_bucket, ph_lookup, global_ph)

    row = {
        "VehPower": policy["VehPower"], "VehAge": policy["VehAge"], "DrivAge": policy["DrivAge"],
        "BonusMalus": policy["BonusMalus"], "LogDensity": float(np.log1p(policy["Density"])),
        "Exposure": policy["Exposure"], "Area": policy["Area"], "VehGas": policy["VehGas"],
        "Region": policy["Region"], "VehBrand": policy["VehBrand"],
    }
    row.update(ph_feats)
    X = pd.DataFrame([row])

    from src.ml_classifier import ALL_FEATURES
    proba = float(model.predict_proba(X[ALL_FEATURES])[0, 1])

    results_path = ROOT / "outputs" / "results" / "results.json"
    op_threshold = 0.02
    high_threshold = 0.05
    if results_path.exists():
        res = json.loads(results_path.read_text())
        mlc = res.get("ml_classifier", {})
        op_threshold = mlc.get("operating_threshold_proba", op_threshold)
        high_threshold = mlc.get("high_tier_threshold_proba", high_threshold)

    flag = proba >= op_threshold
    if proba >= high_threshold:
        tier = "High"
    elif proba >= op_threshold:
        tier = "Medium"
    else:
        tier = "Low"

    return {
        "high_risk_probability": round(proba, 6),
        "high_risk_flag": bool(flag),
        "risk_tier": tier,
        "operating_threshold": round(float(op_threshold), 6),
        "model_version": MODEL_VERSION,
    }
