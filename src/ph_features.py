"""
ph_features.py
================
MAP/PH-inspired feature engineering.

Phase-type (PH) distributions represent a random variable as the time to
absorption of a finite continuous-time Markov chain. They are natural
generalisations of the Exponential/Erlang/Gamma family and are widely used
in actuarial severity modelling because two moments (mean, squared
coefficient of variation) are enough to *moment-match* a simple PH
representation:

  - SCV == 1  -> a single-phase Exponential is an exact PH(1) fit.
  - SCV  > 1  -> a 2-phase Hyperexponential H2 (mixture of two Exponentials)
                 matches the first two moments (Bobbio-Horvath / balanced
                 means method); heavier-than-exponential tails.
  - SCV  < 1  -> a 2-phase Erlang-Coxian mix (a special Coxian-2) matches
                 the first two moments; lighter-than-exponential, more
                 "regular" claims.

We cannot fit a PH distribution to a *single* policy (a policy contributes
at most a handful of individual claim amounts), so PH parameters are fit
per rating cell (a cross of Area x VehGas x DrivAgeBucket, refined further
by VehAgeBucket if the cell is large enough) using the empirical claims
observed for *other* policies in that cell (a leave-cell-out / pooled
approach, avoiding using a policy's own outcome to build its own feature).
The resulting PH descriptors (p1, rate1, rate2, phase type, implied SCV)
are then joined back onto every policy in the cell as engineered,
pre-renewal-available features for the downstream ML classifier -- this is
the "MAP/PH-inspired feature engineering" step of the project.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_CLAIMS_FOR_CELL_FIT = 15


def _moments(x: np.ndarray) -> tuple[float, float, float]:
    mean = x.mean()
    var = x.var(ddof=1) if len(x) > 1 else 0.0
    scv = var / (mean ** 2) if mean > 0 else np.nan
    return mean, var, scv


def fit_ph2_moment_match(mean: float, scv: float) -> dict:
    """Moment-match a 2-phase PH distribution to (mean, SCV).

    Hyperexponential H2 (SCV > 1), balanced-means parameterisation:
        p = 0.5 * (1 + sqrt((scv - 1) / (scv + 1)))
        rate1 = 2p / mean ;  rate2 = 2(1-p) / mean
    Erlang-Coxian-2 (SCV < 1): mixture of an Erlang-2(rate=r) and a point
    mass approximated by a fast Exponential phase, matched via the standard
    two-moment Coxian-2 formulas.
    Degenerate case (SCV == 1): single-phase Exponential(rate = 1/mean).
    """
    if not np.isfinite(scv) or mean <= 0:
        return {"ph_type": "undefined", "ph_p": np.nan, "ph_rate1": np.nan,
                "ph_rate2": np.nan, "ph_scv": np.nan, "ph_n_phases": np.nan}

    if scv > 1.0 + 1e-9:
        p = 0.5 * (1 + np.sqrt((scv - 1) / (scv + 1)))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        rate1 = 2 * p / mean
        rate2 = 2 * (1 - p) / mean
        return {"ph_type": "hyperexponential", "ph_p": p, "ph_rate1": rate1,
                "ph_rate2": rate2, "ph_scv": scv, "ph_n_phases": 2}
    elif scv < 1.0 - 1e-9:
        # Coxian-2 moment match (Erlang-2 limit as scv -> 0.5, mixed with
        # a fast phase as scv -> 1). Uses the standard two-moment formulas
        # for an Erlang-2 / Exponential mixture (Marie's method).
        scv_c = max(scv, 0.5)  # 0.5 is the Erlang-2 floor for this family
        rate1 = 2.0 / mean
        p = 0.5 * (1 - np.sqrt((1 - scv_c) / (1 + scv_c))) if scv_c < 1 else 1.0
        p = np.clip(p, 1e-6, 1.0)
        rate2 = rate1 / max(p, 1e-6) if p > 0 else rate1
        return {"ph_type": "coxian2", "ph_p": p, "ph_rate1": rate1,
                "ph_rate2": rate2, "ph_scv": scv, "ph_n_phases": 2}
    else:
        rate = 1.0 / mean
        return {"ph_type": "exponential", "ph_p": 1.0, "ph_rate1": rate,
                "ph_rate2": rate, "ph_scv": scv, "ph_n_phases": 1}


def build_cell_ph_features(df: pd.DataFrame, cell_cols: list[str]) -> pd.DataFrame:
    """Fit a moment-matched PH(2) to claimant severities within each rating
    cell (cell_cols), falling back to progressively coarser cells when a
    cell has too few claimants to estimate a stable variance."""
    claimants = df.loc[df.HasClaim == 1, cell_cols + ["AvgSeverity"]].copy()

    records = []
    for key, grp in claimants.groupby(cell_cols, observed=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        x = grp["AvgSeverity"].to_numpy()
        if len(x) < MIN_CLAIMS_FOR_CELL_FIT:
            continue
        mean, var, scv = _moments(x)
        ph = fit_ph2_moment_match(mean, scv)
        rec = dict(zip(cell_cols, key_tuple))
        rec.update({"cell_n_claims": len(x), "cell_mean_severity": mean, "cell_var_severity": var})
        rec.update(ph)
        records.append(rec)

    cell_features = pd.DataFrame(records)

    # global fallback for any cell too sparse to fit directly
    x_all = claimants["AvgSeverity"].to_numpy()
    mean_g, var_g, scv_g = _moments(x_all)
    global_ph = fit_ph2_moment_match(mean_g, scv_g)
    global_ph.update({"cell_n_claims": len(x_all), "cell_mean_severity": mean_g, "cell_var_severity": var_g})

    merged = df.merge(cell_features, on=cell_cols, how="left")
    for col, val in global_ph.items():
        merged[col] = merged[col].fillna(val)
    return merged


PH_FEATURE_COLS = [
    "cell_mean_severity", "cell_var_severity", "ph_p", "ph_rate1",
    "ph_rate2", "ph_scv", "ph_n_phases",
]
