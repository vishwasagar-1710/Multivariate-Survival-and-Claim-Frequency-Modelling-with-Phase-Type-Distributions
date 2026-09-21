"""
data_processing.py
===================
Loading, cleaning and feature engineering for the policy-claims dataset.

Also defines `load_real_kaggle_data()` which reads the *actual* French MTPL
CSVs (freMTPL2freq.csv + freMTPL2sev.csv) if a user places them under
data/raw/ -- this lets the whole pipeline run on the real Kaggle dataset
with zero code changes elsewhere.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
SYNTH_PATH = DATA_DIR / "policy_claims.csv"

EPS = 1e-6  # numerical floor before log transforms


def load_real_kaggle_data() -> pd.DataFrame | None:
    """Load the real 'French Motor Third-Party Liability Claims' dataset
    (freMTPL2freq.csv / freMTPL2sev.csv, e.g. from Kaggle user karansarpal's
    freMTPL2-french-motor-tpl-insurance-claims dataset, OpenML ids 41214/
    41215, or the mabilton/fremtpl2 Hugging Face mirror) if present in
    data/raw/. Returns None (triggering the synthetic fallback) otherwise.

    Applies the standard, literature-precedented cleaning steps for this
    dataset (matching scikit-learn's own Tweedie-regression tutorial on the
    same data, and the original CASdatasets documentation):
      - IDpol dtype aligned across the two files before merging
      - severity rows summed per policy (a policy can have >1 claim row)
      - ClaimNb reset to 0 where the matched ClaimAmount is 0 despite
        ClaimNb >= 1 (a known data-entry artifact: ~9.1k such rows)
      - Exposure clipped to at most 1.0 (max is 2.01 in the raw data,
        exceeding a policy-year -- a known data quirk)
      - ClaimNb clipped to at most 4 (a handful of rows report up to 16,
        treated as data errors, consistent with the sklearn tutorial)
      - ClaimAmount clipped to at most 200,000 (avoids a small number of
        multi-million-euro claims dominating the variance of the severity
        distribution)
    """
    freq_path = RAW_DIR / "freMTPL2freq.csv"
    sev_path = RAW_DIR / "freMTPL2sev.csv"
    if not (freq_path.exists() and sev_path.exists()):
        return None

    freq = pd.read_csv(freq_path)
    sev = pd.read_csv(sev_path)
    freq["IDpol"] = freq["IDpol"].astype(np.int64)
    sev["IDpol"] = sev["IDpol"].astype(np.int64)

    sev_agg = sev.groupby("IDpol", as_index=False)["ClaimAmount"].sum()
    df = freq.merge(sev_agg, on="IDpol", how="left")
    df["ClaimAmount"] = df["ClaimAmount"].fillna(0.0)

    # --- standard cleaning (see docstring) -------------------------------
    df.loc[(df["ClaimAmount"] == 0) & (df["ClaimNb"] >= 1), "ClaimNb"] = 0
    df["Exposure"] = df["Exposure"].clip(upper=1.0)
    df["ClaimNb"] = df["ClaimNb"].clip(upper=4)
    df["ClaimAmount"] = df["ClaimAmount"].clip(upper=200_000)

    df["AreaRank"] = df["Area"].map({c: i + 1 for i, c in enumerate("ABCDEF")})
    df["HasClaim"] = (df["ClaimNb"] > 0).astype(int)
    df["AvgSeverity"] = np.where(df.HasClaim == 1, df.ClaimAmount / df.ClaimNb, np.nan)

    def seg_vectorized(d: pd.DataFrame) -> pd.Series:
        age_tag = np.where(d.DrivAge < 30, "Young", np.where(d.DrivAge > 65, "Senior", "MidAge"))
        dens_tag = np.where(d.Density > 2500, "Urban", "Rural")
        return pd.Series([f"{a}-{d_}" for a, d_ in zip(age_tag, dens_tag)], index=d.index)

    df["Segment"] = seg_vectorized(df)
    return df


def load_dataset(prefer_real: bool = True) -> tuple[pd.DataFrame, str]:
    """Returns (dataframe, source_tag) where source_tag is 'kaggle-real' or
    'synthetic'. Falls back to synthetic data automatically."""
    if prefer_real:
        real = load_real_kaggle_data()
        if real is not None:
            return real, "kaggle-real"
    if not SYNTH_PATH.exists():
        raise FileNotFoundError(
            f"No dataset found. Run `python data/generate_data.py` first, "
            f"or place real CSVs in {RAW_DIR}."
        )
    return pd.read_csv(SYNTH_PATH), "synthetic"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Core feature engineering shared by every downstream model:
    exposure-normalised frequency, log transforms for the MVN stage,
    and rating features used by the ML classifier."""
    out = df.copy()
    out["Frequency"] = out["ClaimNb"] / out["Exposure"].clip(lower=EPS)
    out["LogFrequency"] = np.log(out["Frequency"] + EPS)
    out["LogAvgSeverity"] = np.log(out["AvgSeverity"].clip(lower=EPS))

    # For policies with zero claims, AvgSeverity is undefined. For the
    # bivariate (frequency, severity) analysis we only use claimants
    # (HasClaim == 1); the ML classifier below instead predicts, from
    # PRE-RENEWAL rating features alone (no leakage from this year's claim
    # outcome), whether a policy is likely to become a high-cost claimant.
    out["LogBonusMalus"] = np.log(out["BonusMalus"])
    out["LogDensity"] = np.log1p(out["Density"])
    out["VehAgeBucket"] = pd.cut(
        out["VehAge"], bins=[-1, 2, 5, 10, 20, 100], labels=["0-2", "3-5", "6-10", "11-20", "20+"]
    )
    out["DrivAgeBucket"] = pd.cut(
        out["DrivAge"], bins=[17, 25, 35, 50, 65, 100], labels=["18-25", "26-35", "36-50", "51-65", "65+"]
    )
    return out


RATING_FEATURES_NUMERIC = [
    "VehPower", "VehAge", "DrivAge", "BonusMalus", "LogDensity", "Exposure",
]
RATING_FEATURES_CATEGORICAL = ["Area", "VehGas", "Region", "VehBrand"]


def train_test_split_by_policy(df: pd.DataFrame, test_size: float = 0.25, seed: int = 42):
    from sklearn.model_selection import train_test_split

    return train_test_split(df, test_size=test_size, random_state=seed, stratify=df["HasClaim"])
