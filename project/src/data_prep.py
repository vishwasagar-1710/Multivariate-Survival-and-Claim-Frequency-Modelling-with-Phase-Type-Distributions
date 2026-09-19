"""
Data preparation & feature engineering for the freMTPL2 bivariate
frequency-severity risk scoring project.

Source data: French Motor Third-Party Liability policies (freMTPL2freq),
678,007 policies, each with:
    ClaimNb      - number of claims during the observed exposure period
    ClaimTotal   - total claim amount (EUR) over the exposure period
    Exposure     - fraction of year the policy was observed (0, 1]
    + 8 rating factors (Area, VehPower, VehAge, DrivAge, BonusMalus,
      VehBrand, VehGas, Density, Region)

Two "views" of the data are produced:
  1. portfolio_df  - every policy, used to fit the risk classifier
                      (features available at renewal, no claim info).
  2. claimant_df   - policies with ClaimNb > 0, used for the bivariate
                      (frequency, severity) distributional analysis
                      (MVN / MANOVA / Mahalanobis), since severity is
                      only observed conditional on at least one claim.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

RAW_PATH = "data/freMTPL2freq.csv"

NUMERIC_FEATURES = ["VehPower", "VehAge", "DrivAge", "BonusMalus",
                     "Density", "Exposure"]
CATEGORICAL_FEATURES = ["Area", "VehBrand", "VehGas", "Region"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Sanity caps consistent with published freMTPL2 pre-processing
    # (Charpentier / CASdatasets guidance: cap gross outlier values).
    df["Exposure"] = df["Exposure"].clip(upper=1.0)
    df["ClaimNb"] = df["ClaimNb"].clip(upper=4)          # cap rare high counts
    df["VehPower"] = df["VehPower"].clip(upper=12)
    df["VehAge"] = df["VehAge"].clip(upper=40)
    df["DrivAge"] = df["DrivAge"].clip(upper=90)
    df["Density"] = df["Density"].clip(upper=df["Density"].quantile(0.999))
    df = df[df["Exposure"] > 0].reset_index(drop=True)
    return df


def engineer_portfolio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature set usable at renewal time (no realized-claim leakage)."""
    df = df.copy()
    df["AnnualFrequency"] = df["ClaimNb"] / df["Exposure"]
    df["PurePremium"] = df["ClaimTotal"] / df["Exposure"]
    df["LogDensity"] = np.log1p(df["Density"])
    # MAP/PH-inspired engineered covariates: proxies for the *latent
    # claim-generating state* of a policy, echoing how a Markovian
    # Arrival Process / phase-type model would summarize a risk's
    # transition intensity out of a "healthy" (no-claim) phase.
    df["RiskLoadIndex"] = (df["BonusMalus"] - 50) / 300.0          # in [0,1]
    df["VehicleRiskIndex"] = (df["VehPower"] / 12.0) * (1 / (1 + df["VehAge"]))
    df["DriverExperienceProxy"] = np.clip((df["DrivAge"] - 18) / 72.0, 0, 1)
    df["UrbanicityIndex"] = df["LogDensity"] / df["LogDensity"].max()
    return df


def build_risk_labels(df: pd.DataFrame, split_q: float = 0.60) -> pd.DataFrame:
    """
    Label every policy Low / Medium / High risk from its realized pure
    premium (ClaimTotal / Exposure). This is the historical-experience
    target the QDA/ML classifier learns to predict *from rating factors
    alone*, so it can flag risk before the next policy period even
    starts (i.e. before any claims of the new term are observed).

    ~96% of policies in freMTPL2 have zero claims, so a naive global
    quantile split collapses onto zero. Instead:
      - Low    : PurePremium == 0 (no realized loss)
      - Medium : PurePremium > 0, below the median of the *positive*
                 pure-premium distribution
      - High   : PurePremium > 0, at/above that median
    This keeps the tiers actuarially meaningful (a policy with any
    realized loss is never "Low") while giving both minority classes
    workable sample sizes for classification.
    """
    df = df.copy()
    positive = df.loc[df["PurePremium"] > 0, "PurePremium"]
    cut = positive.quantile(split_q)
    conditions = [df["PurePremium"] <= 0,
                  (df["PurePremium"] > 0) & (df["PurePremium"] <= cut),
                  df["PurePremium"] > cut]
    df["RiskTier"] = np.select(conditions, ["Low", "Medium", "High"], default="Low")
    return df


def build_claimant_bivariate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Subset to policies with at least one claim and construct the
    (log-frequency, log-severity) bivariate vector used for MVN /
    MANOVA / Mahalanobis analysis.
    """
    claimants = df[df["ClaimNb"] > 0].copy()
    claimants["Frequency"] = claimants["ClaimNb"] / claimants["Exposure"]
    claimants["Severity"] = claimants["ClaimTotal"] / claimants["ClaimNb"]
    claimants = claimants[claimants["Severity"] > 0]
    claimants["LogFrequency"] = np.log(claimants["Frequency"])
    claimants["LogSeverity"] = np.log(claimants["Severity"])
    return claimants


def get_datasets(path: str = RAW_PATH):
    raw = load_raw(path)
    df = clean(raw)
    df = engineer_portfolio_features(df)
    portfolio_df = build_risk_labels(df)
    claimant_df = build_claimant_bivariate(df)
    return portfolio_df, claimant_df


if __name__ == "__main__":
    portfolio_df, claimant_df = get_datasets()
    print("Portfolio:", portfolio_df.shape)
    print(portfolio_df["RiskTier"].value_counts())
    print("Claimants:", claimant_df.shape)
    print(claimant_df[["LogFrequency", "LogSeverity"]].describe())
