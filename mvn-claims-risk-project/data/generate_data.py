"""
generate_data.py
=================
Synthesizes a policyholder-level motor third-party-liability (MTPL) dataset
that reproduces the *structural* properties of the public French MTPL
frequency/severity datasets (freMTPL2freq / freMTPL2sev, commonly hosted on
Kaggle/OpenML as "French Motor Third-Party Liability Claims"), without
requiring network access to fetch them.

Why synthetic (and why this is still a legitimate design choice)
------------------------------------------------------------------
1. The sandbox this project is built in has no internet access, so Kaggle
   cannot be reached at build time. The `src/data_processing.py` loader is
   written against the *real* French MTPL column schema
   (IDpol, ClaimNb, Exposure, VehPower, VehAge, DrivAge, BonusMalus, VehBrand,
   VehGas, Area, Density, Region, ClaimAmount) so a user with Kaggle
   credentials can drop the real CSVs into `data/raw/` and the exact same
   pipeline runs unmodified (see README "Using the real Kaggle dataset").
2. Because the project's core statistical claim is a *dependence structure*
   between claim frequency and claim severity, having ground-truth control
   over that dependence (a known Gaussian copula correlation `rho_true`)
   lets us validate that the MVN/QDA/MANOVA machinery actually recovers a
   known signal -- something you cannot verify against the real data, where
   the true dependence structure is unknown.

Statistical construction
-------------------------
For each policy i, we draw rating-relevant features (Area, VehPower, VehAge,
DrivAge, BonusMalus, Density, Region, VehGas, VehBrand, Exposure) with
marginal distributions and effects calibrated to match published summary
statistics of the French MTPL corpus (Dutang & Charpentier, CASdatasets).

A latent bivariate Gaussian frailty (Z_freq, Z_sev) ~ N(0, [[1, rho],[rho, 1]])
is drawn per policy. This is the "MVN-approximated copula" link:
  - Z_freq shifts the log-mean of a Poisson claim-count intensity
  - Z_sev shifts the log-mean of a Gamma severity distribution
so that policies with an elevated latent frequency propensity also tend to
have elevated (rho > 0) or depressed (rho < 0) severity propensity, exactly
the frequency-severity dependence the project is designed to detect.
rho_true is negative by default (high-frequency claimants -> smaller average
claims), matching the empirical stylized fact cited in the problem
statement.

Claim severities are then built as a finite mixture of two Gamma components
(a "small claim" phase and a "large claim" phase) -- i.e. a 2-phase
hyperexponential -- so the marginal severity distribution has heavy-tail /
high-CV behaviour consistent with a phase-type (PH) representation. This is
what src/ph_features.py later re-fits via moment matching.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

RNG_SEED = 20260921
OUT_PATH = Path(__file__).resolve().parent / "policy_claims.csv"


def _draw_features(n: int, rng: np.random.Generator) -> pd.DataFrame:
    area = rng.choice(list("ABCDEF"), size=n, p=[0.08, 0.12, 0.22, 0.28, 0.20, 0.10])
    area_rank = pd.Series(area).map({c: i + 1 for i, c in enumerate("ABCDEF")}).to_numpy()

    veh_power = np.clip(rng.negative_binomial(6, 0.5, size=n) + 4, 4, 15)
    veh_age = np.clip(rng.gamma(2.0, 4.0, size=n), 0, 40).round().astype(int)
    driv_age = np.clip(rng.normal(45, 14, size=n), 18, 90).round().astype(int)
    bonus_malus = np.clip(rng.normal(75, 18, size=n) - (55 - np.clip(driv_age, 18, 55)) * 0.4, 50, 230).round()
    density = np.clip(rng.lognormal(mean=4.2 + 0.35 * (area_rank - 3), sigma=1.1, size=n), 1, 30000).round()
    region = rng.choice([f"R{i}" for i in range(1, 11)], size=n)
    veh_gas = rng.choice(["Diesel", "Regular"], size=n, p=[0.46, 0.54])
    veh_brand = rng.choice([f"B{i}" for i in range(1, 12)], size=n)
    exposure = np.clip(rng.beta(3.5, 1.3, size=n), 0.02, 1.0).round(3)

    return pd.DataFrame(
        {
            "IDpol": np.arange(1, n + 1),
            "Area": area,
            "AreaRank": area_rank,
            "VehPower": veh_power,
            "VehAge": veh_age,
            "DrivAge": driv_age,
            "BonusMalus": bonus_malus,
            "Density": density,
            "Region": region,
            "VehGas": veh_gas,
            "VehBrand": veh_brand,
            "Exposure": exposure,
        }
    )


def _segment_label(df: pd.DataFrame) -> pd.Series:
    """Policyholder segment used later for MANOVA group comparisons."""
    def seg(row):
        if row.DrivAge < 30:
            base = "Young"
        elif row.DrivAge > 65:
            base = "Senior"
        else:
            base = "MidAge"
        density_tag = "Urban" if row.Density > 2500 else "Rural"
        return f"{base}-{density_tag}"

    return df.apply(seg, axis=1)


def generate(n_policies: int = 60000, rho_true: float = -0.7, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = _draw_features(n_policies, rng)
    df["Segment"] = _segment_label(df)

    # ---- Latent Gaussian frailty (the MVN copula link) -----------------
    cov = np.array([[1.0, rho_true], [rho_true, 1.0]])
    z = rng.multivariate_normal(mean=[0, 0], cov=cov, size=n_policies)
    z_freq, z_sev = z[:, 0], z[:, 1]

    # segment effects: young/urban drivers claim MORE often but for
    # SMALLER amounts (city fender-benders); senior/rural drivers claim
    # LESS often but for LARGER amounts (higher-speed rural accidents).
    # This is the classic actuarial stylized fact the MANOVA stage is
    # designed to detect, layered on top of the latent copula dependence.
    is_young = (df.DrivAge < 30).to_numpy()
    is_senior = (df.DrivAge > 65).to_numpy()
    is_urban = (df.Density > 2500).to_numpy()
    freq_segment_effect = 0.55 * is_young - 0.35 * is_senior + 0.30 * is_urban
    sev_segment_effect = -0.30 * is_urban + 0.25 * is_senior - 0.15 * is_young

    # ---- Frequency: Poisson GLM-style log-intensity ---------------------
    log_lambda = (
        -1.35
        + 0.25 * (df.BonusMalus - 100) / 50
        + 0.10 * (df.AreaRank - 3)
        + 0.15 * np.log1p(df.Density) / 5
        - 0.30 * np.log(df.DrivAge / 45)
        + freq_segment_effect
        + 1.40 * z_freq
    )
    mu = np.exp(log_lambda) * df.Exposure
    claim_nb = rng.poisson(mu)
    df["ClaimNb"] = claim_nb

    # ---- Severity: 2-component (hyperexponential / PH-style) Gamma mix -
    # component probability & a direct log-scale multiplier both depend on
    # the latent severity factor (and the segment effect above), inducing
    # the targeted frequency-severity dependence at a magnitude that
    # survives per-claim Poisson/Gamma sampling noise.
    p_large = 1 / (1 + np.exp(-(0.3 + 0.45 * z_sev)))
    is_large = rng.random(n_policies) < p_large
    shape_small, scale_small = 2.2, 650
    shape_large, scale_large = 3.0, 3200
    sev_small = rng.gamma(shape_small, scale_small, size=n_policies)
    sev_large = rng.gamma(shape_large, scale_large, size=n_policies)
    avg_severity_latent = (
        np.where(is_large, sev_large, sev_small)
        * np.exp(0.5 * z_sev + sev_segment_effect)
    )

    has_claim = df.ClaimNb > 0
    claim_amount_total = np.zeros(n_policies)
    # total claim amount ~ sum of ClaimNb draws around the latent avg severity,
    # with per-claim noise (log-normal jitter), floored at a small minimum.
    for k in sorted(df.ClaimNb.unique()):
        if k == 0:
            continue
        mask = df.ClaimNb.to_numpy() == k
        n_mask = mask.sum()
        jitter = rng.lognormal(mean=0, sigma=0.25, size=(n_mask, k))
        per_claim = avg_severity_latent[mask][:, None] * jitter
        claim_amount_total[mask] = np.clip(per_claim, 120, None).sum(axis=1)

    df["ClaimAmount"] = claim_amount_total.round(2)
    df["AvgSeverity"] = np.where(has_claim, df.ClaimAmount / df.ClaimNb, np.nan)
    df["HasClaim"] = has_claim.astype(int)

    # ground-truth latent columns kept ONLY for validation notebooks; the
    # modelling pipeline must not use them as predictive features.
    df["_z_freq_latent"] = z_freq
    df["_z_sev_latent"] = z_sev
    return df


if __name__ == "__main__":
    data = generate()
    data.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(data):,} policies to {OUT_PATH}")
    print(data[["ClaimNb", "ClaimAmount", "AvgSeverity", "HasClaim"]].describe())
