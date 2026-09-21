"""
run_pipeline.py
=================
Orchestrates the full analysis:
  1. Load data (real Kaggle CSVs if present, else synthetic fallback)
  2. Feature engineering (log transforms, PH-inspired cell features)
  3. Bivariate MVN fit on claimants + conditional distribution demo
  4. Chi-square / Mahalanobis outlier detection
  5. One-way MANOVA across policyholder segments
  6. QDA risk-tier classification
  7. ML high-risk classifier (Gradient Boosting) + evaluation
  8. Persist all artifacts (model, MVN params, plots, results.json) needed
     by app/main.py (the risk-scoring API)

Run with:  python scripts/run_pipeline.py
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

from src import data_processing as dp
from src import mvn_analysis as mvn
from src import manova_analysis as manova
from src import ph_features as ph
from src import qda_classifier as qda
from src import ml_classifier as mlc
from src import plotting as viz

OUT_PLOTS = ROOT / "outputs" / "plots"
OUT_ART = ROOT / "outputs" / "artifacts"
OUT_RES = ROOT / "outputs" / "results"
for d in (OUT_PLOTS, OUT_ART, OUT_RES):
    d.mkdir(parents=True, exist_ok=True)


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main():
    results = {}

    # ---------------------------------------------------------------- 1&2
    section("1. LOAD DATA & FEATURE ENGINEERING")
    raw, source = dp.load_dataset()
    print(f"Loaded {len(raw):,} policies from source='{source}'")
    df = dp.engineer_features(raw)
    df = ph.build_cell_ph_features(df, cell_cols=["Area", "VehGas", "DrivAgeBucket"])
    results["data_source"] = source
    results["n_policies"] = int(len(df))
    results["claim_rate"] = float(df.HasClaim.mean())

    train_df, test_df = dp.train_test_split_by_policy(df)
    print(f"Train: {len(train_df):,}  |  Test: {len(test_df):,}  |  Claim rate: {df.HasClaim.mean():.4f}")

    claimants = df[df.HasClaim == 1].copy()
    print(f"Claimant subsample for MVN/MANOVA/QDA: {len(claimants):,}")

    # ---------------------------------------------------------------- 3
    section("2. BIVARIATE MVN FIT: (LogFrequency, LogAvgSeverity)")
    fit = mvn.fit_bivariate_mvn(claimants.LogFrequency.to_numpy(), claimants.LogAvgSeverity.to_numpy())
    print(f"mu = {fit.mu.round(4)}")
    print(f"Sigma =\n{np.round(fit.sigma, 4)}")
    print(f"Correlation rho(freq, severity) = {fit.rho:.4f}")
    results["mvn"] = {
        "mu_log_frequency": float(fit.mu[0]), "mu_log_severity": float(fit.mu[1]),
        "sigma": fit.sigma.round(6).tolist(), "rho": float(fit.rho), "n": fit.n,
    }

    # conditional distribution demo at 3 representative frequency levels
    cond_demo = {}
    for pct, label in [(0.10, "low_freq_p10"), (0.50, "median_freq_p50"), (0.90, "high_freq_p90")]:
        x1v = float(claimants.LogFrequency.quantile(pct))
        cmean, cvar = mvn.conditional_severity_given_frequency(fit, x1v)
        cond_demo[label] = {
            "log_frequency": x1v,
            "conditional_mean_log_severity": cmean,
            "conditional_sd_log_severity": float(np.sqrt(cvar)),
            "implied_median_severity": float(np.exp(cmean)),
        }
    results["conditional_severity_given_frequency"] = cond_demo
    print("\nConditional E[severity | frequency] at selected percentiles:")
    for k, v in cond_demo.items():
        print(f"  {k}: median severity ~= {v['implied_median_severity']:.0f}")

    viz.plot_mvn_scatter_contour(
        claimants.LogFrequency.to_numpy(), claimants.LogAvgSeverity.to_numpy(), fit,
        OUT_PLOTS / "01_mvn_scatter_contour.png",
        "Bivariate MVN fit: claim frequency vs average severity",
    )

    # ---------------------------------------------------------------- 4
    section("3. MAHALANOBIS DISTANCE / CHI-SQUARE OUTLIER TEST")
    d2 = mvn.mahalanobis_distances_sq(fit, claimants.LogFrequency.to_numpy(), claimants.LogAvgSeverity.to_numpy())
    pvals, is_outlier, crit, ks_p = mvn.chi_square_outlier_test(d2, dof=2, alpha=0.01)
    print(f"Critical D^2 (alpha=0.01, df=2) = {crit:.4f}")
    print(f"Flagged outliers: {is_outlier.sum():,} / {len(d2):,} ({is_outlier.mean():.2%})")
    print(f"KS test of D^2 vs chi2(2): p-value = {ks_p:.4f} "
          f"({'fails to reject MVN fit' if ks_p > 0.05 else 'rejects exact MVN fit'} at alpha=0.05)")
    results["outlier_test"] = {
        "critical_value_alpha_0.01": float(crit),
        "n_outliers": int(is_outlier.sum()),
        "outlier_rate": float(is_outlier.mean()),
        "ks_pvalue_vs_chi2": float(ks_p),
    }
    theo_q = mvn.theoretical_chi2_quantiles(len(d2), dof=2)
    viz.plot_chi2_qq(d2, theo_q, OUT_PLOTS / "02_chi2_qq_plot.png")
    viz.plot_mahalanobis_outliers(
        claimants.LogFrequency.to_numpy(), claimants.LogAvgSeverity.to_numpy(), d2, crit,
        OUT_PLOTS / "03_mahalanobis_outliers.png",
    )
    claimants = claimants.assign(mahalanobis_d2=d2, is_outlier=is_outlier)

    # ---------------------------------------------------------------- 5
    section("4. ONE-WAY MANOVA ACROSS POLICYHOLDER SEGMENTS")
    manova_res = manova.one_way_manova(claimants, "Segment", ["LogFrequency", "LogAvgSeverity"])
    print(manova.summarize(manova_res))
    results["manova"] = {
        "wilks_lambda": manova_res.wilks_lambda, "pillai_trace": manova_res.pillai_trace,
        "f_stat": manova_res.f_stat, "df1": manova_res.df1, "df2": manova_res.df2,
        "p_value": manova_res.p_value,
        "group_means": manova_res.group_means.round(4).to_dict(orient="index"),
    }
    viz.plot_manova_segment_means(manova_res, OUT_PLOTS / "04_manova_segment_means.png")

    # ---------------------------------------------------------------- 6
    section("5. QDA RISK-TIER CLASSIFICATION")
    claimants["RiskTier"] = qda.make_risk_tiers(claimants)
    c_train, c_test = dp.train_test_split_by_policy(
        claimants.rename(columns={"HasClaim": "_HasClaim"}).assign(HasClaim=1), test_size=0.25
    )
    # (stratify needs a non-constant column; reuse RiskTier for stratification instead)
    from sklearn.model_selection import train_test_split
    c_train, c_test = train_test_split(claimants, test_size=0.25, random_state=42, stratify=claimants["RiskTier"])
    qda_res = qda.fit_qda(c_train, c_test)
    print(f"QDA train accuracy: {qda_res.train_accuracy:.4f}  |  test accuracy: {qda_res.test_accuracy:.4f}")
    print(qda_res.report)
    results["qda"] = {
        "train_accuracy": qda_res.train_accuracy, "test_accuracy": qda_res.test_accuracy,
        "labels": qda_res.labels, "confusion_matrix": qda_res.confusion.tolist(),
    }
    xx, yy, zz = qda.decision_grid(
        qda_res.model,
        (claimants.LogFrequency.min() - 0.3, claimants.LogFrequency.max() + 0.3),
        (claimants.LogAvgSeverity.min() - 0.3, claimants.LogAvgSeverity.max() + 0.3),
    )
    viz.plot_qda_boundary(xx, yy, zz, c_test.LogFrequency.to_numpy(), c_test.LogAvgSeverity.to_numpy(),
                           c_test.RiskTier.to_numpy(), qda_res.labels, OUT_PLOTS / "05_qda_decision_boundary.png")
    joblib.dump(qda_res.model, OUT_ART / "qda_model.joblib")

    # ---------------------------------------------------------------- 7
    section("6. ML HIGH-RISK CLASSIFIER (pre-renewal flag)")
    ml_res = mlc.train_and_evaluate(train_df, test_df)
    print(f"High-risk threshold ({mlc.TARGET_QUANTILE:.0%} pct ClaimAmount): {ml_res.threshold_amount:.2f}")
    print(f"Test AUC = {ml_res.auc:.4f}  |  Test PR-AUC = {ml_res.pr_auc:.4f}")
    print(ml_res.report)
    print("Top 10 feature importances:")
    print(ml_res.feature_importance.head(10).to_string(index=False))
    results["ml_classifier"] = {
        "threshold_amount": ml_res.threshold_amount, "auc": ml_res.auc, "pr_auc": ml_res.pr_auc,
        "operating_threshold_proba": ml_res.operating_threshold_proba,
        "high_tier_threshold_proba": ml_res.high_tier_threshold_proba,
        "confusion_matrix": ml_res.confusion.tolist(),
        "top_features": ml_res.feature_importance.head(10).to_dict(orient="records"),
    }
    viz.plot_roc_pr(ml_res, OUT_PLOTS / "06_roc_pr_curves.png")
    viz.plot_feature_importance(ml_res.feature_importance, OUT_PLOTS / "07_feature_importance.png")
    joblib.dump(ml_res.pipeline, OUT_ART / "ml_high_risk_model.joblib")

    # ---------------------------------------------------------------- 8
    section("7. PERSIST ARTIFACTS FOR THE SCORING API")
    mvn_params = {"mu": fit.mu.tolist(), "sigma": fit.sigma.tolist(),
                  "threshold_d2_alpha01": float(crit)}
    (OUT_ART / "mvn_params.json").write_text(json.dumps(mvn_params, indent=2))

    ph_lookup = df.drop_duplicates(subset=["Area", "VehGas", "DrivAgeBucket"])[
        ["Area", "VehGas", "DrivAgeBucket"] + ph.PH_FEATURE_COLS
    ]
    ph_lookup.to_json(OUT_ART / "ph_cell_lookup.json", orient="records")

    (OUT_RES / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nAll artifacts written to {OUT_ART} and {OUT_RES}")
    print(f"All plots written to {OUT_PLOTS}")
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
