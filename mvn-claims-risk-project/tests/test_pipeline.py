"""
test_pipeline.py
==================
Lightweight assertion-based test suite (pytest is unavailable in this
offline sandbox, so this runs with plain `python3 tests/test_pipeline.py`
and exits non-zero on any failure -- easy to wire into CI as-is, or adapt
to pytest by prefixing each function with `test_` and running under
pytest, which will work unmodified once pytest is installed).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import mvn_analysis as mvn
from src import manova_analysis as manova
from src import ph_features as ph
from src import qda_classifier as qda
from src import data_processing as dp

PASSED, FAILED = [], []


def check(name, cond):
    if cond:
        PASSED.append(name)
        print(f"  [PASS] {name}")
    else:
        FAILED.append(name)
        print(f"  [FAIL] {name}")


def test_mvn_fit_recovers_known_parameters():
    print("\ntest_mvn_fit_recovers_known_parameters")
    rng = np.random.default_rng(0)
    true_mu = np.array([1.0, 5.0])
    true_sigma = np.array([[0.5, -0.2], [-0.2, 1.2]])
    X = rng.multivariate_normal(true_mu, true_sigma, size=50000)
    fit = mvn.fit_bivariate_mvn(X[:, 0], X[:, 1])
    check("mean recovered within tolerance", np.allclose(fit.mu, true_mu, atol=0.05))
    check("covariance recovered within tolerance", np.allclose(fit.sigma, true_sigma, atol=0.05))
    check("rho sign correct (negative)", fit.rho < 0)


def test_mahalanobis_and_chi_square_consistency():
    print("\ntest_mahalanobis_and_chi_square_consistency")
    rng = np.random.default_rng(1)
    mu = np.array([0.0, 0.0])
    sigma = np.eye(2)
    X = rng.multivariate_normal(mu, sigma, size=20000)
    fit = mvn.BivariateMVNFit(mu=mu, sigma=sigma, n=len(X))
    d2 = mvn.mahalanobis_distances_sq(fit, X[:, 0], X[:, 1])
    # for a true bivariate standard normal, D^2 should ~ chi2(2): mean ~2, and
    # roughly 1% should exceed the alpha=0.01 critical value by construction
    _, is_outlier, crit, ks_p = mvn.chi_square_outlier_test(d2, dof=2, alpha=0.01)
    check("mean D^2 close to dof=2", abs(d2.mean() - 2.0) < 0.15)
    check("outlier rate close to alpha=0.01", abs(is_outlier.mean() - 0.01) < 0.01)
    check("KS test does not reject true MVN sample (p > 0.01)", ks_p > 0.01)


def test_conditional_distribution_formula():
    print("\ntest_conditional_distribution_formula")
    # X2 | X1=mu1 should equal mu2 exactly (conditioning at the mean)
    fit = mvn.BivariateMVNFit(mu=np.array([2.0, 3.0]), sigma=np.array([[1.0, 0.5], [0.5, 2.0]]), n=1000)
    cmean, cvar = mvn.conditional_severity_given_frequency(fit, x1_value=2.0)
    check("conditional mean at x1=mu1 equals mu2", abs(cmean - 3.0) < 1e-9)
    check("conditional variance < marginal variance (info gain)", cvar < fit.sigma[1, 1])
    check("conditional variance formula matches manual calc",
          abs(cvar - (2.0 - 0.5 / 1.0 * 0.5)) < 1e-9)


def test_manova_detects_known_group_difference():
    print("\ntest_manova_detects_known_group_difference")
    rng = np.random.default_rng(2)
    n_per_group = 500
    groupA = rng.multivariate_normal([0, 0], np.eye(2), n_per_group)
    groupB = rng.multivariate_normal([3, 3], np.eye(2), n_per_group)  # clearly different mean
    df = pd.DataFrame(np.vstack([groupA, groupB]), columns=["v1", "v2"])
    df["grp"] = ["A"] * n_per_group + ["B"] * n_per_group
    result = manova.one_way_manova(df, "grp", ["v1", "v2"])
    check("Wilks Lambda << 1 for clearly separated groups", result.wilks_lambda < 0.2)
    check("p-value significant for clearly separated groups", result.p_value < 1e-6)

    df_same = pd.DataFrame(rng.multivariate_normal([0, 0], np.eye(2), n_per_group * 2), columns=["v1", "v2"])
    df_same["grp"] = rng.choice(["A", "B"], size=n_per_group * 2)
    result_same = manova.one_way_manova(df_same, "grp", ["v1", "v2"])
    check("Wilks Lambda close to 1 for identical-mean groups", result_same.wilks_lambda > 0.9)
    check("p-value not significant for identical-mean groups", result_same.p_value > 0.05)


def test_ph_moment_matching_roundtrips_moments():
    print("\ntest_ph_moment_matching_roundtrips_moments")
    for target_scv in [0.6, 1.0, 2.5]:
        mean = 1000.0
        params = ph.fit_ph2_moment_match(mean, target_scv)
        check(f"ph fit returns finite params for scv={target_scv}",
              all(np.isfinite(v) for k, v in params.items() if isinstance(v, (int, float))))
        check(f"ph_scv field matches input scv={target_scv}", abs(params["ph_scv"] - target_scv) < 1e-9)


def test_qda_separates_well_separated_tiers():
    print("\ntest_qda_separates_well_separated_tiers")
    rng = np.random.default_rng(3)
    n = 300
    low = rng.multivariate_normal([0, 0], 0.2 * np.eye(2), n)
    med = rng.multivariate_normal([2, 2], 0.2 * np.eye(2), n)
    high = rng.multivariate_normal([4, 4], 0.2 * np.eye(2), n)
    df = pd.DataFrame(np.vstack([low, med, high]), columns=["LogFrequency", "LogAvgSeverity"])
    df["RiskTier"] = ["Low"] * n + ["Medium"] * n + ["High"] * n
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df, test_size=0.3, random_state=0, stratify=df.RiskTier)
    result = qda.fit_qda(train, test)
    check("QDA test accuracy high on well-separated tiers", result.test_accuracy > 0.95)


def test_data_generation_and_feature_engineering_shapes():
    print("\ntest_data_generation_and_feature_engineering_shapes")
    import importlib
    gen_mod_path = ROOT / "data" / "generate_data.py"
    spec = importlib.util.spec_from_file_location("generate_data", gen_mod_path)
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    df = gen.generate(n_policies=2000, seed=99)
    check("generated dataframe has expected n rows", len(df) == 2000)
    check("claim rate is plausible (1%-60%)", 0.01 < df.HasClaim.mean() < 0.6)
    check("no negative claim amounts", (df.ClaimAmount >= 0).all())

    feat = dp.engineer_features(df)
    check("LogFrequency finite for all rows", np.isfinite(feat.LogFrequency).all())
    check("required engineered columns present", {"LogFrequency", "LogAvgSeverity", "Segment"}.issubset(feat.columns))


if __name__ == "__main__":
    test_mvn_fit_recovers_known_parameters()
    test_mahalanobis_and_chi_square_consistency()
    test_conditional_distribution_formula()
    test_manova_detects_known_group_difference()
    test_ph_moment_matching_roundtrips_moments()
    test_qda_separates_well_separated_tiers()
    test_data_generation_and_feature_engineering_shapes()

    print("\n" + "=" * 60)
    print(f"RESULTS: {len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("FAILED:", FAILED)
        sys.exit(1)
    print("All tests passed.")
