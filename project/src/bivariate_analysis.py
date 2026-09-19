"""
Multivariate statistical analysis of the (log-frequency, log-severity)
claim vector, per the course-theory requirements of the project:

  1. Full bivariate normal (MVN) density estimation on the joint
     claim vector (LogFrequency, LogSeverity).
  2. Conditional distribution of severity given frequency, derived
     analytically from the fitted MVN parameters.
  3. Mahalanobis distance of every claimant from the fitted MVN mean,
     with a chi-square (df=2) test used to flag multivariate outliers
     ("anomalous claims" - e.g. suspicious or catastrophic losses).
  4. MANOVA testing whether the mean claim vector differs across
     policyholder segments (Region, VehGas, Area).

Outputs: reports/analysis_summary.json and PNG figures in figures/.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.multivariate.manova import MANOVA

from data_prep import get_datasets

FIG_DIR = "figures"
REPORT_DIR = "reports"


# ---------------------------------------------------------------- MVN ----
def fit_mvn(claimant_df: pd.DataFrame):
    X = claimant_df[["LogFrequency", "LogSeverity"]].to_numpy()
    mu = X.mean(axis=0)
    sigma = np.cov(X, rowvar=False)
    return mu, sigma, X


def mvn_density(x, mu, sigma):
    return stats.multivariate_normal(mean=mu, cov=sigma).pdf(x)


def conditional_severity_given_frequency(mu, sigma, freq_value):
    """
    For X = (F, S) ~ N(mu, sigma), the conditional distribution of
    S | F = f is Normal with:
        mean_cond = mu_S + sigma_SF/sigma_FF * (f - mu_F)
        var_cond  = sigma_SS - sigma_SF^2 / sigma_FF
    """
    mu_f, mu_s = mu
    sigma_ff, sigma_fs = sigma[0, 0], sigma[0, 1]
    sigma_ss = sigma[1, 1]
    mean_cond = mu_s + (sigma_fs / sigma_ff) * (freq_value - mu_f)
    var_cond = sigma_ss - (sigma_fs ** 2) / sigma_ff
    return mean_cond, var_cond


# --------------------------------------------------- Mahalanobis / chi2 ----
def mahalanobis_outliers(X, mu, sigma, alpha: float = 0.01):
    inv_sigma = np.linalg.inv(sigma)
    diff = X - mu
    d2 = np.einsum("ij,jk,ik->i", diff, inv_sigma, diff)  # squared Mahalanobis
    threshold = stats.chi2.ppf(1 - alpha, df=2)
    is_outlier = d2 > threshold
    return d2, threshold, is_outlier


# --------------------------------------------------------------- MANOVA ----
def run_manova(claimant_df: pd.DataFrame, segment_col: str):
    formula = f"LogFrequency + LogSeverity ~ C({segment_col})"
    model = MANOVA.from_formula(formula, data=claimant_df)
    result = model.mv_test()
    return result


def manova_summary_row(result, segment_col: str):
    """Extract Pillai's trace / Wilks' lambda test row as a plain dict."""
    key = f"C({segment_col})"
    table = result.results[key]["stat"]
    row = table.loc["Pillai's trace"]
    return {
        "segment": segment_col,
        "test_stat": float(row["Value"]),
        "F": float(row["F Value"]),
        "df_num": float(row["Num DF"]),
        "df_den": float(row["Den DF"]),
        "p_value": float(row["Pr > F"]),
    }


# ---------------------------------------------------------------- plots ----
def make_plots(claimant_df: pd.DataFrame, mu, sigma, d2, threshold):
    X = claimant_df[["LogFrequency", "LogSeverity"]].to_numpy()

    # 1. Scatter + MVN contours
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(X[:, 0], X[:, 1], s=6, alpha=0.25, color="#3b6fa0",
               label="Claimant policies")
    xg = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 200)
    yg = np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 200)
    Xg, Yg = np.meshgrid(xg, yg)
    pos = np.dstack((Xg, Yg))
    rv = stats.multivariate_normal(mu, sigma)
    ax.contour(Xg, Yg, rv.pdf(pos), levels=8, cmap="Reds")
    ax.set_xlabel("log(Annual Claim Frequency)")
    ax.set_ylabel("log(Average Severity, EUR)")
    ax.set_title("Fitted Bivariate Normal: (log-Frequency, log-Severity)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/mvn_contours.png", dpi=140)
    plt.close(fig)

    # 2. Mahalanobis distance distribution vs chi2
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(d2, bins=60, density=True, alpha=0.6, color="#548c6a",
            label="Observed squared Mahalanobis distance")
    xs = np.linspace(0, d2.max(), 300)
    ax.plot(xs, stats.chi2.pdf(xs, df=2), color="black", lw=2,
            label="chi2(df=2) reference")
    ax.axvline(threshold, color="red", ls="--",
               label=f"99% threshold = {threshold:.2f}")
    ax.set_xlabel("Squared Mahalanobis distance")
    ax.set_ylabel("Density")
    ax.set_title("Mahalanobis Distance Outlier Test (claimant policies)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/mahalanobis_outliers.png", dpi=140)
    plt.close(fig)


def main():
    portfolio_df, claimant_df = get_datasets()

    mu, sigma, X = fit_mvn(claimant_df)
    corr = sigma[0, 1] / np.sqrt(sigma[0, 0] * sigma[1, 1])

    mean_cond_at_mean, var_cond = conditional_severity_given_frequency(
        mu, sigma, freq_value=mu[0])
    mean_cond_high_freq, _ = conditional_severity_given_frequency(
        mu, sigma, freq_value=mu[0] + 2 * np.sqrt(sigma[0, 0]))

    d2, threshold, is_outlier = mahalanobis_outliers(X, mu, sigma, alpha=0.01)

    manova_rows = []
    for seg in ["Region", "VehGas", "Area"]:
        try:
            result = run_manova(claimant_df, seg)
            manova_rows.append(manova_summary_row(result, seg))
        except Exception as e:
            manova_rows.append({"segment": seg, "error": str(e)})

    make_plots(claimant_df, mu, sigma, d2, threshold)

    summary = {
        "n_claimant_policies": int(len(claimant_df)),
        "n_portfolio_policies": int(len(portfolio_df)),
        "mvn_mean": {"LogFrequency": float(mu[0]), "LogSeverity": float(mu[1])},
        "mvn_cov": sigma.tolist(),
        "mvn_correlation_freq_sev": float(corr),
        "conditional_severity_given_mean_frequency": {
            "mean_log_severity": float(mean_cond_at_mean),
            "var_log_severity": float(var_cond),
            "implied_severity_eur": float(np.exp(mean_cond_at_mean + var_cond / 2)),
        },
        "conditional_severity_given_high_frequency_plus2sd": {
            "mean_log_severity": float(mean_cond_high_freq),
            "implied_severity_eur": float(np.exp(mean_cond_high_freq + var_cond / 2)),
        },
        "mahalanobis_outliers": {
            "chi2_threshold_df2_alpha01": float(threshold),
            "n_outliers": int(is_outlier.sum()),
            "pct_outliers": float(is_outlier.mean() * 100),
        },
        "manova_by_segment": manova_rows,
    }

    with open(f"{REPORT_DIR}/analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
