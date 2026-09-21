"""
mvn_analysis.py
================
Core multivariate-statistics module. Implements, from first principles
(course-theory level, using numpy/scipy linear algebra rather than a
black-box library call):

  1. Full bivariate normal (MVN) density estimation on the joint vector
     X = (LogFrequency, LogAvgSeverity) for claimant policies.
  2. The conditional distribution of severity given frequency,
     X2 | X1 = x1 ~ N(mu2 + Sigma21 Sigma11^-1 (x1 - mu1),
                       Sigma22 - Sigma21 Sigma11^-1 Sigma12)
     -- the textbook conditional-MVN formula.
  3. Mahalanobis distance D^2(x) = (x - mu)' Sigma^-1 (x - mu) for every
     claimant, and the associated chi-square goodness-of-fit / outlier test:
     under H0 (x ~ MVN), D^2 ~ chi2_p. We use this both to (a) test overall
     MVN fit via a QQ-plot against chi2_2 quantiles, and (b) flag individual
     policies whose joint (frequency, severity) profile is a statistical
     outlier at a chosen alpha (e.g. 0.01).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class BivariateMVNFit:
    mu: np.ndarray          # (2,) mean vector [mu_freq, mu_sev]
    sigma: np.ndarray        # (2,2) covariance matrix
    n: int                    # sample size used to fit
    labels: tuple[str, str] = ("LogFrequency", "LogAvgSeverity")

    @property
    def rho(self) -> float:
        return self.sigma[0, 1] / np.sqrt(self.sigma[0, 0] * self.sigma[1, 1])

    @property
    def sigma_inv(self) -> np.ndarray:
        return np.linalg.inv(self.sigma)


def fit_bivariate_mvn(x1: np.ndarray, x2: np.ndarray) -> BivariateMVNFit:
    """Maximum-likelihood fit of a bivariate normal: mu = sample mean,
    Sigma = sample covariance (MLE uses 1/n; we report the standard
    unbiased 1/(n-1) estimator, the conventional choice for inference)."""
    X = np.column_stack([x1, x2])
    mu = X.mean(axis=0)
    sigma = np.cov(X, rowvar=False, ddof=1)
    return BivariateMVNFit(mu=mu, sigma=sigma, n=len(X))


def mvn_density(fit: BivariateMVNFit, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Evaluate the fitted bivariate normal density at points (x1, x2)."""
    rv = stats.multivariate_normal(mean=fit.mu, cov=fit.sigma)
    pts = np.column_stack([x1, x2])
    return rv.pdf(pts)


def conditional_severity_given_frequency(fit: BivariateMVNFit, x1_value: float):
    """Textbook MVN conditioning: distribution of X2 (LogAvgSeverity) given
    X1 = x1_value (LogFrequency). Returns (cond_mean, cond_var)."""
    mu1, mu2 = fit.mu
    s11, s12 = fit.sigma[0, 0], fit.sigma[0, 1]
    s21, s22 = fit.sigma[1, 0], fit.sigma[1, 1]
    cond_mean = mu2 + s21 / s11 * (x1_value - mu1)
    cond_var = s22 - s21 / s11 * s12
    return cond_mean, cond_var


def mahalanobis_distances_sq(fit: BivariateMVNFit, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """D^2_i = (x_i - mu)' Sigma^-1 (x_i - mu) for every observation."""
    X = np.column_stack([x1, x2])
    diff = X - fit.mu
    sinv = fit.sigma_inv
    # (n,2) @ (2,2) -> (n,2), then rowwise dot with diff
    return np.einsum("ij,jk,ik->i", diff, sinv, diff)


def chi_square_outlier_test(d2: np.ndarray, dof: int = 2, alpha: float = 0.01):
    """Under H0 (bivariate normality), D^2 ~ chi2_dof.
    Returns (p_values, is_outlier_mask, critical_value, ks_pvalue) where
    ks_pvalue is a Kolmogorov-Smirnov goodness-of-fit p-value comparing the
    empirical D^2 distribution to chi2_dof (an overall MVN-fit diagnostic,
    distinct from the per-point outlier flags)."""
    p_values = 1 - stats.chi2.cdf(d2, df=dof)
    crit = stats.chi2.ppf(1 - alpha, df=dof)
    is_outlier = d2 > crit
    ks_stat, ks_p = stats.kstest(d2, "chi2", args=(dof,))
    return p_values, is_outlier, crit, ks_p


def theoretical_chi2_quantiles(n: int, dof: int = 2) -> np.ndarray:
    """Quantiles used for a chi-square QQ plot of the Mahalanobis distances."""
    probs = (np.arange(1, n + 1) - 0.5) / n
    return stats.chi2.ppf(probs, df=dof)
