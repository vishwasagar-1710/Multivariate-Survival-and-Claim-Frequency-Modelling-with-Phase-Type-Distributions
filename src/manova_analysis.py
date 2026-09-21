"""
manova_analysis.py
====================
One-way Multivariate Analysis of Variance (MANOVA), implemented from the
course-theory linear-algebra definition (statsmodels is unavailable in this
environment, so this is a from-scratch implementation rather than a
library call -- it also makes the mechanics fully transparent).

Tests H0: the mean vector of (LogFrequency, LogAvgSeverity) is identical
across policyholder segments (Young/Senior/MidAge x Urban/Rural), i.e. that
claim *behaviour* (not just raw claim counts) does not differ by segment.

Construction
------------
For p=2 response variables and g groups:
  T = total scatter matrix            = sum_i (x_i - grand_mean)(x_i - grand_mean)'
  W = within-groups scatter matrix    = sum_k sum_{i in k} (x_i - mean_k)(x_i - mean_k)'
  B = between-groups scatter matrix   = T - W
Test statistics from the eigenvalues lambda_j of W^-1 B:
  Wilks' Lambda  = det(W) / det(T) = prod_j 1 / (1 + lambda_j)
  Pillai's trace = sum_j lambda_j / (1 + lambda_j)
Wilks' Lambda is converted to an (approximate, Rao's F) F-statistic with the
standard degrees of freedom for p=2 response variables, which is exact for
p<=2 or g<=3 and an accurate approximation otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ManovaResult:
    wilks_lambda: float
    pillai_trace: float
    f_stat: float
    df1: float
    df2: float
    p_value: float
    n: int
    p_vars: int
    g_groups: int
    group_means: pd.DataFrame
    eigenvalues: np.ndarray


def one_way_manova(df: pd.DataFrame, group_col: str, response_cols: list[str]) -> ManovaResult:
    groups = df[group_col].unique()
    g = len(groups)
    p = len(response_cols)
    X = df[response_cols].to_numpy(dtype=float)
    n = X.shape[0]
    grand_mean = X.mean(axis=0)

    T = np.zeros((p, p))
    W = np.zeros((p, p))
    group_means = {}
    for grp in groups:
        Xk = df.loc[df[group_col] == grp, response_cols].to_numpy(dtype=float)
        mean_k = Xk.mean(axis=0)
        group_means[grp] = mean_k
        diff_k = Xk - mean_k
        W += diff_k.T @ diff_k
    diff_total = X - grand_mean
    T = diff_total.T @ diff_total
    B = T - W

    # eigenvalues of W^-1 B (solve generalized eigenproblem for stability)
    eigvals = np.linalg.eigvals(np.linalg.solve(W, B)).real
    eigvals = np.clip(eigvals, 0, None)
    eigvals_sorted = np.sort(eigvals)[::-1]
    s = min(p, g - 1)
    eigvals_used = eigvals_sorted[:s]

    wilks = np.prod(1.0 / (1.0 + eigvals_used))
    pillai = np.sum(eigvals_used / (1.0 + eigvals_used))

    # Rao's F approximation for Wilks' Lambda
    vh = g - 1               # hypothesis df
    ve = n - g                # error df
    if p == 1 or vh == 1:
        df1 = p * vh
        df2 = ve - p + 1
        f_stat = ((1 - wilks) / wilks) * (df2 / df1) if wilks > 0 else np.inf
    else:
        wprod = p * p + vh * vh - 5
        t = np.sqrt((p * p * vh * vh - 4) / wprod) if wprod > 0 else 1.0
        df1 = p * vh
        df2 = t * (ve + vh - 0.5 * (p + vh + 1)) - 0.5 * (p * vh - 2)
        wilks_pow = wilks ** (1.0 / t)
        f_stat = ((1 - wilks_pow) / wilks_pow) * (df2 / df1) if wilks_pow > 0 else np.inf

    p_value = 1 - stats.f.cdf(f_stat, df1, df2)

    means_df = pd.DataFrame(group_means, index=response_cols).T
    means_df.index.name = group_col

    return ManovaResult(
        wilks_lambda=float(wilks), pillai_trace=float(pillai), f_stat=float(f_stat),
        df1=float(df1), df2=float(df2), p_value=float(p_value), n=n, p_vars=p,
        g_groups=g, group_means=means_df, eigenvalues=eigvals_used,
    )


def summarize(result: ManovaResult) -> str:
    lines = [
        f"One-way MANOVA: {result.g_groups} groups, {result.p_vars} response variables, n={result.n}",
        f"  Wilks' Lambda  = {result.wilks_lambda:.5f}",
        f"  Pillai's Trace = {result.pillai_trace:.5f}",
        f"  F({result.df1:.1f}, {result.df2:.1f}) = {result.f_stat:.3f}, p-value = {result.p_value:.3e}",
        "",
        "Group means:",
        result.group_means.round(4).to_string(),
    ]
    return "\n".join(lines)
