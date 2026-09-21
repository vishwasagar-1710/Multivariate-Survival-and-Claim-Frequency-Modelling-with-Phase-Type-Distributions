"""plotting.py -- all matplotlib figure-generation used by scripts/run_pipeline.py"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.grid": True, "grid.alpha": 0.25})


def plot_mvn_scatter_contour(x1, x2, fit, path, title):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x1, x2, s=6, alpha=0.25, color="#3b6ea5", label="Claimant policies")

    xs = np.linspace(x1.min() - 0.5, x1.max() + 0.5, 200)
    ys = np.linspace(x2.min() - 0.5, x2.max() + 0.5, 200)
    XX, YY = np.meshgrid(xs, ys)
    rv = stats.multivariate_normal(mean=fit.mu, cov=fit.sigma)
    ZZ = rv.pdf(np.dstack([XX, YY]))
    ax.contour(XX, YY, ZZ, levels=8, colors="#c0392b", linewidths=1.1)
    ax.scatter(*fit.mu, color="black", marker="x", s=80, label="MVN mean", zorder=5)
    ax.set_xlabel("Log(Frequency)")
    ax.set_ylabel("Log(Avg Severity)")
    ax.set_title(f"{title}\n(rho = {fit.rho:.3f})")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_chi2_qq(d2, theo_q, path):
    fig, ax = plt.subplots(figsize=(6, 6))
    d2_sorted = np.sort(d2)
    ax.scatter(theo_q, d2_sorted, s=8, alpha=0.4, color="#3b6ea5")
    lims = [0, max(theo_q.max(), d2_sorted.max())]
    ax.plot(lims, lims, color="#c0392b", linestyle="--", label="y = x (perfect fit)")
    ax.set_xlabel("Theoretical chi-square(2) quantiles")
    ax.set_ylabel("Sample Mahalanobis D^2 quantiles")
    ax.set_title("Chi-square Q-Q plot: MVN goodness-of-fit")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_mahalanobis_outliers(x1, x2, d2, crit, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    outlier = d2 > crit
    ax.scatter(x1[~outlier], x2[~outlier], s=6, alpha=0.3, color="#3b6ea5", label="Typical")
    ax.scatter(x1[outlier], x2[outlier], s=18, alpha=0.8, color="#c0392b", label=f"Outlier (D2 > {crit:.2f})")
    ax.set_xlabel("Log(Frequency)")
    ax.set_ylabel("Log(Avg Severity)")
    ax.set_title(f"Mahalanobis-distance outlier detection ({outlier.sum()} flagged)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_manova_segment_means(result, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    gm = result.group_means
    ax.scatter(gm.iloc[:, 0], gm.iloc[:, 1], s=90, color="#c0392b", zorder=5)
    for idx, row in gm.iterrows():
        ax.annotate(str(idx), (row.iloc[0], row.iloc[1]), textcoords="offset points",
                    xytext=(6, 6), fontsize=9)
    ax.set_xlabel(gm.columns[0])
    ax.set_ylabel(gm.columns[1])
    ax.set_title(f"Segment mean claim behaviour (MANOVA)\nWilks' Lambda={result.wilks_lambda:.4f}, "
                 f"p={result.p_value:.2e}")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_qda_boundary(xx, yy, zz, x1, x2, y, labels, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    label_to_int = {l: i for i, l in enumerate(labels)}
    zz_int = np.vectorize(label_to_int.get)(zz)
    ax.contourf(xx, yy, zz_int, alpha=0.25, levels=len(labels) - 1, cmap="RdYlGn_r")
    colors = {"Low": "#2e7d32", "Medium": "#f9a825", "High": "#c0392b"}
    for lab in labels:
        mask = y == lab
        ax.scatter(x1[mask], x2[mask], s=8, alpha=0.5, label=lab, color=colors.get(lab))
    ax.set_xlabel("Log(Frequency)")
    ax.set_ylabel("Log(Avg Severity)")
    ax.set_title("QDA decision regions: risk tier classification")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_roc_pr(ml_result, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].plot(ml_result.fpr, ml_result.tpr, color="#3b6ea5", label=f"AUC={ml_result.auc:.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC curve")
    axes[0].legend(fontsize=8)

    axes[1].plot(ml_result.recall, ml_result.precision, color="#c0392b",
                 label=f"PR-AUC={ml_result.pr_auc:.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall curve")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_feature_importance(fi_df, path, top_n=15):
    fig, ax = plt.subplots(figsize=(7, 6))
    top = fi_df.head(top_n).iloc[::-1]
    ax.barh(top["feature"], top["importance"], color="#3b6ea5")
    ax.set_xlabel("Gradient boosting feature importance")
    ax.set_title("Top predictors of high-risk policy flag")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
