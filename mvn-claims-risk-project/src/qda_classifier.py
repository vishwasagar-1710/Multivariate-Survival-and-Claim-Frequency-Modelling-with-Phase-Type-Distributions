"""
qda_classifier.py
===================
Quadratic Discriminant Analysis (QDA) for a 3-class risk tier
(Low / Medium / High), where the labels are themselves derived from the
fitted joint (LogFrequency, LogAvgSeverity) distribution: a policy's
"risk score" is its estimated expected cost proxy

    risk_score = LogFrequency + LogAvgSeverity   (~ log expected cost)

and Low/Medium/High are its sample tertiles among claimants. QDA is a
natural classifier here (rather than LDA) because the project explicitly
expects *different* covariance structures per risk tier -- high-risk
claimants are hypothesised to have a different frequency-severity
correlation than low-risk claimants (fatter joint tails), which QDA can
represent (separate Sigma_k per class) and LDA (shared Sigma) cannot.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix


@dataclass
class QDAResult:
    model: QuadraticDiscriminantAnalysis
    labels: list[str]
    train_accuracy: float
    test_accuracy: float
    report: str
    confusion: np.ndarray


def make_risk_tiers(df: pd.DataFrame) -> pd.Series:
    """Tertile risk_score labels for claimant policies."""
    risk_score = df["LogFrequency"] + df["LogAvgSeverity"]
    tiers = pd.qcut(risk_score, q=3, labels=["Low", "Medium", "High"])
    return tiers


def fit_qda(train_df: pd.DataFrame, test_df: pd.DataFrame,
            feature_cols=("LogFrequency", "LogAvgSeverity")) -> QDAResult:
    Xtr = train_df[list(feature_cols)].to_numpy()
    ytr = train_df["RiskTier"].to_numpy()
    Xte = test_df[list(feature_cols)].to_numpy()
    yte = test_df["RiskTier"].to_numpy()

    model = QuadraticDiscriminantAnalysis(store_covariance=True)
    model.fit(Xtr, ytr)

    train_acc = model.score(Xtr, ytr)
    test_acc = model.score(Xte, yte)
    y_pred = model.predict(Xte)
    labels = sorted(pd.unique(ytr).tolist())
    report = classification_report(yte, y_pred, labels=labels)
    cm = confusion_matrix(yte, y_pred, labels=labels)

    return QDAResult(model=model, labels=labels, train_accuracy=train_acc,
                      test_accuracy=test_acc, report=report, confusion=cm)


def decision_grid(model: QuadraticDiscriminantAnalysis, x_range, y_range, n=200):
    """Grid of predicted classes for plotting QDA decision boundaries."""
    xx, yy = np.meshgrid(np.linspace(*x_range, n), np.linspace(*y_range, n))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    zz = model.predict(grid).reshape(xx.shape)
    return xx, yy, zz
