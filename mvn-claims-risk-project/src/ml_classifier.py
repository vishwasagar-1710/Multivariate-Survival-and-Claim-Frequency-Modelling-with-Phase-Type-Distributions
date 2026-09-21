"""
ml_classifier.py
==================
Pre-renewal high-risk classifier.

Target definition
------------------
A policy is labelled HIGH_RISK = 1 if its realised total claim cost
(ClaimAmount) exceeds the 95th percentile of the portfolio's loss-cost
distribution -- a standard actuarial "large loss" flag that automatically
captures both high-frequency and high-severity claimants (and, per the
project's dependence thesis, policies that combine moderate frequency with
elevated severity, or vice versa).

Feature set (all available BEFORE renewal, i.e. no leakage from this
period's own claim outcome):
  - standard rating factors: VehPower, VehAge, DrivAge, BonusMalus, Area,
    Density, Region, VehGas, VehBrand, Exposure
  - MAP/PH-inspired engineered features from ph_features.py: the rating
    cell's moment-matched phase-type descriptors (ph_p, ph_rate1, ph_rate2,
    ph_scv, cell_mean_severity, cell_var_severity), which summarise how
    "spiky" vs "regular" claims tend to be for similar policyholders.

Model
-----
GradientBoostingClassifier (sklearn) inside a ColumnTransformer pipeline
(one-hot encoding for categoricals, pass-through/scaling for numerics).
Chosen over a plain GLM because the frequency-severity dependence and PH
features are expected to interact non-additively with rating factors.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    roc_curve, precision_recall_curve, confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

NUMERIC_FEATURES = [
    "VehPower", "VehAge", "DrivAge", "BonusMalus", "LogDensity", "Exposure",
    "cell_mean_severity", "cell_var_severity", "ph_p", "ph_rate1", "ph_rate2", "ph_scv",
]
CATEGORICAL_FEATURES = ["Area", "VehGas", "Region", "VehBrand"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_QUANTILE = 0.98


def make_target(df: pd.DataFrame, quantile: float = TARGET_QUANTILE) -> pd.Series:
    threshold = df["ClaimAmount"].quantile(quantile)
    return (df["ClaimAmount"] > threshold).astype(int), threshold


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    clf = GradientBoostingClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


@dataclass
class MLResult:
    pipeline: Pipeline
    threshold_amount: float
    auc: float
    pr_auc: float
    report: str
    confusion: np.ndarray
    fpr: np.ndarray
    tpr: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    feature_importance: pd.DataFrame
    operating_threshold_proba: float
    high_tier_threshold_proba: float


def train_and_evaluate(train_df: pd.DataFrame, test_df: pd.DataFrame) -> MLResult:
    ytr, thr = make_target(train_df)
    # use the TRAIN threshold amount consistently for both splits to avoid leakage
    yte = (test_df["ClaimAmount"] > thr).astype(int)

    pipe = build_pipeline()
    sample_weight = compute_sample_weight(class_weight="balanced", y=ytr)
    pipe.fit(train_df[ALL_FEATURES], ytr, clf__sample_weight=sample_weight)

    proba = pipe.predict_proba(test_df[ALL_FEATURES])[:, 1]

    auc = roc_auc_score(yte, proba)
    pr_auc = average_precision_score(yte, proba)

    # Business operating point: flag the same proportion of the book as
    # the target prevalence (e.g. "underwrite/review the riskiest 1-Q%"),
    # which is far more informative here than an arbitrary 0.5 cutoff on a
    # rare-event (imbalanced) target.
    positive_rate = ytr.mean()
    op_threshold = float(np.quantile(proba, 1 - positive_rate))
    high_tier_threshold = float(np.quantile(proba, 1 - positive_rate / 2))
    pred = (proba >= op_threshold).astype(int)
    report = classification_report(yte, pred, digits=3, zero_division=0)
    cm = confusion_matrix(yte, pred)
    fpr, tpr, _ = roc_curve(yte, proba)
    precision, recall, _ = precision_recall_curve(yte, proba)

    # feature importances mapped back to readable names
    ohe = pipe.named_steps["pre"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    feat_names = NUMERIC_FEATURES + cat_names
    importances = pipe.named_steps["clf"].feature_importances_
    fi = pd.DataFrame({"feature": feat_names, "importance": importances}) \
        .sort_values("importance", ascending=False).reset_index(drop=True)

    return MLResult(pipeline=pipe, threshold_amount=float(thr), auc=auc, pr_auc=pr_auc,
                     report=report, confusion=cm, fpr=fpr, tpr=tpr,
                     precision=precision, recall=recall, feature_importance=fi,
                     operating_threshold_proba=op_threshold,
                     high_tier_threshold_proba=high_tier_threshold)
