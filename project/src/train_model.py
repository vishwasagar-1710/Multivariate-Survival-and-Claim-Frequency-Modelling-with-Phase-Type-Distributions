"""
Trains the risk-scoring classifier for the API.

Primary technique (course-theory requirement): Quadratic Discriminant
Analysis (QDA), which fits a class-conditional multivariate normal to
the engineered feature space and classifies each policy into
Low / Medium / High risk based on posterior probability - a direct
extension of the MVN theory used in bivariate_analysis.py to a
K-class, p-dimensional setting.

Benchmark: Gradient Boosting classifier, included to contextualize how
much predictive lift a flexible, non-parametric ML model gets over the
distributional QDA baseline on the same features/labels.

Artifacts saved to models/:
  - preprocessor.joblib   (ColumnTransformer: scaling + one-hot)
  - qda_model.joblib
  - gbm_model.joblib
  - mvn_params.joblib     (mu, sigma, from bivariate_analysis, for the
                            Mahalanobis anomaly flag exposed by the API)
  - feature_schema.json   (expected input fields + categorical levels)
"""
from __future__ import annotations
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.inspection import permutation_importance

from data_prep import get_datasets, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from bivariate_analysis import fit_mvn

MODEL_DIR = "models"
REPORT_DIR = "reports"

ENGINEERED_NUMERIC = NUMERIC_FEATURES + [
    "RiskLoadIndex", "VehicleRiskIndex", "DriverExperienceProxy", "UrbanicityIndex"
]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ENGINEERED_NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def compute_explainability_artifacts(gbm, preprocessor, X_test, y_test, portfolio_df):
    """
    Precompute everything the /explain endpoint needs to ground an LLM
    narrative in real numbers rather than having it invent reasons:
      - global permutation importance of the GBM model (which features,
        including individual one-hot categorical levels, actually move
        predictions)
      - portfolio-wide quantiles of each numeric feature (so a single
        policy's value can be expressed as "82nd percentile" etc.)
      - historical risk stats per categorical level (mean pure premium,
        % Medium/High) so e.g. a specific Region can be described
        against its own track record, not just a guess.
    """
    print("[train_model] computing permutation importance (this takes ~1-2 min)...")
    # Subsample for speed; permutation importance only needs to be
    # directionally right at the global level, not exact.
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(X_test), size=min(8000, len(X_test)), replace=False)
    X_sample = X_test[sample_idx]
    y_sample = y_test.iloc[sample_idx] if hasattr(y_test, "iloc") else y_test[sample_idx]

    perm = permutation_importance(
        gbm, X_sample, y_sample, scoring="roc_auc_ovr",
        n_repeats=3, random_state=42, n_jobs=-1,
    )
    feature_names = list(preprocessor.get_feature_names_out())
    importance_rows = sorted(
        [{"feature": f, "importance": float(imp)}
         for f, imp in zip(feature_names, perm.importances_mean)],
        key=lambda r: -r["importance"]
    )
    with open(f"{MODEL_DIR}/feature_importance.json", "w") as f:
        json.dump(importance_rows, f, indent=2)

    # Numeric feature distributions (raw units, on the full portfolio).
    dist = {}
    for feat in ENGINEERED_NUMERIC:
        q = portfolio_df[feat].quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        dist[feat] = {
            "p5": float(q.loc[0.05]), "p25": float(q.loc[0.25]),
            "p50": float(q.loc[0.5]), "p75": float(q.loc[0.75]),
            "p95": float(q.loc[0.95]),
            "min": float(portfolio_df[feat].min()),
            "max": float(portfolio_df[feat].max()),
        }
    with open(f"{MODEL_DIR}/feature_distributions.json", "w") as f:
        json.dump(dist, f, indent=2)

    # Historical risk profile per categorical level.
    cat_stats = {}
    portfolio_high_med = portfolio_df["RiskTier"].isin(["Medium", "High"])
    for cat in CATEGORICAL_FEATURES:
        cat_stats[cat] = {}
        grouped = portfolio_df.groupby(cat)
        for level, group in grouped:
            n = len(group)
            cat_stats[cat][str(level)] = {
                "n_policies": int(n),
                "mean_pure_premium": float(group["PurePremium"].mean()),
                "pct_medium_or_high": float(portfolio_high_med[group.index].mean() * 100),
            }
    with open(f"{MODEL_DIR}/categorical_risk_stats.json", "w") as f:
        json.dump(cat_stats, f, indent=2)

    print("[train_model] explainability artifacts saved: "
          "feature_importance.json, feature_distributions.json, "
          "categorical_risk_stats.json")


def train_and_evaluate():
    portfolio_df, claimant_df = get_datasets()

    X = portfolio_df[ENGINEERED_NUMERIC + CATEGORICAL_FEATURES]
    y = portfolio_df["RiskTier"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()
    Xtr = preprocessor.fit_transform(X_train)
    Xte = preprocessor.transform(X_test)
    # QDA needs dense arrays
    if hasattr(Xtr, "toarray"):
        Xtr = Xtr.toarray()
        Xte = Xte.toarray()

    classes = sorted(y.unique())  # ['High','Low','Medium'] alpha order
    n_classes = len(classes)

    # ---- QDA: balanced priors so the minority "High" risk class isn't
    # simply swamped by the ~96% "Low" base rate. ----
    qda = QuadraticDiscriminantAnalysis(priors=np.full(n_classes, 1 / n_classes),
                                         reg_param=0.15)
    qda.fit(Xtr, y_train)
    qda_pred = qda.predict(Xte)
    qda_proba = qda.predict_proba(Xte)

    # ---- Benchmark: (histogram-based) Gradient Boosting, chosen for
    # speed on ~500K training rows vs. classic GradientBoostingClassifier ----
    gbm = HistGradientBoostingClassifier(random_state=42, max_iter=200,
                                          max_depth=4, learning_rate=0.08)
    gbm.fit(Xtr, y_train)
    gbm_pred = gbm.predict(Xte)
    gbm_proba = gbm.predict_proba(Xte)

    def eval_model(name, y_true, y_pred, y_proba, model_classes):
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=model_classes).tolist()
        y_true_bin = label_binarize(y_true, classes=model_classes)
        try:
            auc = roc_auc_score(y_true_bin, y_proba, multi_class="ovr", average="macro")
        except Exception:
            auc = None
        return {
            "classification_report": report,
            "confusion_matrix": cm,
            "labels_order": list(model_classes),
            "macro_auc_ovr": auc,
        }

    results = {
        "QDA": eval_model("QDA", y_test, qda_pred, qda_proba, qda.classes_),
        "GradientBoosting": eval_model("GBM", y_test, gbm_pred, gbm_proba, gbm.classes_),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "class_balance_test": y_test.value_counts().to_dict(),
    }

    with open(f"{REPORT_DIR}/model_evaluation.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # ---- persist artifacts ----
    joblib.dump(preprocessor, f"{MODEL_DIR}/preprocessor.joblib")
    joblib.dump(qda, f"{MODEL_DIR}/qda_model.joblib")
    joblib.dump(gbm, f"{MODEL_DIR}/gbm_model.joblib")

    mu, sigma, _ = fit_mvn(claimant_df)
    joblib.dump({"mu": mu, "sigma": sigma}, f"{MODEL_DIR}/mvn_params.joblib")

    schema = {
        "numeric_features": ENGINEERED_NUMERIC,
        "raw_numeric_inputs": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "categorical_levels": {
            c: sorted(portfolio_df[c].unique().tolist()) for c in CATEGORICAL_FEATURES
        },
        "risk_tiers": classes,
    }
    with open(f"{MODEL_DIR}/feature_schema.json", "w") as f:
        json.dump(schema, f, indent=2)

    compute_explainability_artifacts(gbm, preprocessor, Xte, y_test, portfolio_df)

    print(json.dumps(results, indent=2, default=str))
    return results


if __name__ == "__main__":
    train_and_evaluate()
