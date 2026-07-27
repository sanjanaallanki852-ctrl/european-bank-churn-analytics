"""
Customer Segmentation & Churn Pattern Analytics - European Banking
Step 2: Machine Learning Layer

A) Unsupervised customer segmentation via KMeans clustering
   (complements the rule-based segmentation with a data-driven view)
B) Supervised churn prediction via Random Forest + XGBoost
   (quantifies which factors drive churn, with model evaluation)
"""

import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score, precision_score,
                              recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

DATA_PATH = "/home/claude/project/data/processed_bank_data.csv"
MODELS_DIR = "/home/claude/project/models"
RESULTS_PATH = "/home/claude/project/models/ml_results.json"

RANDOM_STATE = 42


def run_clustering(df: pd.DataFrame) -> pd.DataFrame:
    """K-Means clustering on standardized numeric features to derive
    data-driven customer segments (complements rule-based segmentation)."""
    features = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
                "EstimatedSalary", "IsActiveMember"]
    X = df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Evaluate k=3..6 for reference, but fix k=4: gains beyond k=4 are marginal
    # (silhouette 0.145->0.163) while 4 segments stay business-interpretable.
    from sklearn.metrics import silhouette_score
    sil_scores = {}
    for k in [3, 4, 5, 6]:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil_scores[k] = silhouette_score(X_scaled, labels, sample_size=3000, random_state=RANDOM_STATE)

    best_k = 4
    km_final = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    df["Cluster"] = km_final.fit_predict(X_scaled)

    joblib.dump(km_final, f"{MODELS_DIR}/kmeans_model.joblib")
    joblib.dump(scaler, f"{MODELS_DIR}/kmeans_scaler.joblib")

    cluster_profile = df.groupby("Cluster")[features + ["Exited"]].mean().round(2)
    cluster_sizes = df["Cluster"].value_counts().sort_index()

    # --- Assign business-meaningful labels based on inspected cluster profiles ---
    # Cluster 0: active, high-balance, low product count, low churn
    # Cluster 1: mixed activity, near-zero balance, higher product count, lowest churn
    # Cluster 2: 100% inactive, high-balance, low product count, high churn -> prime retention target
    # Cluster 3: senior-skewed, mostly active yet highest churn -> age-driven attrition
    label_lookup = {
        0: "Engaged High-Balance Loyalists",
        1: "Low-Balance Multi-Product",
        2: "Inactive High-Balance At-Risk",
        3: "Senior High-Risk",
    }
    # Map by sorting clusters on (IsActiveMember asc within high balance, Age) won't generalize
    # across reruns, so re-derive labels dynamically from profile characteristics each run:
    overall_churn = df["Exited"].mean()
    bal_med = df["Balance"].median()
    age_p75 = df["Age"].quantile(0.75)

    dynamic_labels = {}
    for c, row in cluster_profile.iterrows():
        if row["Age"] >= age_p75 and row["Exited"] > overall_churn:
            dynamic_labels[c] = "Senior High-Risk"
        elif row["IsActiveMember"] < 0.2 and row["Balance"] >= bal_med:
            dynamic_labels[c] = "Inactive High-Balance At-Risk"
        elif row["Balance"] < bal_med and row["NumOfProducts"] >= 1.8:
            dynamic_labels[c] = "Low-Balance Multi-Product"
        elif row["IsActiveMember"] >= 0.8 and row["Exited"] <= overall_churn:
            dynamic_labels[c] = "Engaged High-Balance Loyalists"
        else:
            dynamic_labels[c] = f"Segment {c}"

    seen = dynamic_labels
    df["SegmentLabel"] = df["Cluster"].map(seen)

    print("=== K-MEANS CLUSTERING ===")
    print("Silhouette scores by k:", {k: round(v, 3) for k, v in sil_scores.items()})
    print(f"Selected k={best_k} (fixed for business interpretability)")
    print("\nCluster sizes:\n", cluster_sizes)
    print("\nCluster labels:\n", seen)
    print("\nCluster profile (means):\n", cluster_profile)

    return df, sil_scores, best_k, cluster_profile, cluster_sizes, seen


def run_churn_prediction(df: pd.DataFrame):
    """Supervised churn prediction with Random Forest and XGBoost,
    used to rank feature importance and validate segmentation insights."""
    feature_cols = ["CreditScore", "Geography", "Gender", "Age", "Tenure",
                     "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
                     "EstimatedSalary"]
    X = pd.get_dummies(df[feature_cols], columns=["Geography", "Gender"], drop_first=True)
    y = df["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    results = {}

    # --- Random Forest ---
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=10,
        class_weight="balanced", random_state=RANDOM_STATE
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]

    results["random_forest"] = {
        "accuracy": accuracy_score(y_test, rf_pred),
        "precision": precision_score(y_test, rf_pred),
        "recall": recall_score(y_test, rf_pred),
        "f1": f1_score(y_test, rf_pred),
        "roc_auc": roc_auc_score(y_test, rf_proba),
    }
    rf_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

    # --- XGBoost ---
    xgb = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        random_state=RANDOM_STATE, eval_metric="logloss"
    )
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    xgb_proba = xgb.predict_proba(X_test)[:, 1]

    results["xgboost"] = {
        "accuracy": accuracy_score(y_test, xgb_pred),
        "precision": precision_score(y_test, xgb_pred),
        "recall": recall_score(y_test, xgb_pred),
        "f1": f1_score(y_test, xgb_pred),
        "roc_auc": roc_auc_score(y_test, xgb_proba),
    }
    xgb_importance = pd.Series(xgb.feature_importances_, index=X.columns).sort_values(ascending=False)

    joblib.dump(rf, f"{MODELS_DIR}/random_forest_model.joblib")
    joblib.dump(xgb, f"{MODELS_DIR}/xgboost_model.joblib")
    joblib.dump(list(X.columns), f"{MODELS_DIR}/feature_columns.joblib")

    print("\n=== CHURN PREDICTION MODELS ===")
    for name, m in results.items():
        print(f"{name}: " + ", ".join(f"{k}={v:.3f}" for k, v in m.items()))

    print("\nRandom Forest feature importance:\n", rf_importance.round(3))
    print("\nXGBoost feature importance:\n", xgb_importance.round(3))

    return results, rf_importance, xgb_importance, confusion_matrix(y_test, xgb_pred)


def main():
    df = pd.read_csv(DATA_PATH)
    df, sil_scores, best_k, cluster_profile, cluster_sizes, cluster_labels = run_clustering(df)
    df.to_csv(DATA_PATH, index=False)  # persist Cluster + SegmentLabel columns

    results, rf_imp, xgb_imp, cm = run_churn_prediction(df)

    output = {
        "clustering": {
            "silhouette_scores": {str(k): round(v, 4) for k, v in sil_scores.items()},
            "best_k": best_k,
            "cluster_sizes": cluster_sizes.to_dict(),
            "cluster_labels": cluster_labels,
            "cluster_profile": cluster_profile.to_dict(orient="index"),
        },
        "churn_models": results,
        "rf_feature_importance": rf_imp.round(4).to_dict(),
        "xgb_feature_importance": xgb_imp.round(4).to_dict(),
        "xgb_confusion_matrix": cm.tolist(),
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved ML results -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
