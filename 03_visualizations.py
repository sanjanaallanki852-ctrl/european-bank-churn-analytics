"""
Customer Segmentation & Churn Pattern Analytics - European Banking
Step 3: Visualization generation for the research paper / executive summary
"""

import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import joblib
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

DATA_PATH = "/home/claude/project/data/processed_bank_data.csv"
FIG_DIR = "/home/claude/project/figures"
MODELS_DIR = "/home/claude/project/models"

# Brand-consistent palette
NAVY = "#1B2A4A"
BLUE = "#2E5C8A"
TEAL = "#3D8B8B"
GOLD = "#C9A646"
RED = "#B0413E"
GREY = "#8A93A3"
PALETTE = [NAVY, BLUE, TEAL, GOLD, RED, GREY]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#444444",
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "axes.grid": True,
    "grid.color": "#E5E7EB",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}.png")


def main():
    df = pd.read_csv(DATA_PATH)
    with open(f"{MODELS_DIR}/ml_results.json") as f:
        ml = json.load(f)

    overall_churn = df["Exited"].mean()

    # 1. Overall churn donut
    fig, ax = plt.subplots(figsize=(5, 5))
    sizes = [overall_churn, 1 - overall_churn]
    ax.pie(sizes, labels=["Churned", "Retained"], colors=[RED, BLUE],
           autopct="%1.1f%%", startangle=90, wedgeprops={"width": 0.42, "edgecolor": "white"},
           textprops={"fontsize": 12})
    ax.set_title("Overall Customer Churn Rate", fontsize=14, fontweight="bold", color=NAVY)
    save(fig, "01_overall_churn")

    # 2. Churn rate by Geography
    fig, ax = plt.subplots(figsize=(6, 4.5))
    geo = df.groupby("Geography")["Exited"].mean().sort_values(ascending=False) * 100
    bars = ax.bar(geo.index, geo.values, color=[RED, GOLD, TEAL])
    ax.axhline(overall_churn * 100, color=GREY, linestyle="--", linewidth=1, label=f"Overall avg: {overall_churn*100:.1f}%")
    for b, v in zip(bars, geo.values):
        ax.text(b.get_x() + b.get_width()/2, v + 0.5, f"{v:.1f}%", ha="center", fontweight="bold", fontsize=11)
    ax.set_ylabel("Churn Rate (%)")
    ax.set_title("Churn Rate by Geography", fontsize=14, fontweight="bold", color=NAVY)
    ax.legend(frameon=False)
    ax.set_ylim(0, max(geo.values) + 6)
    save(fig, "02_churn_by_geography")

    # 3. Churn rate by Age Group
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    order = ["<30", "30-45", "46-60", "60+"]
    age = df.groupby("AgeGroup", observed=True)["Exited"].mean().reindex(order) * 100
    bars = ax.bar(age.index, age.values, color=BLUE)
    for b, v in zip(bars, age.values):
        ax.text(b.get_x() + b.get_width()/2, v + 0.8, f"{v:.1f}%", ha="center", fontweight="bold", fontsize=11)
    ax.axhline(overall_churn * 100, color=GREY, linestyle="--", linewidth=1, label=f"Overall avg: {overall_churn*100:.1f}%")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_title("Churn Rate by Age Group", fontsize=14, fontweight="bold", color=NAVY)
    ax.legend(frameon=False)
    save(fig, "03_churn_by_age")

    # 4. Geography x Age interaction heatmap
    fig, ax = plt.subplots(figsize=(7, 4.5))
    pivot = df.groupby(["Geography", "AgeGroup"], observed=True)["Exited"].mean().unstack().reindex(columns=order) * 100
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn_r", ax=ax, cbar_kws={"label": "Churn Rate (%)"},
                linewidths=1, linecolor="white", vmin=0, vmax=70)
    ax.set_title("Churn Rate (%): Geography x Age Group", fontsize=14, fontweight="bold", color=NAVY)
    ax.set_ylabel("")
    ax.set_xlabel("")
    save(fig, "04_geo_age_heatmap")

    # 5. Churn by NumOfProducts
    fig, ax = plt.subplots(figsize=(6, 4.5))
    prod = df.groupby("NumOfProducts")["Exited"].mean() * 100
    bars = ax.bar(prod.index.astype(str), prod.values, color=[BLUE, TEAL, RED, NAVY])
    for b, v in zip(bars, prod.values):
        ax.text(b.get_x() + b.get_width()/2, v + 1, f"{v:.1f}%", ha="center", fontweight="bold", fontsize=11)
    ax.set_xlabel("Number of Products Held")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_title("Churn Rate by Number of Bank Products", fontsize=14, fontweight="bold", color=NAVY)
    save(fig, "05_churn_by_products")

    # 6. Active vs Inactive churn
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    act = df.groupby("ActiveLabel")["Exited"].mean().sort_values(ascending=False) * 100
    bars = ax.bar(act.index, act.values, color=[RED, BLUE])
    for b, v in zip(bars, act.values):
        ax.text(b.get_x() + b.get_width()/2, v + 0.8, f"{v:.1f}%", ha="center", fontweight="bold", fontsize=11)
    ax.set_ylabel("Churn Rate (%)")
    ax.set_title("Engagement Drop Indicator:\nActive vs Inactive Member Churn", fontsize=13, fontweight="bold", color=NAVY)
    save(fig, "06_churn_by_activity")

    # 7. Balance segment churn
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bseg = df.groupby("BalanceSegment", observed=True)["Exited"].mean().reindex(
        ["Zero-balance", "Low-balance", "High-balance"]) * 100
    bars = ax.bar(bseg.index, bseg.values, color=[GREY, TEAL, GOLD])
    for b, v in zip(bars, bseg.values):
        ax.text(b.get_x() + b.get_width()/2, v + 0.6, f"{v:.1f}%", ha="center", fontweight="bold", fontsize=11)
    ax.set_ylabel("Churn Rate (%)")
    ax.set_title("Churn Rate by Account Balance Segment", fontsize=14, fontweight="bold", color=NAVY)
    save(fig, "07_churn_by_balance")

    # 8. Gender x Geography churn
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    gg = df.groupby(["Geography", "Gender"])["Exited"].mean().unstack() * 100
    gg.plot(kind="bar", ax=ax, color=[GOLD, BLUE], width=0.7)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", fontsize=9, fontweight="bold")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_xlabel("")
    ax.set_title("Churn Rate by Geography and Gender", fontsize=14, fontweight="bold", color=NAVY)
    ax.legend(title="", frameon=False)
    plt.xticks(rotation=0)
    save(fig, "08_gender_geo_churn")

    # 9. KMeans cluster scatter (Age vs Balance, colored by segment)
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, seg in enumerate(df["SegmentLabel"].unique()):
        sub = df[df["SegmentLabel"] == seg]
        ax.scatter(sub["Age"], sub["Balance"], s=10, alpha=0.45, color=PALETTE[i % len(PALETTE)], label=seg)
    ax.set_xlabel("Age")
    ax.set_ylabel("Balance (EUR)")
    ax.set_title("Data-Driven Customer Segments (K-Means)", fontsize=14, fontweight="bold", color=NAVY)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    save(fig, "09_kmeans_segments")

    # 10. Cluster churn rate comparison
    fig, ax = plt.subplots(figsize=(7, 4.5))
    seg_churn = df.groupby("SegmentLabel")["Exited"].mean().sort_values(ascending=False) * 100
    seg_size = df["SegmentLabel"].value_counts().reindex(seg_churn.index)
    bars = ax.barh(seg_churn.index, seg_churn.values, color=[RED if v > overall_churn*100 else BLUE for v in seg_churn.values])
    for b, v, n in zip(bars, seg_churn.values, seg_size.values):
        ax.text(v + 0.8, b.get_y() + b.get_height()/2, f"{v:.1f}%  (n={n:,})", va="center", fontsize=9, fontweight="bold")
    ax.axvline(overall_churn * 100, color=GREY, linestyle="--", linewidth=1)
    ax.set_xlabel("Churn Rate (%)")
    ax.set_title("Churn Rate by Data-Driven Customer Segment", fontsize=14, fontweight="bold", color=NAVY)
    ax.set_xlim(0, max(seg_churn.values) + 12)
    save(fig, "10_cluster_churn")

    # 11. Feature importance (XGBoost)
    fig, ax = plt.subplots(figsize=(7, 5))
    fi = pd.Series(ml["xgb_feature_importance"]).sort_values(ascending=True)
    ax.barh(fi.index, fi.values, color=TEAL)
    ax.set_xlabel("Relative Importance")
    ax.set_title("Churn Driver Importance (XGBoost Model)", fontsize=14, fontweight="bold", color=NAVY)
    save(fig, "11_feature_importance")

    # 12. ROC curve comparison
    feature_cols = ["CreditScore", "Geography", "Gender", "Age", "Tenure",
                     "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]
    X = pd.get_dummies(df[feature_cols], columns=["Geography", "Gender"], drop_first=True)
    y = df["Exited"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rf = joblib.load(f"{MODELS_DIR}/random_forest_model.joblib")
    xgb = joblib.load(f"{MODELS_DIR}/xgboost_model.joblib")

    fig, ax = plt.subplots(figsize=(6, 5.5))
    for model, name, color in [(rf, "Random Forest", BLUE), (xgb, "XGBoost", RED)]:
        proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color=GREY, linestyle="--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Churn Prediction Model ROC Curves", fontsize=14, fontweight="bold", color=NAVY)
    ax.legend(frameon=False, loc="lower right")
    save(fig, "12_roc_curve")

    # 13. Credit score band churn
    fig, ax = plt.subplots(figsize=(6, 4.5))
    cb = df.groupby("CreditBand", observed=True)["Exited"].mean().reindex(["Low", "Medium", "High"]) * 100
    bars = ax.bar(cb.index, cb.values, color=[RED, GOLD, TEAL])
    for b, v in zip(bars, cb.values):
        ax.text(b.get_x() + b.get_width()/2, v + 0.4, f"{v:.1f}%", ha="center", fontweight="bold", fontsize=11)
    ax.set_ylabel("Churn Rate (%)")
    ax.set_title("Churn Rate by Credit Score Band", fontsize=14, fontweight="bold", color=NAVY)
    save(fig, "13_churn_by_credit_band")

    print("\nAll figures saved to", FIG_DIR)


if __name__ == "__main__":
    main()
