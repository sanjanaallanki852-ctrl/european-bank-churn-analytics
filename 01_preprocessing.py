"""
Customer Segmentation & Churn Pattern Analytics - European Banking
Step 1: Data Ingestion, Validation & Preprocessing

Per project methodology:
- Load dataset
- Validate engagement and product fields
- Ensure binary variables consistency
- Confirm churn labeling accuracy
- Remove non-analytical fields (surname)
- Convert categorical variables for grouping
- Create derived segmentation fields
"""

import pandas as pd
import numpy as np

RAW_PATH = "/home/claude/project/data/European_Bank.csv"
OUT_PATH = "/home/claude/project/data/processed_bank_data.csv"


def load_and_validate(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    assert df["CustomerId"].is_unique, "CustomerId must be unique"
    assert df["Exited"].isin([0, 1]).all(), "Exited must be binary"
    assert df["HasCrCard"].isin([0, 1]).all(), "HasCrCard must be binary"
    assert df["IsActiveMember"].isin([0, 1]).all(), "IsActiveMember must be binary"
    assert df["NumOfProducts"].between(1, 4).all(), "NumOfProducts out of expected range"
    assert df["Geography"].isin(["France", "Spain", "Germany"]).all(), "Unexpected geography value"
    assert df["Gender"].isin(["Male", "Female"]).all(), "Unexpected gender value"
    assert df.isnull().sum().sum() == 0, "Unexpected missing values"

    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns. Validation passed.")
    return df


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=["Surname"])

    # --- Segmentation dimensions (per Analytical Methodology) ---
    df["AgeGroup"] = pd.cut(
        df["Age"], bins=[0, 30, 45, 60, 120],
        labels=["<30", "30-45", "46-60", "60+"], right=True
    )

    df["CreditBand"] = pd.cut(
        df["CreditScore"], bins=[0, 580, 700, 900],
        labels=["Low", "Medium", "High"], right=True
    )

    df["TenureGroup"] = pd.cut(
        df["Tenure"], bins=[-1, 2, 6, 10],
        labels=["New", "Mid-term", "Long-term"], right=True
    )

    df["BalanceSegment"] = pd.cut(
        df["Balance"], bins=[-1, 0, 100000, 1_000_000],
        labels=["Zero-balance", "Low-balance", "High-balance"], right=True
    )

    # --- High-value customer flag (top quartile balance) ---
    bal_p75 = df["Balance"].quantile(0.75)
    df["IsHighValue"] = (df["Balance"] >= bal_p75).astype(int)

    # --- Engagement risk flag: inactive + low product count ---
    df["EngagementRisk"] = ((df["IsActiveMember"] == 0) & (df["NumOfProducts"] == 1)).astype(int)

    # Readable labels
    df["ChurnLabel"] = df["Exited"].map({0: "Retained", 1: "Churned"})
    df["ActiveLabel"] = df["IsActiveMember"].map({0: "Inactive", 1: "Active"})
    df["CrCardLabel"] = df["HasCrCard"].map({0: "No Card", 1: "Has Card"})

    return df


def main():
    df = load_and_validate(RAW_PATH)
    df = clean_and_engineer(df)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved processed dataset -> {OUT_PATH}")
    print(f"Final shape: {df.shape}")
    print("\nColumns:", list(df.columns))


if __name__ == "__main__":
    main()
