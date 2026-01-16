import os
import pandas as pd
import numpy as np

PROJECT_ROOT = r"C:\AI_IVF_Trigger_day"

RAW_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "Trigger_day_prediction.csv")
PROCESSED_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "ivf_trigger_preprocessed.csv")


def load_raw():
    df = pd.read_csv(RAW_PATH)
    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(
        columns={
            "Patient_ID": "patient_id",
            "Age": "age",
            "AMH (ng/mL)": "amh_ng_ml",
            "Day": "day",
            "Avg_Follicle_Size_mm": "avg_follicle_size_mm",
            "Follicle_Count": "follicle_count",
            "Estradiol_pg_mL": "estradiol_pg_ml",
            "Progesterone_ng_mL": "progesterone_ng_ml",
            "Trigger_Recommended (0/1)": "trigger_recommended",
        }
    )
    df["patient_id"] = df["patient_id"].str.upper().str.strip()

    numeric_cols = [
        "age",
        "amh_ng_ml",
        "day",
        "avg_follicle_size_mm",
        "follicle_count",
        "estradiol_pg_ml",
        "progesterone_ng_ml",
        "trigger_recommended",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["trigger_recommended"] = df["trigger_recommended"].astype("Int64")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    # Drop rows where key clinical fields are missing
    required = [
        "age",
        "amh_ng_ml",
        "day",
        "avg_follicle_size_mm",
        "follicle_count",
        "estradiol_pg_ml",
        "progesterone_ng_ml",
        "trigger_recommended",
    ]
    before = len(df)
    df = df.dropna(subset=required)
    after = len(df)
    print(f"Dropped {before - after} rows due to critical NaNs; remaining {after}")
    return df


def drop_impossible_values(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)

    cond = (
        (df["age"].between(18, 50))
        & (df["amh_ng_ml"].between(0.1, 15))
        & (df["avg_follicle_size_mm"].between(8, 30))
        & (df["follicle_count"].between(1, 60))
        & (df["estradiol_pg_ml"].between(20, 6000))
        & (df["progesterone_ng_ml"].between(0.1, 5))
    )
    df = df[cond].copy()
    after = len(df)
    print(f"Dropped {before - after} rows due to impossible clinical ranges; remaining {after}")
    return df


def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # Age groups
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 29, 34, 37, 40, 100],
        labels=["<30", "30-34", "35-37", "38-40", ">40"],
        right=True,
    )

    # AMH groups (example thresholds)
    df["amh_group"] = pd.cut(
        df["amh_ng_ml"],
        bins=[0, 1.0, 3.5, 100],
        labels=["low", "normal", "high"],
        right=True,
    )

    # Follicle size bands
    df["follicle_size_band"] = pd.cut(
        df["avg_follicle_size_mm"],
        bins=[0, 12, 19, 100],
        labels=["<12", "12-19", ">=20"],
        right=True,
    )
    df["follicle_size_12_19"] = (
        (df["avg_follicle_size_mm"] >= 12) & (df["avg_follicle_size_mm"] <= 19)
    ).astype(int)

    # High follicle count flag
    df["high_follicle_count"] = (df["follicle_count"] >= 14).astype(int)

    # High estradiol / high progesterone flags
    df["high_e2"] = (df["estradiol_pg_ml"] >= 2500).astype(int)
    df["high_p4"] = (df["progesterone_ng_ml"] >= 1.0).astype(int)

    # Late-cycle flag
    df["late_cycle"] = (df["day"] >= 10).astype(int)

    return df


def main():
    df = load_raw()
    print(f"Loaded raw rows: {len(df)}")

    df = standardize_columns(df)
    df = handle_missing(df)
    df = drop_impossible_values(df)
    df = add_feature_engineering(df)

    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Saved preprocessed data to {PROCESSED_PATH} with {len(df)} rows and {df.shape[1]} columns")


if __name__ == "__main__":
    main()
