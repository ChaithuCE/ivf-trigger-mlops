import pandas as pd

def convert_csv_to_parquet():
    # Read original CSV
    df = pd.read_csv("data/trigger_day_prediction.csv")

    # Simple missing handling (optional)
    df["AMH (ng/mL)"] = df["AMH (ng/mL)"].fillna(df["AMH (ng/mL)"].median())

    # Add a synthetic event_timestamp column (required by Feast)
    from datetime import datetime
    df["event_timestamp"] = datetime.utcnow()

    # Save as Parquet for Feast offline store
    df.to_parquet("data/trigger_day_prediction.parquet", index=False)
    print("Saved Parquet file to data/trigger_day_prediction.parquet")

if __name__ == "__main__":
    convert_csv_to_parquet()
