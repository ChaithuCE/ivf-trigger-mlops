import os
import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus

DB_HOST = "localhost"
DB_PORT = 3306
DB_NAME = "ivf_trigger_db"
DB_USER = "root"
DB_PASSWORD = "Chaithu@143"

PROJECT_ROOT = r"C:\AI_IVF_Trigger_day"
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "Trigger_day_prediction.csv")
TABLE_NAME = "ivf_trigger_data"


def main():
    encoded_password = quote_plus(DB_PASSWORD)
    connection_url = (
        f"mysql+mysqlconnector://{DB_USER}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    engine = create_engine(connection_url)

    df = pd.read_csv(CSV_PATH)

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

    with engine.begin() as conn:
        df.to_sql(
            name=TABLE_NAME,
            con=conn,
            if_exists="append",
            index=False,
        )

    print(f"Loaded {len(df)} rows into {DB_NAME}.{TABLE_NAME}")


if __name__ == "__main__":
    main()
