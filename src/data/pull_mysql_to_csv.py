import pandas as pd
import sqlalchemy as sa
from urllib.parse import quote_plus
import os

DB_HOST = "localhost"
DB_PORT = 3306
DB_NAME = "ivf_trigger_db"
DB_USER = "root"
DB_PASSWORD = "Chaithu@143"

TABLE_NAME = "ivf_trigger_data"  # <== change if your table name is different

RAW_PATH = os.path.join("data", "raw", "ivf_from_mysql.csv")


def main():
    url = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = sa.create_engine(url)

    query = f"SELECT * FROM {ivf_trigger_data};"
    df = pd.read_sql(query, engine)

    os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)
    df.to_csv(RAW_PATH, index=False)
    print(f"Wrote {len(df)} rows to {RAW_PATH}")


if __name__ == "__main__":
    main()
