import os
import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus

PROJECT_ROOT = r"C:\AI_IVF_Trigger_day"

# processed CSV from previous step
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "ivf_trigger_preprocessed.csv")

# DB config – same as before
DB_HOST = "localhost"   # so Docker containers can reach Windows MySQL
DB_PORT = 3306
DB_NAME = "ivf_trigger_db"
DB_USER = "root"
DB_PASSWORD = "Chaithu@143"

TABLE_NAME = "ivf_trigger_data_clean"


def main():
    encoded_password = quote_plus(DB_PASSWORD)
    connection_url = (
        f"mysql+mysqlconnector://{DB_USER}:{encoded_password}"
        f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    engine = create_engine(connection_url)

    df = pd.read_csv(CSV_PATH)

    with engine.begin() as conn:
        df.to_sql(
            name=TABLE_NAME,
            con=conn,
            if_exists="replace",   # overwrite clean table each time
            index=False,
        )

    print(f"Loaded {len(df)} rows into {DB_NAME}.{TABLE_NAME}")


if __name__ == "__main__":
    main()
