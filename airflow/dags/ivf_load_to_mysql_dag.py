from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

import os
import sys

# === make sure Python can see your project ===
PROJECT_ROOT = r"C:\AI_IVF_Trigger_day"
SRC_DATA_PATH = os.path.join(PROJECT_ROOT, "src", "data")
sys.path.append(SRC_DATA_PATH)

from load_ivf_csv_to_mysql import main as load_ivf_main  # we will add main() below


with DAG(
    dag_id="ivf_load_to_mysql_dag",
    start_date=datetime(2024, 1, 1),
    schedule=None,   # run only when you click trigger
    catchup=False,
    tags=["ivf", "ingestion"],
) as dag:

    load_ivf_task = PythonOperator(
        task_id="load_ivf_csv_to_mysql",
        python_callable=load_ivf_main,
    )
