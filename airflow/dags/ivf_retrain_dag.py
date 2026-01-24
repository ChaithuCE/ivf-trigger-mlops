from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

# Path to your project inside the Airflow containers
PROJECT_ROOT = "/opt/airflow/project"
PYTHON_EXE = "python"

default_args = {
    "owner": "chaithu",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="ivf_trigger_retraining",
    default_args=default_args,
    description="IVF trigger model retraining with MLflow model registry",
    start_date=datetime(2026, 1, 17),
    schedule_interval="@daily",   # adjust to @weekly if needed
    catchup=False,
    tags=["ivf", "mlops", "mlflow"],
) as dag:

    # 1) Pull latest IVF data from MySQL into data/raw/ivf_from_mysql.csv
    pull_mysql = BashOperator(
        task_id="pull_mysql_to_csv",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            f"{PYTHON_EXE} src/data/pull_mysql_to_csv.py"
        ),
    )

    # 2) Run Great Expectations + preprocessing
    #    ge_validate_ivf_preprocessed.py should:
    #      - read data/raw/ivf_from_mysql.csv
    #      - validate with Great Expectations
    #      - write data/processed/ivf_trigger_preprocessed.csv
    ge_validate = BashOperator(
        task_id="ge_validate_and_preprocess",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            f"{PYTHON_EXE} ge_validate_ivf_preprocessed.py"
        ),
    )

    # 3) Train models and log to MLflow
    train_mlflow = BashOperator(
        task_id="train_models_mlflow",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            f"MLFLOW_TRACKING_URI=sqlite:///mlflow.db "
            f"{PYTHON_EXE} mlflow_training.py"
        ),
    )

    # 4) Register best model in MLflow Model Registry
    register_best = BashOperator(
        task_id="register_best_model",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            f"MLFLOW_TRACKING_URI=sqlite:///mlflow.db "
            f"{PYTHON_EXE} register_best_model.py"
        ),
    )

    pull_mysql >> ge_validate >> train_mlflow >> register_best
