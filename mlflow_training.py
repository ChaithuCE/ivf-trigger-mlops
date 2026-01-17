import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder

import mlflow
import mlflow.sklearn

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
DATA_PATH = r"data/processed/ivf_trigger_preprocessed.csv"
TARGET_COL = "trigger_recommended"


# -------------------------------------------------------------------
# DATA LOADING + PREPROCESSING
# -------------------------------------------------------------------
def load_data():
    df = pd.read_csv(DATA_PATH)

    # Target
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    # Encode categorical columns
    cat_cols = X.select_dtypes(include="object").columns
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Handle missing values
    X = X.fillna(X.mean(numeric_only=True))

    return X, y


# -------------------------------------------------------------------
# TRAIN + LOG TO MLFLOW
# -------------------------------------------------------------------
def train_and_log():
    # Create / use experiment
    mlflow.set_experiment("IVF_Trigger_Prediction")

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            random_state=42,
        ),
    }

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            # Log parameters
            mlflow.log_params(model.get_params())

            # Train
            model.fit(X_train, y_train)

            # Predictions
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            # Metrics
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_proba),
            }

            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            # Log model
            mlflow.sklearn.log_model(model, artifact_path="model")

                # ---------------------------------------------------------
    # Choose best model by ROC AUC
    # ---------------------------------------------------------
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("IVF_Trigger_Prediction")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.roc_auc DESC"],
        max_results=1,
    )

    best_run = runs[0]
    best_run_id = best_run.info.run_id
    best_roc_auc = best_run.data.metrics["roc_auc"]

    print("Best run_id:", best_run_id)
    print("Best roc_auc:", best_roc_auc)



# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
if __name__ == "__main__":
    train_and_log()
