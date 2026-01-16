import pandas as pd
from sklearn.preprocessing import LabelEncoder
import mlflow

# ---------------- CONFIG ----------------
DATA_PATH = r"data/processed/ivf_trigger_preprocessed.csv"  # or any new CSV with same columns
TARGET_COL = "trigger_recommended"

# Best model run ID from MLflow
BEST_RUN_ID = "287c1645058940a097ec282b5eef181d"


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Apply same preprocessing as in mlflow_training.py (without touching target)."""
    if TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])

    # Encode categorical columns
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    df = df.fillna(df.mean(numeric_only=True))
    return df


def load_best_model():
    """Load the GradientBoosting model from MLflow using the run ID."""
    model_uri = f"runs:/{BEST_RUN_ID}/model"
    model = mlflow.sklearn.load_model(model_uri)
    return model


def predict_on_csv(input_path: str):
    print(f"📁 Loading data from: {input_path}")
    df = pd.read_csv(input_path)

    features = preprocess(df.copy())
    model = load_best_model()

    print("🤖 Making predictions...")
    proba = model.predict_proba(features)[:, 1]
    preds = (proba >= 0.5).astype(int)

    # Attach predictions to original data
    df["pred_trigger_recommended"] = preds
    df["pred_trigger_probability"] = proba

    output_path = "ivf_trigger_predictions.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ Saved predictions to: {output_path}")


if __name__ == "__main__":
    predict_on_csv(DATA_PATH)
