import pandas as pd
from sklearn.preprocessing import LabelEncoder
import mlflow
from feast import FeatureStore
import os
from datetime import datetime

# ===================================================================
# CONFIG
# ===================================================================
DATA_PATH = r"data/processed/ivf_trigger_preprocessed.csv"
TARGET_COL = "trigger_recommended"
BEST_RUN_ID = "287c1645058940a097ec282b5eef181d"  # Update with your best run ID

FEAST_REPO_PATH = os.path.join(os.path.dirname(__file__), "feast", "feature_repo")
fs = FeatureStore(repo_path=FEAST_REPO_PATH)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply same preprocessing as in mlflow_training.py without touching target
    """
    if TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])
    
    # Encode categorical columns
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    
    # Handle missing values
    df = df.fillna(df.mean(numeric_only=True))
    
    return df


def load_best_model():
    """
    Load the best model from MLflow using the run ID
    """
    model_uri = f"runs/{BEST_RUN_ID}/model"
    model = mlflow.sklearn.load_model(model_uri)
    return model


def predict_on_csv(input_path: str):
    """
    Predict on CSV file and save results with FEAST info
    """
    print("\n" + "="*70)
    print("📊 BATCH PREDICTION WITH FEAST")
    print("="*70)
    
    # Materialize FEAST features
    print("🔄 Materializing FEAST features...")
    try:
        fs.materialize_incremental(end_date=datetime.now())
        print("✅ FEAST features materialized!")
    except Exception as e:
        print(f"⚠️  FEAST: {e}")
        print("   Continuing with batch prediction...")
    
    print(f"\n📥 Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    print("🧹 Preprocessing data...")
    features = preprocess(df.copy())
    
    print("🤖 Loading best model...")
    model = load_best_model()
    
    print("🔮 Making predictions...")
    proba = model.predict_proba(features)[:, 1]
    preds = (proba > 0.5).astype(int)
    
    # Encode categorical columns
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        le = LabelEncoder
