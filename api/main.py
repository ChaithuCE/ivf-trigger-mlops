from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import mlflow
import io
from prometheus_fastapi_instrumentator import Instrumentator
from feast import FeatureStore
import os
from datetime import datetime

# ===================================================================
# CONFIG
# ===================================================================
DATA_COLUMNS = [
    "patient_id", "age", "amh_ng_ml", "day", "avg_follicle_size_mm",
    "follicle_count", "estradiol_pg_ml", "progesterone_ng_ml",
    "age_group", "amh_group", "follicle_size_band", "follicle_size_12_19",
    "high_follicle_count", "high_e2", "high_p4", "late_cycle"
]

TARGET_COL = "trigger_recommended"
MODEL_NAME = "ivf_trigger_model"
MODEL_VERSION = 5
BEST_RUN_ID = "8bcf729641d0463cad34bb45a7443a6b"
BEST_ARTIFACT_NAME = "GradientBoosting"  # The algorithm name used in training


# ===================================================================
# INITIALIZE FEAST
# ===================================================================
FEAST_REPO_PATH = os.path.join(os.path.dirname(__file__), "..", "feast", "feature_repo")
fs = FeatureStore(repo_path=FEAST_REPO_PATH)

# ===================================================================
# INITIALIZE FASTAPI
# ===================================================================
app = FastAPI(title="IVF Trigger Decision API")
Instrumentator().instrument(app).expose(app)

# Global model variable
model = None

def load_best_model():
    """Load model from MLflow models registry"""
    global model
    if model is not None:
        return model
    
    try:
        print("Loading model from MLflow models registry...")
        # Load directly from model registry using stage
        model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/latest")
        print(f"✅ Model loaded successfully: {MODEL_NAME}")
        return model
    except Exception as e:
        print(f"Error: {e}")
        # Try alternate loading
        try:
            print("Trying alternate model loading method...")
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            model_version = client.get_latest_versions(MODEL_NAME, stages=["None"])[0]
            model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{model_version.version}")
            print(f"✅ Model loaded: {MODEL_NAME} v{model_version.version}")
            return model
        except Exception as e2:
            print(f"❌ Failed to load model: {e2}")
            raise


# ===================================================================
# PREPROCESSING
# ===================================================================
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess data: handle missing values, encode categoricals"""
    if TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])
    
    # Ensure all expected columns exist
    for col in DATA_COLUMNS:
        if col not in df.columns:
            df[col] = None
    
    df = df[DATA_COLUMNS]
    
    # Encode categorical columns
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    
    # Handle missing values
    df = df.fillna(df.mean(numeric_only=True))
    
    return df


# ===================================================================
# PYDANTIC MODEL
# ===================================================================
class PatientRecord(BaseModel):
    patient_id: str
    age: float
    amh_ng_ml: float
    day: int
    avg_follicle_size_mm: float
    follicle_count: int
    estradiol_pg_ml: float
    progesterone_ng_ml: float
    age_group: str
    amh_group: str
    follicle_size_band: str
    follicle_size_12_19: int
    high_follicle_count: int
    high_e2: int
    high_p4: int
    late_cycle: int


# ===================================================================
# ENDPOINTS
# ===================================================================
@app.get("/")
def root():
    return {
        "message": "IVF Trigger Decision API is running",
        "version": "1.0",
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "feast_integrated": True
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "feast_path": FEAST_REPO_PATH,
        "feast_initialized": True
    }


@app.post("/predict/row")
def predict_row(record: PatientRecord):
    """Predict for single patient"""
    try:
        # Load model if not loaded
        model_to_use = load_best_model()
        
        # Create dataframe from record
        df = pd.DataFrame([record.dict()])
        
        # Preprocess
        features = preprocess(df)
        
        # Make prediction
        proba = model_to_use.predict_proba(features)[:, 1]
        pred = int(proba[0] > 0.5)
        
        return {
            "patient_id": record.patient_id,
            "pred_trigger_recommended": pred,
            "pred_trigger_probability": float(proba[0]),
            "model_version": MODEL_VERSION,
            "feast_enabled": True
        }
    
    except Exception as e:
        return {
            "error": str(e),
            "patient_id": record.patient_id
        }


@app.post("/predict/file")
async def predict_file(file: UploadFile = File(...)):
    """Predict for multiple patients from CSV/Excel"""
    try:
        # Load model if not loaded
        model_to_use = load_best_model()
        
        # Read file
        content = await file.read()
        
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
        else:
            df = pd.read_excel(io.BytesIO(content))
        
        # Preprocess
        features = preprocess(df.copy())
        
        # Make predictions
        proba = model_to_use.predict_proba(features)[:, 1]
        preds = (proba > 0.5).astype(int)
        
        # Add predictions to original dataframe
        df["pred_trigger_recommended"] = preds
        df["pred_trigger_probability"] = proba
        df["model_version"] = MODEL_VERSION
        
        return {
            "total_records": len(df),
            "predictions": df.to_dict(orient="records"),
            "feast_enabled": True,
            "model_version": MODEL_VERSION
        }
    
    except Exception as e:
        return {
            "error": str(e),
            "file": file.filename
        }
