from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import mlflow
import io

DATA_COLUMNS = [
    "patient_id",
    "age",
    "amh_ng_ml",
    "day",
    "avg_follicle_size_mm",
    "follicle_count",
    "estradiol_pg_ml",
    "progesterone_ng_ml",
    "age_group",
    "amh_group",
    "follicle_size_band",
    "follicle_size_12_19",
    "high_follicle_count",
    "high_e2",
    "high_p4",
    "late_cycle",
]

TARGET_COL = "trigger_recommended"
BEST_RUN_ID = "287c1645058940a097ec282b5eef181d"

app = FastAPI(title="IVF Trigger Decision API")


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    if TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])

    # ensure all expected columns exist
    for col in DATA_COLUMNS:
        if col not in df.columns:
            df[col] = None

    df = df[DATA_COLUMNS]

    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    df = df.fillna(df.mean(numeric_only=True))
    return df


def load_best_model():
    model_uri = f"runs:/{BEST_RUN_ID}/model"
    model = mlflow.sklearn.load_model(model_uri)
    return model


model = load_best_model()


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


@app.get("/")
def root():
    return {"message": "IVF Trigger Decision API is running"}


@app.post("/predict_row")
def predict_row(record: PatientRecord):
    df = pd.DataFrame([record.dict()])
    features = preprocess(df)
    proba = model.predict_proba(features)[:, 1][0]
    pred = int(proba >= 0.5)
    return {
        "pred_trigger_recommended": pred,
        "pred_trigger_probability": float(proba),
    }


@app.post("/predict_file")
async def predict_file(file: UploadFile = File(...)):
    content = await file.read()
    # supports CSV or Excel
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        df = pd.read_excel(io.BytesIO(content))

    features = preprocess(df.copy())
    proba = model.predict_proba(features)[:, 1]
    preds = (proba >= 0.5).astype(int)

    df["pred_trigger_recommended"] = preds
    df["pred_trigger_probability"] = proba

    return df.to_dict(orient="records")
