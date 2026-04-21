from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
from src.predict import score_record, load_metadata

app = FastAPI(
    title="Production MLOps Pipeline API",
    version="1.0.0",
    description="Portfolio-grade churn scoring API with monitoring-friendly outputs.",
)

class TelcoRequest(BaseModel):
    gender: Literal["Female", "Male"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.get("/health")
def health():
    meta = load_metadata()
    return {"status": "ok", "model_version": meta["version"], "model_name": meta["model_name"]}

@app.get("/metadata")
def metadata():
    return load_metadata()

@app.post("/predict")
def predict(payload: TelcoRequest):
    scored = score_record(payload.model_dump())
    return {
        "score": round(float(scored["score"]), 4),
        "predicted_label": int(scored["predicted_label"]),
        "priority_band": str(scored["priority_band"]),
        "recommended_action": (
            "Route to retention specialist within 24h"
            if float(scored["score"]) >= 0.75
            else "Queue for proactive outreach"
            if float(scored["score"]) >= 0.5
            else "Standard monitoring"
        ),
    }
