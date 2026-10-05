"""
FastAPI Backend Application for Predictive Maintenance.
Exposes REST endpoints for model inference, health checks,
fleet telemetry inspection, and benchmark metadata.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib

from src.config import (
    DEFAULT_SUBSET,
    MODELS_DIR,
    RESULTS_DIR,
    RAW_DATA_DIR,
    FAILURE_THRESHOLD
)
from src.predict import get_predictor
from src.data_loader import load_raw_train_data
from src.rul import compute_run_to_failure_rul
from src.feature_engineering import generate_engine_features
from api.schemas import (
    TelemetryInput,
    EnginePredictionRequest,
    PredictionResponse,
    ModelInfoResponse,
    HealthResponse
)

app = FastAPI(
    title="AI-Based Predictive Maintenance API",
    description="Prognostics and Remaining Useful Life (RUL) estimation service for NASA C-MAPSS turbofan fleet.",
    version="1.0.0"
)

# Enable Cross-Origin Resource Sharing (CORS) for cross-platform browser accessibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global dataset cache for real-time fleet queries
_fleet_df: pd.DataFrame = None
_fleet_fe: pd.DataFrame = None


def get_fleet_data() -> pd.DataFrame:
    global _fleet_df, _fleet_fe
    if _fleet_fe is None:
        raw_df = load_raw_train_data()
        df_rul = compute_run_to_failure_rul(raw_df)
        _fleet_df = df_rul
        _fleet_fe, _ = generate_engine_features(df_rul)
    return _fleet_fe


@app.get("/", tags=["General"])
def root():
    """Root landing endpoint with system overview."""
    return {
        "service": "AI-Based Predictive Maintenance System",
        "description": "Remaining Useful Life (RUL) Prognostics API",
        "dataset": "NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)",
        "docs_url": "/docs",
        "status": "operational"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """System liveness and model readiness probe."""
    predictor = get_predictor()
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        system="FastAPI + PyTorch/Scikit-Learn/XGBoost",
        models_loaded=bool(predictor.best_model_name is not None)
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Prognostics"])
def get_model_info():
    """Retrieves experimental benchmarks and model governance records."""
    results_path = RESULTS_DIR / "model_comparison.json"
    clf_path = RESULTS_DIR / "classifier_metrics.json"
    
    if not results_path.exists():
        raise HTTPException(status_code=500, detail="Model metrics have not been generated.")
        
    with open(results_path, "r", encoding="utf-8") as f:
        comp_data = json.load(f)
        
    clf_metrics = {}
    if clf_path.exists():
        with open(clf_path, "r", encoding="utf-8") as f:
            clf_metrics = json.load(f)
            
    predictor = get_predictor()
    
    return ModelInfoResponse(
        champion_model=comp_data.get("best_model", "Linear Regression"),
        dataset=DEFAULT_SUBSET,
        selection_criterion="Lowest Validation RMSE on held-out 15% validation engines",
        feature_count=len(predictor.feature_cols),
        models=comp_data.get("models", []),
        classifier_metrics=clf_metrics
    )


@app.get("/engines", tags=["Fleet"])
def list_fleet_engines():
    """Returns directory of monitored engines, operational spans, and split assignments."""
    fleet = get_fleet_data()
    split_info = joblib.load(MODELS_DIR / "split_info.joblib")
    
    engine_list = []
    for eng_id in sorted(fleet["engine_id"].unique()):
        eng_data = fleet[fleet["engine_id"] == eng_id]
        max_cycle = int(eng_data["cycle"].max())
        
        split_name = "train"
        if eng_id in split_info["val_engines"]:
            split_name = "validation"
        elif eng_id in split_info["test_engines"]:
            split_name = "test"
            
        engine_list.append({
            "engine_id": int(eng_id),
            "max_cycle": max_cycle,
            "partition": split_name
        })
        
    return {
        "total_engines": len(engine_list),
        "engines": engine_list
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prognostics"])
def predict_rul(payload: TelemetryInput):
    """
    Predicts Remaining Useful Life (RUL) and failure risk for an engine.
    Supports either pre-indexed fleet engines or direct manual sensor telemetry dictionary.
    """
    predictor = get_predictor()
    model_choice = payload.model_name or "Best"
    
    fleet = get_fleet_data()
    eng_records = fleet[fleet["engine_id"] == payload.engine_id]
    
    if eng_records.empty:
        raise HTTPException(status_code=404, detail=f"Engine ID {payload.engine_id} not found in telemetry registry.")
        
    # Match specified cycle or pick latest available
    if payload.cycle is not None:
        row = eng_records[eng_records["cycle"] == payload.cycle]
        if row.empty:
            # Pick closest cycle
            idx = (eng_records["cycle"] - payload.cycle).abs().idxmin()
            row = eng_records.loc[[idx]]
    else:
        row = eng_records.loc[[eng_records["cycle"].idxmax()]]
        
    selected_cycle = int(row["cycle"].values[0])
    
    # Extract features
    features_dict = row.iloc[0].to_dict()
    
    # If custom sensor values were passed in payload, override them
    if payload.sensors:
        features_dict.update(payload.sensors)
        
    result = predictor.predict(features_dict=features_dict, model_name=model_choice)
    
    return PredictionResponse(
        engine_id=payload.engine_id,
        cycle=selected_cycle,
        predicted_rul=result["predicted_rul"],
        risk_category=result["risk_category"],
        failure_probability=result["failure_probability"],
        failure_soon=result["failure_soon"],
        model_used=result["model_used"],
        maintenance_recommendation=result["maintenance_recommendation"]
    )


@app.get("/engine/{engine_id}/trajectory", tags=["Fleet"])
def get_engine_trajectory(engine_id: int):
    """Returns chronological sensor readings and actual vs predicted RUL profile."""
    fleet = get_fleet_data()
    eng_df = fleet[fleet["engine_id"] == engine_id].sort_values("cycle")
    
    if eng_df.empty:
        raise HTTPException(status_code=404, detail=f"Engine {engine_id} not found.")
        
    predictor = get_predictor()
    scaled_feats = predictor.scaler.transform(eng_df[predictor.feature_cols].values)
    
    # Model predictions
    preds = predictor.xgb_model.predict(scaled_feats)
    
    response = {
        "engine_id": engine_id,
        "cycles": eng_df["cycle"].tolist(),
        "ground_truth_rul": eng_df["rul"].tolist() if "rul" in eng_df.columns else [],
        "predicted_rul": [round(float(p), 2) for p in preds],
        "sensor_2": eng_df["sensor_2"].tolist(),
        "sensor_3": eng_df["sensor_3"].tolist(),
        "sensor_4": eng_df["sensor_4"].tolist(),
        "sensor_7": eng_df["sensor_7"].tolist(),
        "sensor_11": eng_df["sensor_11"].tolist()
    }
    return response
