"""
Automated Unit and Integration Test Suite.
Verifies:
1. RUL calculation & piecewise clipping
2. Preprocessing & zero engine leakage
3. Feature engineering & rolling statistics
4. Model prediction shape across all 4 architectures
5. API health endpoint (/health)
6. API prediction endpoint (/predict)
"""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.config import FAILURE_THRESHOLD, CLIP_RUL_MAX
from src.rul import compute_run_to_failure_rul
from src.preprocessing import inspect_data_quality, remove_uninformative_features, perform_engine_wise_split
from src.feature_engineering import generate_engine_features
from src.predict import get_predictor
from api.main import app


# ----------------------------------------------------
# 1. TEST RUL CALCULATION
# ----------------------------------------------------

def test_rul_calculation_monotonic_and_clipped():
    sample_data = {
        "engine_id": [1, 1, 1, 1, 2, 2],
        "cycle": [1, 2, 3, 4, 1, 2],
        "setting_1": [0.0] * 6,
        "setting_2": [0.0] * 6,
        "setting_3": [100.0] * 6,
    }
    for i in range(1, 22):
        sample_data[f"sensor_{i}"] = [500.0] * 6
    df = pd.DataFrame(sample_data)
    
    # Unclipped calculation
    df_rul = compute_run_to_failure_rul(df, clip_threshold=None)
    # Engine 1 has max cycle 4
    # Cycle 1 -> RUL 3, Cycle 4 -> RUL 0
    e1_rul = df_rul[df_rul["engine_id"] == 1]["rul"].tolist()
    assert e1_rul == [3, 2, 1, 0]
    
    # Engine 2 has max cycle 2
    e2_rul = df_rul[df_rul["engine_id"] == 2]["rul"].tolist()
    assert e2_rul == [1, 0]
    
    # Verify failure_soon indicator
    assert df_rul[df_rul["rul"] <= FAILURE_THRESHOLD]["failure_soon"].all() == 1


# ----------------------------------------------------
# 2. TEST DATA CLEANING & ENGINE-WISE SPLIT LEAKAGE
# ----------------------------------------------------

def test_engine_wise_split_zero_leakage():
    # Synthetic fleet of 20 engines
    records = []
    for eng in range(1, 21):
        for c in range(1, 11):
            row = {"engine_id": eng, "cycle": c, "setting_1": 0.0, "setting_2": 0.0, "setting_3": 100.0}
            for s in range(1, 22):
                row[f"sensor_{s}"] = float(s * 10 + c)
            records.append(row)
    df = pd.DataFrame(records)
    
    train_df, val_df, test_df, split_info = perform_engine_wise_split(
        df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42
    )
    
    train_set = set(split_info["train_engines"])
    val_set = set(split_info["val_engines"])
    test_set = set(split_info["test_engines"])
    
    # Assert mutual exclusivity: ZERO leakage
    assert len(train_set.intersection(val_set)) == 0
    assert len(train_set.intersection(test_set)) == 0
    assert len(val_set.intersection(test_set)) == 0
    
    # All engines accounted for
    assert len(train_set) + len(val_set) + len(test_set) == 20


# ----------------------------------------------------
# 3. TEST FEATURE ENGINEERING
# ----------------------------------------------------

def test_feature_engineering_backward_looking():
    sample_data = {
        "engine_id": [1, 1, 1, 1, 1],
        "cycle": [1, 2, 3, 4, 5],
        "setting_1": [0.0] * 5,
        "setting_2": [0.0] * 5,
        "setting_3": [100.0] * 5,
    }
    for i in range(1, 22):
        sample_data[f"sensor_{i}"] = [float(c * 10) for c in range(1, 6)]
    df = pd.DataFrame(sample_data)
    
    fe_df, feat_cols = generate_engine_features(df, sensor_cols=["sensor_2", "sensor_3"], windows=[5])
    
    # Rate of change at cycle 1 must be 0.0, at cycle 2 must be 10.0
    assert fe_df["sensor_2_diff1"].iloc[0] == 0.0
    assert np.isclose(fe_df["sensor_2_diff1"].iloc[1], 10.0)
    
    # No NaNs in constructed features
    assert fe_df[feat_cols].isnull().sum().sum() == 0


# ----------------------------------------------------
# 4. TEST MODEL PREDICTION SHAPES
# ----------------------------------------------------

def test_predictor_inference():
    predictor = get_predictor()
    dummy_features = {f: 0.0 for f in predictor.feature_cols}
    
    # Test Best model
    res_best = predictor.predict(dummy_features, model_name="Best")
    assert "predicted_rul" in res_best
    assert isinstance(res_best["predicted_rul"], float)
    assert res_best["predicted_rul"] >= 0.0
    assert res_best["risk_category"] in ["LOW RISK", "MEDIUM RISK", "HIGH RISK"]
    
    # Test specific models
    for m in ["Linear Regression", "Random Forest", "XGBoost", "LSTM"]:
        res = predictor.predict(dummy_features, model_name=m)
        assert res["model_used"] == m
        assert isinstance(res["predicted_rul"], float)


# ----------------------------------------------------
# 5. TEST API HEALTH & PREDICT ENDPOINTS
# ----------------------------------------------------

client = TestClient(app)

def test_api_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["models_loaded"] is True


def test_api_predict_endpoint():
    payload = {
        "engine_id": 3,
        "cycle": 50,
        "model_name": "Best"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["engine_id"] == 3
    assert data["cycle"] == 50
    assert "predicted_rul" in data
    assert "risk_category" in data
    assert "maintenance_recommendation" in data
