"""
Pydantic schemas for FastAPI request validation and structured responses.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class TelemetryInput(BaseModel):
    """Payload representing a single operational cycle's sensor and setting telemetry."""
    engine_id: Optional[int] = Field(default=1, description="Engine identifier")
    cycle: Optional[int] = Field(default=100, description="Current operating cycle count")
    model_name: Optional[str] = Field(default="Best", description="Model architecture to use: 'Best', 'Linear Regression', 'Random Forest', 'XGBoost', 'LSTM'")
    sensors: Optional[Dict[str, float]] = Field(default=None, description="Optional raw dictionary of sensor readings: e.g. {'sensor_2': 642.5, ...}")


class EnginePredictionRequest(BaseModel):
    """Request for predicting RUL for a specific engine in the fleet at a given cycle."""
    engine_id: int = Field(..., description="Engine ID (e.g., 3, 15, 21, etc.)")
    cycle: Optional[int] = Field(default=None, description="Cycle to predict at; if None, latest cycle is used")
    model_name: Optional[str] = Field(default="Best", description="Selected model architecture")


class PredictionResponse(BaseModel):
    """Output prognostic assessment and maintenance guidance."""
    engine_id: Optional[int] = None
    cycle: Optional[int] = None
    predicted_rul: float = Field(..., description="Estimated Remaining Useful Life in operating cycles")
    risk_category: str = Field(..., description="'LOW RISK', 'MEDIUM RISK', or 'HIGH RISK'")
    failure_probability: float = Field(..., description="Estimated probability of failure within 20 cycles (%)")
    failure_soon: bool = Field(..., description="True if RUL <= 20 cycles or high failure probability")
    model_used: str = Field(..., description="Name of the model employed for inference")
    maintenance_recommendation: str = Field(..., description="Actionable maintenance recommendation")


class ModelMetric(BaseModel):
    Model: str
    MAE: float
    RMSE: float
    R2: float
    Training_Time_s: float = Field(..., alias="Training Time (s)")
    Selected_Best: bool = Field(..., alias="Selected Best")


class ModelInfoResponse(BaseModel):
    """Model governance metadata, benchmark metrics, and selected champion model."""
    champion_model: str
    dataset: str
    selection_criterion: str
    feature_count: int
    models: List[Dict[str, Any]]
    classifier_metrics: Dict[str, float]


class HealthResponse(BaseModel):
    status: str
    version: str
    system: str
    models_loaded: bool
