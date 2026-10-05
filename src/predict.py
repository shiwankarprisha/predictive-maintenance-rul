"""
Inference and prediction pipeline.
Loads trained model artifacts, applies consistent scaling,
computes RUL predictions, failure probability, and maintenance recommendations.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union
import joblib
import numpy as np
import pandas as pd
import torch

from src.config import (
    MODELS_DIR,
    RESULTS_DIR,
    FAILURE_THRESHOLD,
    get_risk_category,
    get_maintenance_recommendation
)
from src.models import TurbofanLSTMNet


class MaintenancePredictor:
    """
    Production-ready inference engine for Predictive Maintenance.
    Handles artifact caching, multi-model selection, and failure prognostics.
    """
    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = Path(models_dir)
        self._load_artifacts()
        
    def _load_artifacts(self):
        self.scaler = joblib.load(self.models_dir / "scaler.joblib")
        self.feature_cols = joblib.load(self.models_dir / "feature_cols.joblib")
        self.lr_model = joblib.load(self.models_dir / "linear_regression.joblib")
        self.rf_model = joblib.load(self.models_dir / "random_forest.joblib")
        self.xgb_model = joblib.load(self.models_dir / "xgboost.joblib")
        self.clf_model = joblib.load(self.models_dir / "failure_classifier.joblib")
        
        # Load LSTM
        lstm_meta = joblib.load(self.models_dir / "lstm_meta.joblib")
        self.lstm_net = TurbofanLSTMNet(input_dim=lstm_meta["input_dim"])
        self.lstm_net.load_state_dict(torch.load(self.models_dir / "lstm_weights.pt", map_location="cpu"))
        self.lstm_net.eval()
        
        # Load best model name from training results
        results_path = RESULTS_DIR / "model_comparison.json"
        if results_path.exists():
            import json
            with open(results_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.best_model_name = data.get("best_model", "XGBoost")
        else:
            self.best_model_name = "XGBoost"
            
    def predict(
        self, 
        features_dict: Dict[str, float], 
        model_name: str = "Best"
    ) -> Dict[str, Any]:
        """
        Accepts raw/engineered feature dictionary, applies scaling, and generates RUL & risk forecast.
        """
        # Map selected model
        selected_model = self.best_model_name if model_name.lower() in ["best", "auto"] else model_name
        
        # Construct feature DataFrame matching trained column names
        feat_df = pd.DataFrame([{c: features_dict.get(c, 0.0) for c in self.feature_cols}])
        feat_scaled = self.scaler.transform(feat_df)
        
        # Predict RUL
        if selected_model == "Linear Regression":
            pred_rul = float(np.clip(self.lr_model.predict(feat_scaled)[0], 0, None))
        elif selected_model == "Random Forest":
            pred_rul = float(self.rf_model.predict(feat_scaled)[0])
        elif selected_model == "XGBoost":
            pred_rul = float(self.xgb_model.predict(feat_scaled)[0])
        elif selected_model == "LSTM":
            # For a single sample, tile features across sequence length
            seq = np.tile(feat_scaled, (1, 20, 1)).astype(np.float32)
            with torch.no_grad():
                pred_rul = float(np.clip(self.lstm_net(torch.tensor(seq)).cpu().numpy()[0], 0, None))
        else:
            pred_rul = float(self.xgb_model.predict(feat_scaled)[0])
            
        # Estimate failure probability using dedicated Random Forest classifier
        # Probability of class 1 (failure within FAILURE_THRESHOLD cycles)
        clf_probs = self.clf_model.predict_proba(feat_scaled)[0]
        failure_prob = float(clf_probs[1]) if len(clf_probs) > 1 else 0.0
        
        risk_category = get_risk_category(pred_rul)
        recommendation = get_maintenance_recommendation(risk_category, pred_rul)
        
        return {
            "predicted_rul": round(pred_rul, 2),
            "risk_category": risk_category,
            "failure_probability": round(failure_prob * 100, 2),  # In percent
            "failure_soon": bool(pred_rul <= FAILURE_THRESHOLD or failure_prob >= 0.5),
            "model_used": selected_model,
            "maintenance_recommendation": recommendation
        }


# Global singleton predictor
_predictor_instance: Optional[MaintenancePredictor] = None

def get_predictor() -> MaintenancePredictor:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = MaintenancePredictor()
    return _predictor_instance
