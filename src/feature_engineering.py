"""
Feature engineering module for time-series degradation telemetry.
Computes backward-looking rolling statistics, rate-of-change, and trends per engine.
Guarantees zero future temporal leakage.
"""

from typing import List, Tuple
import pandas as pd
import numpy as np

from src.config import ROLLING_WINDOWS, INFORMATIVE_SENSORS_FD001


def generate_engine_features(
    df: pd.DataFrame,
    sensor_cols: List[str] = INFORMATIVE_SENSORS_FD001,
    windows: List[int] = ROLLING_WINDOWS
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Constructs domain-specific degradation indicators for each engine:
    1. Backward-looking rolling averages (smooths sensor high-frequency noise)
    2. Backward-looking rolling standard deviations (detects increasing vibration/instability)
    3. Backward-looking rolling min & max (captures operational envelope)
    4. Rate-of-change (first difference: current - previous cycle)
    
    IMPORTANT:
    Operations are grouped strictly per engine (`engine_id`), using `min_periods=1`
    so no future records are referenced and initial cycles remain intact.
    
    Returns:
        df_engineered: DataFrame with newly engineered columns
        feature_names: List of all numeric predictor feature column names
    """
    df_out = df.copy()
    
    # Sort strictly by engine and cycle to ensure proper temporal ordering
    df_out = df_out.sort_values(by=["engine_id", "cycle"]).reset_index(drop=True)
    
    engineered_cols = []
    
    # Rate-of-change (1-step diff)
    for col in sensor_cols:
        diff_col = f"{col}_diff1"
        df_out[diff_col] = df_out.groupby("engine_id")[col].diff().fillna(0.0)
        engineered_cols.append(diff_col)
        
    # Multi-scale rolling statistics
    for w in windows:
        for col in sensor_cols:
            mean_col = f"{col}_roll_mean_{w}"
            std_col = f"{col}_roll_std_{w}"
            min_col = f"{col}_roll_min_{w}"
            max_col = f"{col}_roll_max_{w}"
            
            grouped = df_out.groupby("engine_id")[col]
            
            # Use min_periods=1 so the beginning of the engine life is not lost to NaNs
            df_out[mean_col] = grouped.rolling(window=w, min_periods=1).mean().reset_index(drop=True)
            df_out[std_col] = grouped.rolling(window=w, min_periods=1).std().fillna(0.0).reset_index(drop=True)
            df_out[min_col] = grouped.rolling(window=w, min_periods=1).min().reset_index(drop=True)
            df_out[max_col] = grouped.rolling(window=w, min_periods=1).max().reset_index(drop=True)
            
            engineered_cols.extend([mean_col, std_col, min_col, max_col])
            
    # Base predictor features
    base_features = ["cycle"] + [c for c in ["setting_1", "setting_2", "setting_3"] if c in df_out.columns] + sensor_cols
    all_feature_cols = base_features + engineered_cols
    
    # Fill any lingering edge NaNs gracefully
    df_out[all_feature_cols] = df_out[all_feature_cols].fillna(0.0)
    
    return df_out, all_feature_cols
