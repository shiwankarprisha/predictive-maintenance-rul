"""
Data cleaning, anomaly/outlier analysis, feature pruning, and engine-wise splitting module.
Guarantees zero data leakage across train, validation, and test subsets.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import (
    INDEX_COLUMNS,
    SETTING_COLUMNS,
    SENSOR_COLUMNS,
    RANDOM_SEED,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    LOW_VARIANCE_SENSORS_FD001
)


def inspect_data_quality(df: pd.DataFrame) -> Dict[str, any]:
    """
    Performs comprehensive data hygiene checks:
    - Missing values
    - Duplicate rows
    - Low-variance constant sensors
    """
    missing_count = int(df.isnull().sum().sum())
    duplicate_count = int(df.duplicated(subset=INDEX_COLUMNS).sum())
    
    # Calculate feature variances for sensor columns
    variances = df[SENSOR_COLUMNS].var()
    constant_features = variances[variances < 1e-4].index.tolist()
    
    return {
        "missing_values": missing_count,
        "duplicate_rows": duplicate_count,
        "low_variance_sensors": constant_features,
        "sensor_variances": variances.to_dict()
    }


def remove_uninformative_features(
    df: pd.DataFrame, 
    drop_sensors: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Drops sensors that provide negligible variance across operational cycles.
    For FD001: Sensors 1, 5, 10, 16, 18, 19 operate under constant sea-level conditions
    and show zero variance; dropping them reduces curse of dimensionality and noise.
    """
    to_drop = drop_sensors if drop_sensors is not None else LOW_VARIANCE_SENSORS_FD001
    existing_to_drop = [c for c in to_drop if c in df.columns]
    return df.drop(columns=existing_to_drop)


def perform_engine_wise_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_seed: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, List[int]]]:
    """
    Splits the dataset at the ENGINE LEVEL to strictly prevent temporal and data leakage.
    
    VIVA EXPLANATION:
    "Because each engine generates a temporal degradation trajectory, a random row-wise
    split would leak sequential sensor correlation between training and testing. Therefore,
    the dataset was split at the engine level so that no engine's operational cycles appear
    in more than one partition."
    
    Returns:
        train_df, val_df, test_df, engine_split_dict
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"
    
    unique_engines = np.sort(df["engine_id"].unique())
    num_engines = len(unique_engines)
    
    # Deterministic shuffle
    rng = np.random.RandomState(random_seed)
    shuffled_engines = rng.permutation(unique_engines)
    
    n_train = int(round(train_ratio * num_engines))
    n_val = int(round(val_ratio * num_engines))
    
    train_engines = shuffled_engines[:n_train].tolist()
    val_engines = shuffled_engines[n_train:n_train + n_val].tolist()
    test_engines = shuffled_engines[n_train + n_val:].tolist()
    
    # Verify mutual exclusivity
    assert len(set(train_engines).intersection(val_engines)) == 0, "Leakage detected: train & val overlap"
    assert len(set(train_engines).intersection(test_engines)) == 0, "Leakage detected: train & test overlap"
    assert len(set(val_engines).intersection(test_engines)) == 0, "Leakage detected: val & test overlap"
    
    train_df = df[df["engine_id"].isin(train_engines)].copy()
    val_df = df[df["engine_id"].isin(val_engines)].copy()
    test_df = df[df["engine_id"].isin(test_engines)].copy()
    
    split_info = {
        "train_engines": sorted(train_engines),
        "val_engines": sorted(val_engines),
        "test_engines": sorted(test_engines)
    }
    
    return train_df, val_df, test_df, split_info


def fit_and_scale_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Fits a StandardScaler ONLY on the training engine features,
    then transforms train, validation, and test sets.
    This guarantees zero leakage of validation/test distributions into the scaler.
    """
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])
    
    train_scaled = train_df.copy()
    val_scaled = val_df.copy()
    test_scaled = test_df.copy()
    
    train_scaled[feature_cols] = scaler.transform(train_df[feature_cols])
    val_scaled[feature_cols] = scaler.transform(val_df[feature_cols])
    test_scaled[feature_cols] = scaler.transform(test_df[feature_cols])
    
    return train_scaled, val_scaled, test_scaled, scaler
