"""
Data loading module for NASA C-MAPSS turbofan engine degradation dataset.
Supports configurable subsets (FD001, FD002, FD003, FD004) and custom paths.
"""

import os
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd

from src.config import ALL_RAW_COLUMNS, RAW_DATA_DIR, DEFAULT_SUBSET


def load_raw_train_data(subset: str = DEFAULT_SUBSET, data_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load raw training data for the specified C-MAPSS subset.
    
    Args:
        subset: Name of subset, e.g., 'FD001'
        data_dir: Path to directory containing raw files. Defaults to RAW_DATA_DIR.
        
    Returns:
        pd.DataFrame with named columns (engine_id, cycle, setting_1..3, sensor_1..21)
    """
    directory = Path(data_dir) if data_dir else RAW_DATA_DIR
    file_path = directory / f"train_{subset}.txt"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Training dataset file not found at {file_path}")
        
    df = pd.read_csv(file_path, sep=r"\s+", header=None)
    # C-MAPSS files may contain extra trailing whitespace causing extra empty columns
    if df.shape[1] > len(ALL_RAW_COLUMNS):
        df = df.iloc[:, :len(ALL_RAW_COLUMNS)]
    df.columns = ALL_RAW_COLUMNS
    
    # Cast identifiers and cycle to integers
    df["engine_id"] = df["engine_id"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    
    return df


def load_raw_test_data(subset: str = DEFAULT_SUBSET, data_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load raw test data for the specified C-MAPSS subset.
    """
    directory = Path(data_dir) if data_dir else RAW_DATA_DIR
    file_path = directory / f"test_{subset}.txt"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Test dataset file not found at {file_path}")
        
    df = pd.read_csv(file_path, sep=r"\s+", header=None)
    if df.shape[1] > len(ALL_RAW_COLUMNS):
        df = df.iloc[:, :len(ALL_RAW_COLUMNS)]
    df.columns = ALL_RAW_COLUMNS
    
    df["engine_id"] = df["engine_id"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    
    return df


def load_rul_ground_truth(subset: str = DEFAULT_SUBSET, data_dir: Optional[Path] = None) -> pd.Series:
    """
    Load ground-truth RUL values for the NASA test set engines.
    """
    directory = Path(data_dir) if data_dir else RAW_DATA_DIR
    file_path = directory / f"RUL_{subset}.txt"
    
    if not file_path.exists():
        raise FileNotFoundError(f"RUL ground truth file not found at {file_path}")
        
    df = pd.read_csv(file_path, sep=r"\s+", header=None)
    # The first column contains the true RUL value at the last recorded cycle
    rul_series = df.iloc[:, 0].astype(float)
    rul_series.index = range(1, len(rul_series) + 1)
    rul_series.index.name = "engine_id"
    return rul_series


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """
    Generate high-level metadata summary of loaded dataset.
    """
    num_engines = df["engine_id"].nunique()
    total_records = len(df)
    max_cycles = df.groupby("engine_id")["cycle"].max()
    
    return {
        "num_engines": num_engines,
        "total_records": total_records,
        "min_cycles_to_failure": int(max_cycles.min()),
        "max_cycles_to_failure": int(max_cycles.max()),
        "avg_cycles_to_failure": float(max_cycles.mean()),
        "features_count": len(df.columns)
    }
