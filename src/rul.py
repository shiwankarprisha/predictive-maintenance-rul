"""
Remaining Useful Life (RUL) target generation module.
Calculates ground truth degradation trajectories and failure risk indicators.
"""

from typing import Optional
import pandas as pd
from src.config import CLIP_RUL_MAX, FAILURE_THRESHOLD


def compute_run_to_failure_rul(df: pd.DataFrame, clip_threshold: Optional[float] = CLIP_RUL_MAX) -> pd.DataFrame:
    """
    Computes RUL for complete run-to-failure trajectories (training data).
    
    For each engine:
        RUL(t) = max_cycle - cycle(t)
        
    Optionally clips RUL at a maximum threshold to represent the healthy initial plateau,
    which is standard practice in turbofan prognostics (Heimes 2008, Babu et al. 2016).
    
    Args:
        df: DataFrame containing 'engine_id' and 'cycle' columns.
        clip_threshold: Maximum RUL ceiling (e.g. 125). If None, unclipped linear RUL is used.
        
    Returns:
        DataFrame with added 'rul' and 'rul_raw' columns.
    """
    df_out = df.copy()
    
    # Calculate maximum operating cycle for each individual engine
    max_cycles = df_out.groupby("engine_id")["cycle"].transform("max")
    
    # Ground truth remaining useful life
    df_out["rul_raw"] = max_cycles - df_out["cycle"]
    
    # Piece-wise linear RUL (clipping during initial healthy operational phase)
    if clip_threshold is not None and clip_threshold > 0:
        df_out["rul"] = df_out["rul_raw"].clip(upper=clip_threshold)
    else:
        df_out["rul"] = df_out["rul_raw"]
        
    # Binary failure indicator: 1 if failure imminent within FAILURE_THRESHOLD cycles, else 0
    df_out["failure_soon"] = (df_out["rul_raw"] <= FAILURE_THRESHOLD).astype(int)
    
    return df_out


def compute_test_set_rul(
    test_df: pd.DataFrame, 
    rul_truth: pd.Series, 
    clip_threshold: Optional[float] = CLIP_RUL_MAX
) -> pd.DataFrame:
    """
    Computes RUL for truncated test trajectories using NASA ground-truth final RUL vector.
    
    For test engine i at cycle t:
        RUL(t) = RUL_truth(i) + (max_recorded_cycle_in_test - cycle(t))
    """
    df_out = test_df.copy()
    
    # Maximum cycle recorded in the truncated test trajectory for each engine
    max_test_cycles = df_out.groupby("engine_id")["cycle"].transform("max")
    
    # Map the engine ground-truth final remaining cycles
    engine_truth_map = rul_truth.to_dict()
    df_out["truth_at_end"] = df_out["engine_id"].map(engine_truth_map)
    
    # Trajectory RUL
    df_out["rul_raw"] = df_out["truth_at_end"] + (max_test_cycles - df_out["cycle"])
    
    if clip_threshold is not None and clip_threshold > 0:
        df_out["rul"] = df_out["rul_raw"].clip(upper=clip_threshold)
    else:
        df_out["rul"] = df_out["rul_raw"]
        
    df_out["failure_soon"] = (df_out["rul_raw"] <= FAILURE_THRESHOLD).astype(int)
    df_out.drop(columns=["truth_at_end"], inplace=True)
    
    return df_out
