"""
Machine learning and Deep learning model definitions for RUL prediction and failure risk.
Implements:
1. Linear Regression (Baseline)
2. Random Forest Regressor
3. XGBoost Regressor
4. PyTorch LSTM Sequence Regressor
5. Random Forest Failure Classifier (Risk Classification)
"""

import time
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    RANDOM_SEED,
    LSTM_SEQUENCE_LENGTH,
    LSTM_HIDDEN_DIM,
    LSTM_NUM_LAYERS,
    LSTM_DROPOUT,
    LSTM_BATCH_SIZE,
    LSTM_EPOCHS,
    LSTM_LEARNING_RATE
)


# ==========================================
# 1. SEQUENCE GENERATOR FOR LSTM
# ==========================================

def create_lstm_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "rul",
    seq_length: int = LSTM_SEQUENCE_LENGTH
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Transforms tabular engine trajectories into 3D sequential sliding windows:
    Shape: (num_samples, seq_length, num_features).
    
    Target: RUL at the LAST time step of each window.
    Zero future leakage: each window contains strictly past and current time-steps.
    If an engine trajectory has length < seq_length, it is zero-padded at the start.
    
    Returns:
        X_seq: (N, seq_length, D)
        y_seq: (N,)
        meta_df: DataFrame with engine_id and cycle corresponding to each sequence endpoint
    """
    sequences = []
    targets = []
    meta = []
    
    for engine_id, group in df.groupby("engine_id"):
        group = group.sort_values("cycle").reset_index(drop=True)
        feats = group[feature_cols].values
        targs = group[target_col].values if target_col in group.columns else np.zeros(len(group))
        cycles = group["cycle"].values
        
        n_samples = len(group)
        if n_samples < seq_length:
            # Pad beginning with zeros
            pad_len = seq_length - n_samples
            pad_feats = np.zeros((pad_len, feats.shape[1]))
            padded_feats = np.vstack([pad_feats, feats])
            sequences.append(padded_feats)
            targets.append(targs[-1])
            meta.append({"engine_id": engine_id, "cycle": cycles[-1]})
        else:
            for i in range(seq_length, n_samples + 1):
                sequences.append(feats[i - seq_length:i, :])
                targets.append(targs[i - 1])
                meta.append({"engine_id": engine_id, "cycle": cycles[i - 1]})
                
    X_seq = np.array(sequences, dtype=np.float32)
    y_seq = np.array(targets, dtype=np.float32)
    meta_df = pd.DataFrame(meta)
    
    return X_seq, y_seq, meta_df


# ==========================================
# 2. PYTORCH LSTM ARCHITECTURE
# ==========================================

class TurbofanLSTMNet(nn.Module):
    """
    Deep recurrent neural network using stacked LSTM layers and dense projection.
    Captures temporal sensor degradation dynamics over past sequence_length cycles.
    """
    def __init__(self, input_dim: int, hidden_dim: int = LSTM_HIDDEN_DIM, 
                 num_layers: int = LSTM_NUM_LAYERS, dropout: float = LSTM_DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        lstm_out, _ = self.lstm(x)
        # Take hidden state of final time step in sequence
        last_hidden = lstm_out[:, -1, :]
        out = self.fc1(last_hidden)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out.squeeze(-1)


class PyTorchLSTMRegressor:
    """
    Scikit-learn compatible wrapper for the PyTorch LSTM model.
    """
    def __init__(self, input_dim: int, hidden_dim: int = LSTM_HIDDEN_DIM,
                 num_layers: int = LSTM_NUM_LAYERS, epochs: int = LSTM_EPOCHS,
                 batch_size: int = LSTM_BATCH_SIZE, lr: float = LSTM_LEARNING_RATE):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model = TurbofanLSTMNet(input_dim, hidden_dim, num_layers)
        self.history = {"train_loss": [], "val_loss": []}
        self.training_time = 0.0
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                                      torch.tensor(y_train, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        start_time = time.time()
        self.model.train()
        
        for epoch in range(self.epochs):
            total_train_loss = 0.0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item() * len(batch_y)
                
            epoch_train_loss = total_train_loss / len(X_train)
            self.history["train_loss"].append(epoch_train_loss)
            
            # Validation loss tracking
            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_t_x = torch.tensor(X_val, dtype=torch.float32)
                    val_t_y = torch.tensor(y_val, dtype=torch.float32)
                    val_preds = self.model(val_t_x)
                    val_loss = criterion(val_preds, val_t_y).item()
                    self.history["val_loss"].append(val_loss)
                self.model.train()
                
        self.training_time = time.time() - start_time
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            tensor_x = torch.tensor(X, dtype=torch.float32)
            preds = self.model(tensor_x).cpu().numpy()
        # RUL cannot be physically negative
        return np.clip(preds, 0, None)


# ==========================================
# 3. CLASSICAL & ENSEMBLE MODEL BUILDERS
# ==========================================

def build_linear_regression() -> LinearRegression:
    """Baseline Model: Standard Ordinary Least Squares regression."""
    return LinearRegression()


def build_random_forest_regressor() -> RandomForestRegressor:
    """Ensemble Tree Model: Robust non-linear interactions & bagging."""
    return RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )


def build_xgboost_regressor() -> xgb.XGBRegressor:
    """Gradient Boosted Trees: High tabular predictive accuracy."""
    return xgb.XGBRegressor(
        n_estimators=120,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )


def build_failure_risk_classifier() -> RandomForestClassifier:
    """Dedicated Failure Event Classifier for failure probability within threshold cycles."""
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        class_weight="balanced",  # Critical for high recall on rare failure events
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
