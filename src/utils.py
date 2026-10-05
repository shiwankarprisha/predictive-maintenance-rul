"""
Utility functions for metrics calculation, publication-grade plotting, and result serialization.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import FIGURES_DIR, RESULTS_DIR

# Clean publication style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes regression evaluation metrics:
    - MAE: Mean Absolute Error (cycles)
    - RMSE: Root Mean Squared Error (cycles)
    - R2: Coefficient of Determination
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4)
    }


def save_json(data: Dict[str, Any], filepath: Path) -> None:
    """Save dictionary to formatted JSON."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


# ==========================================
# PUBLICATION FIGURES GENERATION
# ==========================================

def plot_engine_cycles_distribution(df: pd.DataFrame, save_path: Path = FIGURES_DIR / "eda_engine_cycles.png") -> None:
    """EDA Figure: Distribution of maximum operating cycles across engines."""
    fig, ax = plt.subplots(figsize=(8, 5))
    max_cycles = df.groupby("engine_id")["cycle"].max()
    
    sns.histplot(max_cycles, bins=15, kde=True, color="#1f77b4", ax=ax)
    ax.axvline(max_cycles.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean: {max_cycles.mean():.1f} cycles")
    ax.axvline(max_cycles.median(), color="green", linestyle=":", linewidth=1.5, label=f"Median: {max_cycles.median():.1f} cycles")
    
    ax.set_title("Distribution of Turbofan Engine Operational Lifespans (FD001)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Engine Lifespan to Failure (Cycles)", fontsize=11)
    ax.set_ylabel("Engine Count", fontsize=11)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_sensor_correlation_heatmap(df: pd.DataFrame, sensor_cols: List[str], save_path: Path = FIGURES_DIR / "eda_sensor_correlation.png") -> None:
    """EDA Figure: Correlation heatmap among informative sensor channels."""
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[sensor_cols].corr()
    
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax, cbar_kws={"label": "Pearson Correlation"})
    ax.set_title("Correlation Heatmap of Informative Turbofan Sensor Channels", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_sensor_degradation_trajectories(df: pd.DataFrame, sample_engines: List[int] = [1, 2, 3],
                                         sensors: List[str] = ["sensor_2", "sensor_3", "sensor_4", "sensor_7"],
                                         save_path: Path = FIGURES_DIR / "eda_degradation_trajectories.png") -> None:
    """EDA Figure: Telemetry trend over operational cycles for sample engines."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()
    
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, s in enumerate(sensors):
        ax = axes[i]
        for idx, eng in enumerate(sample_engines):
            subset = df[df["engine_id"] == eng]
            ax.plot(subset["cycle"], subset[s], label=f"Engine {eng}", color=palette[idx % len(palette)], alpha=0.85)
        ax.set_title(f"Degradation Telemetry: {s}", fontsize=11, fontweight="semibold")
        ax.set_ylabel("Sensor Reading", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.6)
        if i >= 2:
            ax.set_xlabel("Operational Cycle", fontsize=10)
        if i == 0:
            ax.legend(loc="upper left")
            
    plt.suptitle("Sensor Trajectories Exhibiting Monotonic Degradation Toward Failure", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_model_comparison_bars(results_df: pd.DataFrame, save_path: Path = FIGURES_DIR / "model_metric_comparison.png") -> None:
    """Model Comparison Bar Charts: MAE, RMSE, and R2 across models."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    # 1. MAE
    sns.barplot(data=results_df, x="Model", y="MAE", ax=axes[0], palette="Blues_r")
    axes[0].set_title("Mean Absolute Error (Lower is Better)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("MAE (Cycles)")
    axes[0].tick_params(axis="x", rotation=15)
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha="center", va="bottom", xytext=(0, 3), textcoords="offset points", fontsize=9)
        
    # 2. RMSE
    sns.barplot(data=results_df, x="Model", y="RMSE", ax=axes[1], palette="Oranges_r")
    axes[1].set_title("Root Mean Squared Error (Lower is Better)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("RMSE (Cycles)")
    axes[1].tick_params(axis="x", rotation=15)
    for p in axes[1].patches:
        axes[1].annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha="center", va="bottom", xytext=(0, 3), textcoords="offset points", fontsize=9)
        
    # 3. R2
    sns.barplot(data=results_df, x="Model", y="R2", ax=axes[2], palette="Greens")
    axes[2].set_title("R² Score (Higher is Better)", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("R²")
    axes[2].tick_params(axis="x", rotation=15)
    for p in axes[2].patches:
        axes[2].annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha="center", va="bottom", xytext=(0, 3), textcoords="offset points", fontsize=9)
        
    plt.suptitle("Comparative Evaluation Across Tested RUL Architectures", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, model_name: str,
                             save_path: Path = None) -> None:
    """Actual vs Predicted RUL Scatter Plot."""
    if save_path is None:
        clean_name = model_name.lower().replace(" ", "_")
        save_path = FIGURES_DIR / f"actual_vs_pred_{clean_name}.png"
        
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(y_true, y_pred, alpha=0.35, edgecolors="none", color="#2b5c8f", s=18)
    
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([0, max_val], [0, max_val], color="crimson", linestyle="--", linewidth=1.8, label="Ideal (y = x)")
    
    ax.set_title(f"Actual vs Predicted RUL — {model_name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Ground Truth RUL (Cycles)", fontsize=11)
    ax.set_ylabel("Predicted RUL (Cycles)", fontsize=11)
    ax.legend(frameon=True)
    ax.set_xlim(0, max_val + 5)
    ax.set_ylim(0, max_val + 5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, model_name: str,
                   save_path: Path = None) -> None:
    """Residual error distribution."""
    if save_path is None:
        clean_name = model_name.lower().replace(" ", "_")
        save_path = FIGURES_DIR / f"residuals_{clean_name}.png"
        
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.histplot(residuals, bins=30, kde=True, color="#4a7c59", ax=ax)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
    
    ax.set_title(f"Residual Error Distribution (True - Pred) — {model_name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Prediction Error (Cycles)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_feature_importance(importances: np.ndarray, feature_names: List[str], model_name: str,
                            top_n: int = 15, save_path: Path = None) -> None:
    """Feature importance horizontal bar chart."""
    if save_path is None:
        clean_name = model_name.lower().replace(" ", "_")
        save_path = FIGURES_DIR / f"feature_importance_{clean_name}.png"
        
    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    fi_df = fi_df.sort_values(by="Importance", ascending=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=fi_df, x="Importance", y="Feature", palette="viridis", ax=ax)
    ax.set_title(f"Top {top_n} Predictive Telemetry Features — {model_name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Normalized Importance", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_lstm_loss_curve(train_loss: List[float], val_loss: List[float],
                         save_path: Path = FIGURES_DIR / "lstm_learning_curve.png") -> None:
    """LSTM Training and Validation Loss per Epoch."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    epochs = range(1, len(train_loss) + 1)
    ax.plot(epochs, train_loss, label="Training Loss (MSE)", color="#1f77b4", linewidth=2)
    if val_loss:
        ax.plot(epochs, val_loss, label="Validation Loss (MSE)", color="#ff7f0e", linestyle="--", linewidth=2)
        
    ax.set_title("LSTM Convergence Profile: Training vs. Validation Loss", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Mean Squared Error", fontsize=11)
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_single_engine_degradation_prediction(
    engine_df: pd.DataFrame, 
    preds: np.ndarray, 
    engine_id: int,
    save_path: Path = None
) -> None:
    """
    Plots true vs predicted RUL trajectory over entire operational life of a specific engine.
    """
    if save_path is None:
        save_path = FIGURES_DIR / f"engine_{engine_id}_trajectory_prediction.png"
        
    fig, ax = plt.subplots(figsize=(9, 5))
    cycles = engine_df["cycle"].values
    y_true = engine_df["rul"].values
    
    ax.plot(cycles, y_true, label="Ground Truth RUL", color="black", linewidth=2.5)
    ax.plot(cycles, preds, label="Predicted RUL", color="#e41a1c", linestyle="--", linewidth=2)
    
    ax.axhline(20, color="darkorange", linestyle=":", label="Failure Risk Threshold (20 cycles)")
    ax.set_title(f"Engine {engine_id}: Degradation Trajectory & Life Estimation", fontsize=13, fontweight="bold")
    ax.set_xlabel("Operational Cycle", fontsize=11)
    ax.set_ylabel("Remaining Useful Life (Cycles)", fontsize=11)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
