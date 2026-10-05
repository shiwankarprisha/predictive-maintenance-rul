"""
Dedicated Model Evaluation Module.
Enables standalone evaluation of trained models against test datasets,
generating MAE, RMSE, and R2 performance metrics.
"""

from typing import Dict, Any
import pandas as pd
from src.config import RESULTS_DIR
from src.utils import calculate_metrics


def load_model_comparison_results() -> pd.DataFrame:
    """Load cached model comparison results as DataFrame."""
    results_path = RESULTS_DIR / "model_comparison.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Model comparison table not found at {results_path}. Run training first.")
    return pd.read_csv(results_path)


def format_markdown_table(df: pd.DataFrame) -> str:
    """Format evaluation DataFrame into a clean Markdown table for reports and README."""
    headers = ["Model", "MAE ↓", "RMSE ↓", "R² ↑", "Training Time (s)", "Best Selected"]
    rows = []
    for _, r in df.iterrows():
        rows.append(f"| {r['Model']:<17} | {r['MAE']:>6.2f} | {r['RMSE']:>6.2f} | {r['R2']:>5.3f} | {r['Training Time (s)']:>17.3f}s | {'★ Best' if r['Selected Best'] else '-'} |")
        
    divider = "|:------------------|-------:|-------:|------:|-------------------:|:-------------:|"
    header_str = "| " + " | ".join(headers) + " |"
    return "\n".join([header_str, divider] + rows)


if __name__ == "__main__":
    df = load_model_comparison_results()
    print("\n" + format_markdown_table(df) + "\n")
