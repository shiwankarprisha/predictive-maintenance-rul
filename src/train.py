"""
End-to-End Training and Model Comparison Pipeline.
Executes data preprocessing, engine-wise splitting, feature engineering,
model training (Linear Regression, Random Forest, XGBoost, LSTM, and Failure Classifier),
validation-based model selection, unbiased test evaluation, figure generation, and artifact persistence.
"""

import os
import time
from pathlib import Path
from typing import Dict, Any
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

from src.config import (
    DEFAULT_SUBSET,
    MODELS_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    FAILURE_THRESHOLD
)
from src.data_loader import load_raw_train_data, get_dataset_summary
from src.rul import compute_run_to_failure_rul
from src.preprocessing import (
    inspect_data_quality,
    remove_uninformative_features,
    perform_engine_wise_split,
    fit_and_scale_features
)
from src.feature_engineering import generate_engine_features
from src.models import (
    build_linear_regression,
    build_random_forest_regressor,
    build_xgboost_regressor,
    build_failure_risk_classifier,
    PyTorchLSTMRegressor,
    create_lstm_sequences
)
from src.utils import (
    calculate_metrics,
    save_json,
    plot_engine_cycles_distribution,
    plot_sensor_correlation_heatmap,
    plot_sensor_degradation_trajectories,
    plot_model_comparison_bars,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_feature_importance,
    plot_lstm_loss_curve,
    plot_single_engine_degradation_prediction
)


def run_training_pipeline(subset: str = DEFAULT_SUBSET) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"STARTING PREDICTIVE MAINTENANCE ML PIPELINE FOR {subset}")
    print(f"=======================================================\n")
    
    # ----------------------------------------------------
    # STEP 1: Data Loading
    # ----------------------------------------------------
    print(">>> 1. Loading raw NASA C-MAPSS dataset...")
    raw_df = load_raw_train_data(subset=subset)
    summary = get_dataset_summary(raw_df)
    print(f"    Loaded {summary['num_engines']} engines, {summary['total_records']} telemetry rows.")
    
    # ----------------------------------------------------
    # STEP 2: RUL Target Generation
    # ----------------------------------------------------
    print(">>> 2. Computing ground-truth Remaining Useful Life (RUL)...")
    df_rul = compute_run_to_failure_rul(raw_df)
    
    # ----------------------------------------------------
    # STEP 3: Exploratory Data Analysis & Figures
    # ----------------------------------------------------
    print(">>> 3. Generating EDA figures...")
    plot_engine_cycles_distribution(df_rul)
    # Informative sensor columns present in raw data
    eda_sensors = [c for c in df_rul.columns if c.startswith("sensor_")]
    plot_sensor_correlation_heatmap(df_rul, eda_sensors)
    plot_sensor_degradation_trajectories(df_rul, sample_engines=[1, 2, 3])
    
    # ----------------------------------------------------
    # STEP 4: Data Cleaning & Low Variance Removal
    # ----------------------------------------------------
    print(">>> 4. Inspecting data quality and pruning zero-variance channels...")
    quality_report = inspect_data_quality(df_rul)
    print(f"    Missing values: {quality_report['missing_values']}, Duplicate rows: {quality_report['duplicate_rows']}")
    print(f"    Identified zero-variance sensors: {quality_report['low_variance_sensors']}")
    clean_df = remove_uninformative_features(df_rul)
    
    # ----------------------------------------------------
    # STEP 5: Engine-wise Split (Zero Leakage)
    # ----------------------------------------------------
    print(">>> 5. Performing Engine-wise Train (70%) / Val (15%) / Test (15%) split...")
    train_raw, val_raw, test_raw, split_info = perform_engine_wise_split(clean_df)
    print(f"    Train engines ({len(split_info['train_engines'])}): {split_info['train_engines'][:5]}...")
    print(f"    Val engines ({len(split_info['val_engines'])}): {split_info['val_engines']}")
    print(f"    Test engines ({len(split_info['test_engines'])}): {split_info['test_engines']}")
    
    # ----------------------------------------------------
    # STEP 6: Temporal Feature Engineering
    # ----------------------------------------------------
    print(">>> 6. Engineering backward-looking rolling statistics & rate-of-change...")
    train_fe, feature_cols = generate_engine_features(train_raw)
    val_fe, _ = generate_engine_features(val_raw)
    test_fe, _ = generate_engine_features(test_raw)
    print(f"    Constructed {len(feature_cols)} total predictive features.")
    
    # ----------------------------------------------------
    # STEP 7: Feature Scaling (Fit on Train ONLY)
    # ----------------------------------------------------
    print(">>> 7. Fitting StandardScaler on training set only...")
    train_scaled, val_scaled, test_scaled, scaler = fit_and_scale_features(
        train_fe, val_fe, test_fe, feature_cols
    )
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    joblib.dump(feature_cols, MODELS_DIR / "feature_cols.joblib")
    joblib.dump(split_info, MODELS_DIR / "split_info.joblib")
    
    X_train = train_scaled[feature_cols].values
    y_train = train_scaled["rul"].values
    
    X_val = val_scaled[feature_cols].values
    y_val = val_scaled["rul"].values
    
    X_test = test_scaled[feature_cols].values
    y_test = test_scaled["rul"].values
    
    # ----------------------------------------------------
    # STEP 8: Train Model 1 — Linear Regression (Baseline)
    # ----------------------------------------------------
    print(">>> 8. Training Model 1: Linear Regression (Baseline)...")
    lr_model = build_linear_regression()
    t0 = time.time()
    lr_model.fit(X_train, y_train)
    lr_time = time.time() - t0
    joblib.dump(lr_model, MODELS_DIR / "linear_regression.joblib")
    
    # ----------------------------------------------------
    # STEP 9: Train Model 2 — Random Forest Regressor
    # ----------------------------------------------------
    print(">>> 9. Training Model 2: Random Forest Regressor...")
    rf_model = build_random_forest_regressor()
    t0 = time.time()
    rf_model.fit(X_train, y_train)
    rf_time = time.time() - t0
    joblib.dump(rf_model, MODELS_DIR / "random_forest.joblib")
    
    # ----------------------------------------------------
    # STEP 10: Train Model 3 — XGBoost Regressor
    # ----------------------------------------------------
    print(">>> 10. Training Model 3: XGBoost Regressor...")
    xgb_model = build_xgboost_regressor()
    t0 = time.time()
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_time = time.time() - t0
    joblib.dump(xgb_model, MODELS_DIR / "xgboost.joblib")
    
    # ----------------------------------------------------
    # STEP 11: Train Model 4 — PyTorch LSTM Sequence Model
    # ----------------------------------------------------
    print(">>> 11. Preparing sequences and training Model 4: PyTorch LSTM...")
    X_train_seq, y_train_seq, meta_train = create_lstm_sequences(train_scaled, feature_cols)
    X_val_seq, y_val_seq, meta_val = create_lstm_sequences(val_scaled, feature_cols)
    X_test_seq, y_test_seq, meta_test = create_lstm_sequences(test_scaled, feature_cols)
    
    lstm_model = PyTorchLSTMRegressor(input_dim=len(feature_cols))
    lstm_model.fit(X_train_seq, y_train_seq, X_val_seq, y_val_seq)
    lstm_time = lstm_model.training_time
    torch.save(lstm_model.model.state_dict(), MODELS_DIR / "lstm_weights.pt")
    joblib.dump({
        "input_dim": len(feature_cols),
        "history": lstm_model.history,
        "training_time": lstm_time
    }, MODELS_DIR / "lstm_meta.joblib")
    plot_lstm_loss_curve(lstm_model.history["train_loss"], lstm_model.history["val_loss"])
    
    # ----------------------------------------------------
    # STEP 12: Train Failure Risk Classifier
    # ----------------------------------------------------
    print(">>> 12. Training Failure Risk Classifier (Random Forest Classifier)...")
    clf_model = build_failure_risk_classifier()
    y_train_clf = train_scaled["failure_soon"].values
    clf_model.fit(X_train, y_train_clf)
    joblib.dump(clf_model, MODELS_DIR / "failure_classifier.joblib")
    
    y_test_clf = test_scaled["failure_soon"].values
    y_test_clf_pred = clf_model.predict(X_test)
    clf_metrics = {
        "accuracy": round(float(accuracy_score(y_test_clf, y_test_clf_pred)), 4),
        "precision": round(float(precision_score(y_test_clf, y_test_clf_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test_clf, y_test_clf_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test_clf, y_test_clf_pred, zero_division=0)), 4)
    }
    print(f"    Classifier Test Recall: {clf_metrics['recall']}, Precision: {clf_metrics['precision']}, F1: {clf_metrics['f1']}")
    save_json(clf_metrics, RESULTS_DIR / "classifier_metrics.json")
    
    # ----------------------------------------------------
    # STEP 13: Model Selection via Validation Performance
    # ----------------------------------------------------
    print("\n>>> 13. Performing Validation-Based Model Selection...")
    val_preds_dict = {
        "Linear Regression": np.clip(lr_model.predict(X_val), 0, None),
        "Random Forest": rf_model.predict(X_val),
        "XGBoost": xgb_model.predict(X_val),
        "LSTM": lstm_model.predict(X_val_seq)
    }
    
    val_metrics = {}
    for name, preds in val_preds_dict.items():
        # For tabular, compare against y_val; for LSTM, compare against y_val_seq
        target = y_val_seq if name == "LSTM" else y_val
        val_metrics[name] = calculate_metrics(target, preds)
        print(f"    Validation -> {name:<18} | RMSE: {val_metrics[name]['rmse']:6.2f} | MAE: {val_metrics[name]['mae']:6.2f} | R²: {val_metrics[name]['r2']:6.3f}")
        
    # Selection rule: Primary criterion lowest validation RMSE
    best_model_name = min(val_metrics.keys(), key=lambda k: val_metrics[k]["rmse"])
    print(f"    ===> SELECTED BEST MODEL: {best_model_name} (Validation RMSE: {val_metrics[best_model_name]['rmse']})")
    
    # ----------------------------------------------------
    # STEP 14: Final Unbiased Evaluation on Test Set
    # ----------------------------------------------------
    print("\n>>> 14. Evaluating All Models on Unseen Test Engines...")
    test_preds_dict = {
        "Linear Regression": np.clip(lr_model.predict(X_test), 0, None),
        "Random Forest": rf_model.predict(X_test),
        "XGBoost": xgb_model.predict(X_test),
        "LSTM": lstm_model.predict(X_test_seq)
    }
    
    training_times = {
        "Linear Regression": lr_time,
        "Random Forest": rf_time,
        "XGBoost": xgb_time,
        "LSTM": lstm_time
    }
    
    comparison_records = []
    for name in ["Linear Regression", "Random Forest", "XGBoost", "LSTM"]:
        preds = test_preds_dict[name]
        target = y_test_seq if name == "LSTM" else y_test
        m = calculate_metrics(target, preds)
        t_sec = training_times[name]
        
        comparison_records.append({
            "Model": name,
            "MAE": m["mae"],
            "RMSE": m["rmse"],
            "R2": m["r2"],
            "Training Time (s)": round(t_sec, 3),
            "Selected Best": (name == best_model_name)
        })
        
        # Plot model-specific diagnostic figures
        plot_actual_vs_predicted(target, preds, name)
        plot_residuals(target, preds, name)
        
    results_df = pd.DataFrame(comparison_records)
    results_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    save_json({"models": comparison_records, "best_model": best_model_name, "val_metrics": val_metrics},
              RESULTS_DIR / "model_comparison.json")
    
    print("\n=======================================================")
    print("FINAL TEST PERFORMANCE COMPARISON TABLE")
    print("=======================================================")
    print(results_df.to_string(index=False))
    print("=======================================================\n")
    
    # ----------------------------------------------------
    # STEP 15: Generate Comparative Charts & Feature Importances
    # ----------------------------------------------------
    print(">>> 15. Generating comparative visual artifacts...")
    plot_model_comparison_bars(results_df)
    plot_feature_importance(rf_model.feature_importances_, feature_cols, "Random Forest")
    plot_feature_importance(xgb_model.feature_importances_, feature_cols, "XGBoost")
    
    # Plot example test engine degradation trajectory
    sample_test_engine = split_info["test_engines"][0]
    sample_engine_df = test_scaled[test_scaled["engine_id"] == sample_test_engine].sort_values("cycle")
    sample_preds = xgb_model.predict(sample_engine_df[feature_cols].values)
    plot_single_engine_degradation_prediction(sample_engine_df, sample_preds, sample_test_engine)
    
    print(">>> Pipeline completed successfully! All artifacts and figures persisted.")
    return {
        "results_df": results_df,
        "best_model": best_model_name,
        "split_info": split_info,
        "feature_count": len(feature_cols)
    }


if __name__ == "__main__":
    run_training_pipeline()
