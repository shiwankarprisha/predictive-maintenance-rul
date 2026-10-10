# AI-Based Predictive Maintenance and Remaining Useful Life Estimation Using Machine Learning

> **Dataset:** NASA C-MAPSS Turbofan Engine Degradation Simulation (FD001)

---

## 📌 Problem Statement

In safety-critical aviation and industrial power systems, unexpected turbofan engine component failure can cause catastrophic damage, unscheduled flight cancellations, and severe economic losses. Traditional maintenance regimes follow fixed calendar intervals or reactionary breakdowns, resulting in either premature parts retirement or dangerous in-flight failures.

Predictive Maintenance (PdM) leverages multivariate time-series sensor telemetry to quantify equipment degradation and estimate the **Remaining Useful Life (RUL)**—the exact number of operational flight cycles remaining before an engine breaches its structural safety envelope.

---

## 🎯 Objectives

1. Develop an end-to-end Machine Learning prognostic pipeline from raw sensor readings to actionable maintenance decisions.
2. Implement strict **engine-wise data partitioning** to eliminate temporal data leakage.
3. Engineer backward-looking temporal rolling statistics and rate-of-change indicators without future knowledge.
4. Train and benchmark four state-of-the-art predictive architectures:
   - **Linear Regression** (Interpretable baseline)
   - **Random Forest Regressor** (Non-linear bagging ensemble)
   - **XGBoost Regressor** (Gradient-boosted decision trees)
   - **Deep LSTM Network** (PyTorch recurrent sequential model)
5. Build a dedicated failure risk classification component (predicting failure within 20 cycles) with high recall to avoid missed failures.
6. Deploy a **FastAPI backend** and an interactive, responsive **Streamlit dashboard** containerized via Docker for cross-platform browser accessibility.

---

## 📊 Dataset & Characteristics

The benchmark dataset is the **NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)** turbofan engine dataset (`FD001` subset):

- **Fleet Size:** 100 aircraft turbofan engines run to catastrophic failure.
- **Total Records:** 20,631 operational cycles.
- **Operating Conditions:** Sea-level flight envelope (1 single operating regime, fault mode: HPC degradation).
- **Telemetry Channels:**
  - 3 Operational Settings (Altitude, Mach number, TRA)
  - 21 Temperature, Pressure, Fan Speed, and Flow Ratio Sensors

### Zero-Variance Sensor Removal

In the `FD001` subset, seven sensors operate under constant values due to sea-level constraints:

- `sensor_1`, `sensor_5`, `sensor_6`, `sensor_10`, `sensor_16`, `sensor_18`, `sensor_19` (variance < 1e-4)
  These constant channels are pruned during data hygiene to avoid numerical instability and multicollinearity.

---

## 🛡️ Critical Data Science Rigor: Engine-Wise Splitting

> **Why a standard random train/test split is strictly forbidden:**  
> Each turbofan engine generates a continuous temporal degradation trajectory over time. A naive row-wise split would randomly assign Cycle 50 of Engine 1 to the test set while placing Cycle 49 and Cycle 51 into the training set. This causes catastrophic **data leakage**, as the model simply interpolates neighboring cycles rather than learning genuine degradation physics.

### Partitioning Strategy

- **Training Set (70%):** 70 engines (`[1, 4, 5, 6, 7, 8, ...]`)
- **Validation Set (15%):** 15 engines (`[2, 30, 33, 38, 42, 49, 58, 59, 60, 64, 76, 80, 85, 95, 99]`)
- **Test Set (15%):** 15 engines (`[3, 15, 21, 22, 24, 52, 53, 61, 72, 75, 83, 87, 88, 92, 93]`)
- **Mutual Exclusivity:** The intersection between train, validation, and test engines is **strictly empty**.
- **Scaler Isolation:** Feature scalers (`StandardScaler`) are fitted **only** on the training engines and applied downstream.

---

## ⚙️ Feature Engineering

To capture physical degradation dynamics, 214 features were engineered strictly backward-looking per engine:

1. **Multi-Scale Rolling Means ($w \in \{5, 10, 20\}$):** Attenuates high-frequency sensor noise.
2. **Rolling Standard Deviations ($w \in \{5, 10, 20\}$):** Detects growing instability and aero-thermal fluctuations.
3. **Rolling Minima & Maxima ($w \in \{5, 10, 20\}$):** Establishes running operating envelopes.
4. **Rate-of-Change (1st Difference):** $\Delta s_t = s_t - s_{t-1}$ detects instantaneous degradation slopes.

---

## 🔬 Experimental Model Comparison & Results

All models were evaluated on the **exact same held-out test engines** under identical zero-leakage conditions.

### Test Set Performance Benchmarks

| Model                 | MAE (Cycles) ↓ | RMSE (Cycles) ↓ | R² Score ↑ | Training Time (s) | Best Model Selected |
| :-------------------- | -------------: | --------------: | ---------: | ----------------: | :-----------------: |
| **Linear Regression** |          13.73 |       **16.99** |  **0.833** |        **0.235s** | **★ Selected Best** |
| **Random Forest**     |          13.24 |           18.90 |      0.794 |           57.247s |          -          |
| **XGBoost Regressor** |      **12.92** |           18.31 |      0.806 |            5.056s |          -          |
| **PyTorch Deep LSTM** |          13.37 |           18.10 |      0.813 |          136.169s |          -          |

_Note: Champion model selection was performed strictly on validation set performance (Lowest Validation RMSE: 16.34) prior to test set evaluation._

### Binary Failure Risk Classifier (Impending Failure $\le 20$ Cycles)

- **Recall (Sensitivity):** **85.71%** _(Prioritized to prevent missed in-flight failures)_
- **Precision:** **89.11%**
- **F1-Score:** **0.8738**
- **Accuracy:** **94.20%**

---

## 📈 Generated Figures

All diagnostic and EDA figures are automatically generated and saved in `reports/figures/`:

- `eda_engine_cycles.png`: Lifespan distribution of turbofan engines.
- `eda_sensor_correlation.png`: Inter-sensor correlation matrix.
- `eda_degradation_trajectories.png`: Telemetry trajectories showing physical degradation.
- `actual_vs_pred_*.png`: Scatter plots for all 4 models.
- `residuals_*.png`: Error distributions for all 4 models.
- `model_metric_comparison.png`: Comparison bar charts for MAE, RMSE, and R².
- `feature_importance_xgboost.png`: Top telemetry contributors.
- `lstm_learning_curve.png`: Convergence profile over 35 training epochs.
- `engine_3_trajectory_prediction.png`: Whole-life degradation tracking on test Engine 3.

---

## 🏗️ System Architecture

```
NASA C-MAPSS Raw Data (train_FD001.txt)
                     ↓
Data Ingestion & Hygiene (Missing checks, constant sensor pruning)
                     ↓
RUL Formulation (Piecewise threshold clipping at 125 cycles)
                     ↓
Engine-Wise Split (70% Train / 15% Val / 15% Test)
                     ↓
Temporal Feature Engineering (Rolling statistics, rate-of-change)
                     ↓
Model Training & Validation Benchmarking (Linear, RF, XGBoost, LSTM)
                     ↓
Best Model Selection & Test Evaluation (Artifact persistence in models/)
                     ↓
FastAPI Backend (REST API on port 8000)
                     ↓ HTTP Request / Response
Streamlit Interactive Dashboard (Web UI on port 8501)
                     ↓
Docker & Docker-Compose Deployment
```

---

## 💻 Installation & Setup

### Prerequisites

- Python 3.10+ (Tested on Python 3.12)
- Docker & Docker-Compose (optional, for containerized run)

### Local Environment Setup

```bash
# 1. Clone repository
git clone <repo-url>
cd predictive-maintenance

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Running the End-to-End Pipeline

```bash
# Execute training, evaluation, and figure generation
python -m src.train

# Run automated tests
pytest tests/ -v
```

### Launching the Web Application

```bash
# Terminal 1: Launch FastAPI Backend
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2: Launch Streamlit Dashboard
streamlit run app/streamlit_app.py --server.port 8501
```

Open your browser at `http://localhost:8501`.

---

## 🐳 Docker Deployment

To launch the full containerized stack:

```bash
docker-compose up --build
```

- **Streamlit Web UI:** `http://localhost:8501`
- **FastAPI Interactive Docs:** `http://localhost:8000/docs`

---

## 📡 API Endpoints

- `GET /`: Service metadata and welcome message.
- `GET /health`: Liveness and model status probe.
- `GET /model-info`: Benchmark metrics, training time, and champion architecture.
- `GET /engines`: Directory of monitored engines with split assignments.
- `GET /engine/{engine_id}/trajectory`: Historical telemetry and RUL predictions for an engine.
- `POST /predict`: Predict RUL, failure probability, and maintenance recommendation for an engine or custom sensor reading.

---

## 📚 Literature Review Foundations

1. **Costa & Sánchez (2022)** — _Reliability Engineering & System Safety_, DOI: `10.1016/j.ress.2022.108353`
2. **Sahoo et al. (2022)** — _IEEE Transactions on Industrial Informatics_, DOI: `10.1109/TII.2022.3156965`
3. **Zhou et al. (2023)** — _Sensors_, DOI: `10.3390/s23125669`
4. **Wang et al. (2023)** — _Applied Sciences_, DOI: `10.3390/app13127186`
5. **Zhang et al. (2023)** — _IEEE Transactions on Neural Networks and Learning Systems_, DOI: `10.1109/TNNLS.2023.3257038`

---

## ⚠️ Limitations & Future Work

- **Single Operating Regime:** FD001 models sea-level conditions. Future iterations will generalize to multi-regime datasets (FD002/FD004).
- **Physical Sensor Noise:** Real-world sensor calibration drift could be addressed via dynamic kalman filtering.
- **Embedded Edge Deployment:** Quantizing the PyTorch LSTM using ONNX runtime for on-wing edge compute hardware.

---

## 👥 Project Team

- Divya Tiwari - A3-B4-68
- Prisha Shiwankar - A3-B4-52
- Bhoomika Khilnani - A3-B3-38
