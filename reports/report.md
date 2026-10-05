# AI-Based Predictive Maintenance and Remaining Useful Life Estimation Using Machine Learning

**Academic Mini-Project Technical Report**  
*Course:* B.Tech Machine Learning Laboratory  
*Evaluation Date:* 12 October  
*Domain:* Prognostics and Health Management (PHM), Machine Learning & MLOps  

---

## Abstract

Predictive Maintenance (PdM) has emerged as an indispensable paradigm in modern industrial engineering and aviation, transitioning maintenance strategies from reactive breakdown repairs and rigid preventive calendar overhauls to condition-based, data-driven prognostic interventions. In this project, an end-to-end, production-grade Machine Learning and Deep Learning system is developed to estimate the Remaining Useful Life (RUL) of aircraft turbofan engines using the benchmark NASA C-MAPSS dataset. Addressing the critical vulnerability of temporal data leakage prevalent in naive time-series modeling, we implement a strict engine-wise data partitioning protocol (70% training, 15% validation, and 15% testing) alongside domain-guided backward-looking feature engineering. Four diverse regression architectures—Ordinary Least Squares Linear Regression (baseline), Random Forest Regressor (ensemble bagging), XGBoost Regressor (gradient boosting), and a stacked PyTorch Deep Long Short-Term Memory (LSTM) recurrent neural network—are trained and evaluated under identical, leakage-free conditions. Additionally, a dedicated Random Forest failure classification model is implemented, achieving an 85.71% recall on critical failure events. The system is served via a high-performance FastAPI microservice and visualized through an interactive, cross-platform Streamlit dashboard containerized with Docker.

---

## 1. Introduction

### 1.1 Background & Predictive Maintenance Paradigm
Traditional industrial maintenance regimes fall into two classical paradigms: **Corrective Maintenance** (Run-to-Failure), where machinery is operated until catastrophic physical failure occurs, and **Preventive Maintenance** (Time-Based Overhaul), where components are serviced or retired after predetermined calendar durations or operating hours regardless of their actual physical condition. While corrective maintenance incurs severe downtime expenses, uncontained component damage, and safety hazards, preventive maintenance leads to premature retirement of viable parts and unnecessary maintenance labor.

**Predictive Maintenance (PdM)** bridges this dilemma by continuously processing multivariate telemetry streams (vibrational accelerometers, pyrometers, thermocouple sensors, pressure transducers) to infer internal degradation states, detect incipient anomalies, and calculate the **Remaining Useful Life (RUL)**—defined as the number of remaining operational cycles or flight hours before an asset violates its defined safety margins.

### 1.2 Problem Statement
In turbofan engines, degradation is governed by severe thermal cycling, aerodynamic erosion, blade creep, and compressor fouling. Given multivariate sensor telemetry collected across successive flight cycles, the challenge is to construct an algorithmic model $f(\mathbf{X}_{1:t}) \rightarrow \hat{y}_t \in \mathbb{R}^+$ that accurately predicts the remaining cycles until failure without incurring temporal leakage or future information contamination.

### 1.3 Motivation & Objectives
The primary objectives of this project are:
1. **Zero Data Leakage:** Overcoming the common pitfall of random row-wise splitting by enforcing strictly engine-wise isolation.
2. **Domain-Specific Feature Engineering:** Extracting multi-scale rolling statistics (windows of 5, 10, and 20 cycles) and rate-of-change metrics strictly using backward-looking operations.
3. **Four-Model Benchmark:** Formulating and rigorously comparing four distinct predictive paradigms:
   - Linear Baseline: Ordinary Least Squares (OLS) Linear Regression.
   - Non-linear Bagging: Random Forest Regressor.
   - Gradient Boosting: XGBoost Regressor.
   - Sequential Deep Learning: PyTorch Stacked LSTM.
4. **Safety-Oriented Failure Risk Classification:** Constructing an operational risk model with high recall ($>80\%$) to alert engineers when RUL $\le 20$ cycles.
5. **Full-Stack Deployment:** Exposing prognostic endpoints via FastAPI and deploying an intuitive, responsive Streamlit dashboard containerized via Docker for cross-platform browser accessibility.

---

## 2. Literature Review

Prognostics and Health Management (PHM) using the NASA C-MAPSS dataset has witnessed tremendous advancements between 2022 and 2026. Contemporary scholarship focuses heavily on sequence modeling, physics-informed hybrid learning, and uncertainty quantification. Five recent, peer-reviewed studies are analyzed below:

### 2.1 Discussion of Recent Studies (2022–2026)

1. **Costa & Sánchez (2022)** (*Reliability Engineering & System Safety*):
   Proposed a Variational Autoencoder (VAE) framework mapping multivariate turbofan sensor readings into a low-dimensional latent manifold. Their methodology demonstrated that latent trajectory paths align monotonically with degradation stages, offering interpretability. However, the latent decoding step introduces considerable computational overhead during high-frequency real-time inference.

2. **Sahoo et al. (2022)** (*IEEE Transactions on Industrial Informatics*):
   Addressed the deterministic limitation of point prediction by implementing Conformalized Quantile Regression on C-MAPSS telemetry. The model provides statistically valid confidence intervals for RUL predictions under specified coverage guarantees. While exceptional for risk-aware aviation logistics, interval bounds widen significantly under fluctuating operational regimes.

3. **Zhou et al. (2023)** (*Sensors*):
   Developed a physics-informed machine learning approach coupling Weibull hazard distributions with ensemble gradient boosted trees. By infusing mechanical failure prior distributions into the learning objective, they achieved enhanced stability in early degradation stages. The limitation remains the dependence on accurate prior hazard distribution fitting.

4. **Wang et al. (2023)** (*Applied Sciences*):
   Investigated feature selection dynamics on C-MAPSS FD001, demonstrating that eliminating zero-variance sensors via Random Forest importance rankings improved the convergence and RUL accuracy of Multi-Layer Perceptrons (MLPs). Nevertheless, standard feed-forward networks struggle to retain temporal memory across extended flight cycles.

5. **Zhang et al. (2023)** (*IEEE Transactions on Neural Networks and Learning Systems*):
   Proposed **DLformer**, a dynamic-length Transformer network utilizing self-attention mechanisms to dynamically capture variable-length temporal dependencies across sensor histories. While establishing state-of-the-art benchmark scores on complex multi-regime subsets (FD002/FD004), self-attention displays quadratic memory scaling and requires substantial GPU infrastructure for deployment.

### 2.2 Literature Comparison Table

| # | Author(s) & Year | Publication Venue | Dataset | Methodology | Key Result / Contribution | Identified Limitation | DOI / Link |
|---|---|---|---|---|---|---|---|
| 1 | Costa & Sánchez (2022) | *Reliab. Eng. Syst. Saf.* | C-MAPSS FD001–004 | Variational Autoencoder (VAE) | High interpretability via latent degradation manifold | High computational complexity in production | [10.1016/j.ress.2022.108353](https://doi.org/10.1016/j.ress.2022.108353) |
| 2 | Sahoo et al. (2022) | *IEEE Trans. Ind. Inform.* | C-MAPSS | Conformalized Quantile Regression | Risk-aware prediction intervals with finite-sample coverage | Wide confidence bands under regime shifts | [10.1109/TII.2022.3156965](https://doi.org/10.1109/TII.2022.3156965) |
| 3 | Zhou et al. (2023) | *Sensors* (MDPI) | C-MAPSS FD001 | Weibull Distribution + Informed ML | Infuses physical reliability priors into boosting trees | Assumes fixed parametric lifetime distribution | [10.3390/s23125669](https://doi.org/10.3390/s23125669) |
| 4 | Wang et al. (2023) | *Appl. Sci.* (MDPI) | C-MAPSS FD001 | RF Feature Selection + MLP | Prunes 7 uninformative sensors; boosts regression convergence | Lacks recurrent temporal sequence modeling | [10.3390/app13127186](https://doi.org/10.3390/app13127186) |
| 5 | Zhang et al. (2023) | *IEEE Trans. Neural Netw.* | C-MAPSS | Dynamic Length Transformer (DLformer) | Captures multi-scale attention over sensor sequences | High memory footprint; complex deployment | [10.1109/TNNLS.2023.3257038](https://doi.org/10.1109/TNNLS.2023.3257038) |

---

## 3. Methodology & System Architecture

### 3.1 End-to-End Pipeline
The system follows a modular, reproducible data science architecture depicted below:

```
[Raw NASA C-MAPSS FD001 Telemetry]
              ↓
[Data Cleaning & Hygiene] → Prune 7 zero-variance sensors
              ↓
[Piecewise RUL Target Formulation] → Capped at 125 cycles
              ↓
[Engine-Wise Partitioning] → 70% Train (70) / 15% Val (15) / 15% Test (15)
              ↓
[Backward-Looking Feature Engineering] → Rolling Mean/Std/Min/Max (w=5,10,20) & Diffs
              ↓
[StandardScaler] → Fitted exclusively on training engines
              ↓
[Multi-Model Training]
  ├── Linear Regression (OLS Baseline)
  ├── Random Forest Regressor (100 Trees, Depth 15)
  ├── XGBoost Regressor (120 Estimators, Learning Rate 0.05)
  ├── PyTorch Deep LSTM (2 Layers, 64 Hidden Units, Dropout 0.2)
  └── Random Forest Classifier (Failure Risk <= 20 cycles)
              ↓
[Validation Selection] → Select Champion via lowest Validation RMSE
              ↓
[Test Evaluation] → Unbiased benchmark on held-out 15 test engines
              ↓
[FastAPI REST Service] (Port 8000) ⇄ [Streamlit UI Dashboard] (Port 8501)
              ↓
[Dockerized Deployment & Cross-Platform Browser Access]
```

### 3.2 RUL Mathematical Formulation
For any engine unit $u$ with operational trajectory spanning cycles $t \in [1, T_u]$, where $T_u$ is the cycle of terminal catastrophic failure, the unclipped raw Remaining Useful Life is defined as:
$$RUL_{\text{raw}}(u, t) = T_u - t$$

Turbofan components do not experience noticeable structural degradation immediately from their first flight cycle; they operate in a healthy baseline condition for initial cycles before fatigue initiates a downward degradation curve. Following established PHM literature (Heimes, 2008), we formulate a **piecewise linear RUL target** clipped at $RUL_{\max} = 125$ cycles:
$$RUL(u, t) = \min(RUL_{\max}, T_u - t)$$
This clipping prevents regression algorithms from penalizing healthy, non-degrading sensor readings during early operational cycles.

---

## 4. Data Cleaning, Preprocessing & Feature Engineering

### 4.1 Data Hygiene & Quality Verification
Inspection of the raw `train_FD001.txt` dataset confirmed 20,631 records across 100 engines. An automated hygiene audit established:
- **Missing Values:** Exactly 0 missing or NaN values across all 26 raw columns.
- **Duplicate Rows:** Exactly 0 duplicate records across `(engine_id, cycle)`.
- **Zero-Variance Pruning:** Variance analysis identified seven sensor channels displaying variance $< 10^{-4}$: `sensor_1`, `sensor_5`, `sensor_6`, `sensor_10`, `sensor_16`, `sensor_18`, and `sensor_19`. These sensors operate under constant sea-level static measurements and were pruned to eliminate multicollinearity and reduce dimensional noise.

### 4.2 Leakage-Free Engine-Wise Splitting
To completely prevent temporal information leakage, data partitioning is enforced strictly at the **engine unit level** with a fixed random seed (`seed=42`):
- **Training Set (70%):** 70 complete engine trajectories (14,482 cycles).
- **Validation Set (15%):** 15 complete engine trajectories (3,091 cycles): Engine IDs `[2, 30, 33, 38, 42, 49, 58, 59, 60, 64, 76, 80, 85, 95, 99]`.
- **Test Set (15%):** 15 complete engine trajectories (3,058 cycles): Engine IDs `[3, 15, 21, 22, 24, 52, 53, 61, 72, 75, 83, 87, 88, 92, 93]`.

The intersection of engine identifiers across the three subsets is verified to be empty ($\emptyset$). Furthermore, the `StandardScaler` was fitted strictly on the training partition and transformed onto validation and test partitions without recalculating mean or variance.

### 4.3 Feature Engineering Strategy
A total of **214 predictive features** were synthesized by computing backward-looking temporal aggregations grouped strictly per `engine_id`:
1. **Multi-Scale Rolling Averages:** Calculated over windows $w \in \{5, 10, 20\}$ with `min_periods=1` to smooth sensor measurement fluctuations:
   $$\mu_{s, w}(t) = \frac{1}{\min(t, w)} \sum_{k=0}^{\min(t, w)-1} s_{t-k}$$
2. **Rolling Standard Deviations:** Measuring telemetry volatility and degradation instability:
   $$\sigma_{s, w}(t) = \sqrt{\frac{1}{\min(t, w)-1} \sum_{k=0}^{\min(t, w)-1} (s_{t-k} - \mu_{s, w}(t))^2}$$
3. **Running Minima & Maxima:** Capturing dynamic extremes over windows $w \in \{5, 10, 20\}$.
4. **Rate-of-Change (First Difference):** $\Delta s_t = s_t - s_{t-1}$, capturing instantaneous acceleration of degradation.

---

## 5. Experimental Results and Discussion

### 5.1 Validation-Based Model Selection
Model selection was conducted strictly using performance on the held-out 15 validation engines, utilizing the lowest validation Root Mean Squared Error (RMSE) as the primary criterion:
- **Linear Regression:** Validation RMSE = **16.34 cycles**, Validation MAE = 12.92 cycles, $R^2$ = 0.846
- **XGBoost Regressor:** Validation RMSE = 16.62 cycles, Validation MAE = **11.42 cycles**, $R^2$ = 0.840
- **PyTorch Deep LSTM:** Validation RMSE = 17.05 cycles, Validation MAE = 12.78 cycles, $R^2$ = 0.834
- **Random Forest:** Validation RMSE = 17.28 cycles, Validation MAE = 11.64 cycles, $R^2$ = 0.827

Linear Regression achieved the lowest validation RMSE (16.34), earning selection as the champion model under the objective selection policy, closely followed by XGBoost.

### 5.2 Unbiased Evaluation on Test Engines
The final unbiased evaluation was executed on the 15 held-out test engines (3,058 telemetry records). All metric values reflect actual execution:

| Model Architecture | MAE (Cycles) ↓ | RMSE (Cycles) ↓ | $R^2$ Score ↑ | Training Time (s) | Model Selected |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression (Baseline)** | 13.73 | **16.99** | **0.833** | **0.235s** | **★ Selected Best** |
| **Random Forest Regressor** | 13.24 | 18.90 | 0.794 | 57.247s | - |
| **XGBoost Regressor** | **12.92** | 18.31 | 0.806 | 5.056s | - |
| **PyTorch Deep LSTM** | 13.37 | 18.10 | 0.813 | 136.169s | - |

### 5.3 Diagnostic Error & Trajectory Analysis
1. **Mean Absolute Error (MAE):** XGBoost achieved the lowest absolute error across test engines at 12.92 cycles, demonstrating outstanding point prediction accuracy on structured rolling features.
2. **Root Mean Squared Error (RMSE):** Linear Regression yielded the lowest RMSE (16.99 cycles), reflecting fewer extreme outlier over-predictions due to the regularizing effect of linear combinations across 214 features.
3. **Coefficient of Determination ($R^2$):** All four architectures accounted for $\sim 80\% - 83.3\%$ of total target variance, verifying that engineered features effectively encode degradation physics.
4. **Computational Efficiency:** Linear Regression completed fitting in 0.235 seconds, XGBoost in 5.056 seconds, while the deep LSTM required 136.169 seconds over 35 epochs. In real-time edge devices, tree ensembles and linear models provide immense latency advantages.

### 5.4 Binary Failure Risk Classifier Performance
In safety-critical aviation, missing an impending breakdown is far more costly than triggering an early inspection. Evaluating the Random Forest classifier on the test engines for the threshold condition ($RUL \le 20$ cycles) yielded:
- **Recall (Sensitivity):** **85.71%** (Successfully identifies the vast majority of impending engine failures).
- **Precision:** **89.11%** (False alert rate $< 11\%$).
- **F1-Score:** **0.8738**
- **Overall Accuracy:** **94.20%**

---

## 6. State-of-the-Art Model Comparison & Discussion

### 6.1 Architectural Strengths and Trade-Offs

#### 1. Linear Regression (Baseline)
- **Strengths:** Extreme computational efficiency ($0.235\text{s}$), complete mathematical transparency, stable gradients, no danger of runaway tree extrapolation.
- **Weaknesses:** Cannot model complex non-linear sensor cross-interactions without explicit polynomial feature generation.

#### 2. Random Forest Regressor
- **Strengths:** Robust to extreme sensor noise via bootstrap aggregation; provides intrinsic Gini-based feature importance rankings.
- **Weaknesses:** Substantial memory footprint ($24.3\text{ MB}$ saved binary), slow training time ($57.2\text{s}$), inability to extrapolate beyond the maximum training cycle bounds.

#### 3. XGBoost Regressor
- **Strengths:** Best overall Mean Absolute Error (12.92 cycles); highly efficient parallelized histogram tree building; resilient to unscaled inputs; built-in L1/L2 regularization prevents overfitting.
- **Weaknesses:** Requires hyperparameter calibration (learning rate, tree depth, subsample ratios).

#### 4. PyTorch Deep LSTM
- **Strengths:** Naturally processes sequential time-series windows without manual rolling feature extraction; maintains recurrent cell memory across successive cycles.
- **Weaknesses:** High computational latency ($136.2\text{s}$), sensitive to sequence length hyperparameters, requires extensive training data to outperform feature-engineered gradient boosted trees on tabular benchmarks.

### 6.2 Feature Importance Interpretability
Feature importance analysis from XGBoost and Random Forest highlighted the following top predictive channels:
1. `sensor_11_roll_mean_20`: Static pressure at High-Pressure Compressor (HPC) outlet.
2. `sensor_9_roll_mean_20`: Physical fan rotational speed.
3. `sensor_4_roll_mean_20`: Total temperature at Low-Pressure Turbine (LPT) outlet.
4. `sensor_12_roll_mean_10`: Ratio of fuel flow to static pressure.
These findings align directly with turbofan thermodynamics: HPC fouling causes pressure drops, which forces combustor temperatures to elevate to maintain thrust, manifesting as distinct upward trends in Sensor 4 and Sensor 11.

---

## 7. User Interface Development & Deployment

### 7.1 Streamlit Frontend Dashboard
An industrial-grade, responsive user interface was constructed using Streamlit (`app/streamlit_app.py`). Designed for non-technical maintenance supervisors and lab evaluators, the UI features five primary modules:
1. **Executive Dashboard:** Fleet-wide KPI summary, active engine counts, and categorized failure risk breakdown.
2. **Single-Engine Prognostics:** Dropdown engine selector, interactive cycle slider, real-time RUL prediction, failure probability metric, and color-coded risk badge.
3. **Sensor Telemetry Monitoring:** Multi-channel interactive Plotly charts overlaying raw sensor telemetry with 10-cycle rolling degradation trajectories.
4. **State-of-the-Art Model Comparison:** Interactive tables and side-by-side bar plots visualizing test set MAE, RMSE, $R^2$, and training times.
5. **About Project & Literature Review:** Detailed methodology breakdown, architecture diagram, literature review table, and viva defense notes.

### 7.2 FastAPI Backend Architecture
The backend is powered by FastAPI (`api/main.py`), offering high-throughput asynchronous REST endpoints:
- `GET /health`: System liveness and model loading status.
- `GET /model-info`: Serves serialized benchmark results and champion model metadata.
- `GET /engines`: Returns directory of all 100 fleet engines and their train/val/test partition assignments.
- `POST /predict`: Pydantic-validated endpoint accepting engine telemetry and returning predicted RUL, failure probability, and maintenance prescriptions.
- `GET /engine/{id}/trajectory`: Retrieves complete historical sensor readings and predicted RUL vectors.

### 7.3 Cross-Platform Accessibility & Dockerization
The platform is containerized using `Dockerfile` and `docker-compose.yml`, provisioning two coordinated services:
- `backend`: Runs Uvicorn serving FastAPI on port 8000.
- `frontend`: Runs Streamlit on port 8501, configured to route API queries to the backend container.

Because the system is deployed as a standard web application, cross-platform accessibility is universally supported across Windows, Linux, macOS, Android, and iOS browsers without client-side installation.

---

## 8. References

1. **Costa, N., & Sánchez, L. (2022).** Variational encoding approach for interpretable assessment of remaining useful life estimation. *Reliability Engineering & System Safety*, 222, 108353. https://doi.org/10.1016/j.ress.2022.108353
2. **Sahoo, A., Kumar, A., & Shankar, R. (2022).** Remaining Useful Life Estimation for Aircraft Engines with Risk-Aware Prediction Intervals via Conformalized Quantile Regression. *IEEE Transactions on Industrial Informatics*, 18(10), 7274–7284. https://doi.org/10.1109/TII.2022.3156965
3. **Zhou, Y., Chen, Z., & Liu, X. (2023).** Multiform Informed Machine Learning for Turbofan Engine Remaining Useful Life Prediction. *Sensors*, 23(12), 5669. https://doi.org/10.3390/s23125669
4. **Wang, Y., Zhang, Q., & Sun, H. (2023).** Remaining Useful Life Prediction of Aircraft Turbofan Engine Based on Random Forest Feature Selection and Multi-Layer Perceptron. *Applied Sciences*, 13(12), 7186. https://doi.org/10.3390/app13127186
5. **Zhang, J., Wang, P., & Yan, R. (2023).** DLformer: A Dynamic Length Transformer-Based Network for Efficient Feature Representation in Remaining Useful Life Prediction. *IEEE Transactions on Neural Networks and Learning Systems*. https://doi.org/10.1109/TNNLS.2023.3257038
6. **Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008).** Damage propagation modeling for aircraft engine run-to-failure simulation. In *2008 IEEE International Conference on Prognostics and Health Management* (pp. 1–9). IEEE.
7. **Heimes, F. O. (2008).** Recurrent neural networks for remaining useful life estimation. In *2008 International Conference on Prognostics and Health Management* (pp. 1–6). IEEE.
