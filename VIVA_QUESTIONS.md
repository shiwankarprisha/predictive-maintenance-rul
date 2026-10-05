# Comprehensive Viva Voce Preparation Guide (35 Questions & Answers)

**Project Title:** AI-Based Predictive Maintenance and Remaining Useful Life Estimation Using Machine Learning  
**Target Evaluation:** B.Tech ML Lab Viva (12 October)  

---

### Q1: What is predictive maintenance (PdM)?
**Answer:** Predictive Maintenance is a condition-driven maintenance strategy that uses sensor data and machine learning to monitor the real-time health of machinery. Instead of repairing equipment only after it breaks down (reactive/corrective) or adhering to rigid calendar schedules (preventive), predictive maintenance forecasts exactly when equipment is nearing failure so maintenance can be scheduled just in time, maximizing equipment lifespan and preventing catastrophic downtime.

---

### Q2: Why did you choose the NASA C-MAPSS dataset?
**Answer:** The NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset is the globally recognized gold-standard benchmark in prognostics and health management (PHM). It provides realistic run-to-failure simulations of turbofan aircraft engines across 21 sensor channels (temperatures, pressures, fan speeds), allowing us to demonstrate authentic multi-sensor degradation dynamics.

---

### Q3: What is Remaining Useful Life (RUL)?
**Answer:** Remaining Useful Life (RUL) is the estimated amount of operational time, cycles, or distance remaining before a component or machine can no longer perform its intended function safely. In our turbofan dataset, RUL is expressed as the number of remaining operational flight cycles before the engine reaches failure.

---

### Q4: How is RUL mathematically calculated in your training dataset?
**Answer:** For run-to-failure training engines, RUL at cycle $t$ is calculated as the difference between the engine's maximum lifespan ($T_{\text{max}}$) and its current cycle:
$$\text{RUL}(t) = T_{\text{max}} - t$$
For example, if Engine 1 operated until cycle 200, at cycle 1 its RUL is $200 - 1 = 199$, and at cycle 200 its RUL is 0. Furthermore, we apply piecewise linear clipping (at 125 cycles) because engines operate without degradation during their initial healthy lifespan.

---

### Q5: Why is RUL formulated as a regression problem?
**Answer:** RUL is an inherently continuous physical quantity representing the remaining operating cycles. Formulating it as regression allows us to predict the continuous progression of degradation (e.g., 78.4 cycles, 42.1 cycles, 12.0 cycles) rather than merely a coarse label.

---

### Q6: Why did you also implement a failure classification component?
**Answer:** While RUL regression provides continuous estimation, operations and maintenance managers need immediate operational risk classification. We defined a threshold of $\text{RUL} \le 20$ cycles to trigger a high-risk failure alarm. Implementing a dedicated classifier allows us to optimize specifically for **recall** (catching 85.7% of critical failures), which is paramount in safety-critical aviation.

---

### Q7: Why can't we randomly split this dataset row-by-row?
**Answer:** In time-series degradation, successive rows for the same engine are sequentially correlated. If we used a random row-wise split, Cycle 50 of Engine 1 might be in the test set while Cycle 49 and Cycle 51 are in the training set. The model would trivially interpolate between adjacent cycles rather than learning true degradation patterns. This causes severe **temporal data leakage**.

---

### Q8: What is data leakage, and how did you prevent it?
**Answer:** Data leakage occurs when information from outside the training dataset (such as future observations or test distribution statistics) is inadvertently used to train the model. We prevented it by:
1. Enforcing strict **engine-wise splitting** (no engine appears in both train and test sets).
2. Performing feature engineering strictly backward-looking per engine.
3. Fitting the `StandardScaler` **exclusively** on training engines.

---

### Q9: Why use Linear Regression in a complex ML project?
**Answer:** Linear Regression serves as our interpretable, transparent **baseline benchmark**. It has zero hyperparameters, trains in milliseconds (0.235s), and establishes the minimum performance floor that more complex ensemble and deep learning models must beat.

---

### Q10: Why use Random Forest Regressor?
**Answer:** Random Forest is an ensemble bagging technique of decision trees that handles non-linear relationships, resists overfitting by averaging multiple decorrelated trees, handles unscaled tabular features well, and provides built-in feature importance scores.

---

### Q11: Why use XGBoost Regressor?
**Answer:** XGBoost (Extreme Gradient Boosting) builds trees sequentially, with each new tree minimizing the residual errors of the previous ones. It features regularized loss functions (L1/L2), efficient histogram-based tree learning, and is renowned for achieving state-of-the-art accuracy on structured tabular datasets.

---

### Q12: Why use an LSTM network?
**Answer:** Long Short-Term Memory (LSTM) is a recurrent neural network architecture designed specifically for sequential and time-series data. Through input, forget, and output gates, LSTMs retain long-term historical degradation memory across consecutive flight cycles without suffering from vanishing gradients.

---

### Q13: What is the main difference between Random Forest and XGBoost?
**Answer:** 
- **Random Forest** uses **bagging** (Bootstrap Aggregating): trees are trained independently and in parallel on random data subsets, and their predictions are averaged to reduce variance.
- **XGBoost** uses **boosting**: trees are trained sequentially, where each tree learns from the residual errors of prior trees to reduce bias.

---

### Q14: Why does LSTM excel with sequential sensor data?
**Answer:** Sensor degradation is non-Markovian—the current wear of a turbine blade depends on its cumulative thermal and mechanical stress history. LSTMs maintain an internal cell state vector that accumulates and selectively forgets historical stress states over consecutive time cycles.

---

### Q15: What is Mean Absolute Error (MAE)?
**Answer:** MAE is the average of the absolute differences between actual RUL and predicted RUL:
$$\text{MAE} = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|$$
It represents the expected error in units of flight cycles. Lower is better. In our project, XGBoost achieved the best MAE at 12.92 cycles.

---

### Q16: What is Root Mean Squared Error (RMSE)?
**Answer:** RMSE is the square root of the average of squared prediction errors:
$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2}$$
Because errors are squared before averaging, RMSE penalizes large prediction errors more severely than MAE.

---

### Q17: Why is RMSE especially important for predictive maintenance?
**Answer:** In aviation safety, a prediction error of 40 cycles when an engine is about to fail is far more catastrophic than four small errors of 10 cycles. RMSE disproportionately penalizes large outlier errors, making it the primary metric for model selection.

---

### Q18: What is the $R^2$ Score (Coefficient of Determination)?
**Answer:** $R^2$ indicates the proportion of variance in the true RUL that is explained by the model's features:
$$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
A score of 1.0 represents perfect prediction, while 0.0 means the model performs no better than predicting the mean RUL. Our models achieved $R^2$ scores between 0.79 and 0.83.

---

### Q19: Which evaluation metric is most critical for maintenance decisions?
**Answer:** For RUL regression, **RMSE** is primary because large over-estimations lead to unexpected failures. For failure classification ($\text{RUL} \le 20$), **Recall** is most critical because missing an impending failure (False Negative) is unacceptable in aviation.

---

### Q20: What is feature engineering, and what features did you engineer?
**Answer:** Feature engineering is the extraction of domain-informative variables from raw data. We constructed 214 features by calculating:
1. Multi-window rolling means ($w \in \{5, 10, 20\}$) to smooth sensor noise.
2. Multi-window rolling standard deviations to measure vibration/instability.
3. Running minima and maxima.
4. First-order rate-of-change differences ($s_t - s_{t-1}$).

---

### Q21: Why use rolling averages instead of only raw sensor values?
**Answer:** High-frequency turbulence and measurement noise cause individual raw sensor readings to fluctuate wildly from cycle to cycle. Rolling averages filter out high-frequency noise and highlight the underlying long-term monotonic physical degradation trajectory.

---

### Q22: How did you guarantee that feature engineering did not leak future information?
**Answer:** All rolling statistics were computed using strictly backward-looking windows within each engine group (using past cycles $t, t-1, \dots, t-w+1$). No centered or forward-looking windows were used, and operations were strictly isolated by `engine_id`.

---

### Q23: Why use engine-wise splitting instead of K-fold cross-validation on rows?
**Answer:** Row-based K-fold cross-validation suffers from temporal leakage because adjacent cycles of the same engine would be present in both train and validation folds. Engine-wise splitting ensures entire engines are held out, mirroring real-world deployment where a model must predict on completely unseen new engines.

---

### Q24: How was the champion (best) model selected?
**Answer:** The best model was selected strictly on the **Validation Set** (15 engines) using the lowest validation RMSE as the primary criterion and lowest validation MAE as the secondary criterion. It was **not** selected based on the test set, preserving the test set for unbiased final evaluation.

---

### Q25: What does the user interface (UI) do?
**Answer:** The Streamlit dashboard offers an industrial monitoring interface with 5 modules:
1. Fleet Health Dashboard with status KPI cards.
2. Single-engine RUL prediction with interactive cycle sliders and failure probability.
3. Multi-sensor telemetry monitoring with trend curves.
4. Model comparison table and interactive benchmark charts.
5. Project documentation, literature review, and viva notes.

---

### Q26: Why use Streamlit for the frontend?
**Answer:** Streamlit allows rapid, Python-native construction of reactive, data-dense web dashboards. It supports interactive widgets (sliders, selectors), integrates seamlessly with Plotly charts, and eliminates the need for complex JavaScript frontend frameworks.

---

### Q27: Why use FastAPI for the backend instead of putting everything in Streamlit?
**Answer:** FastAPI decouples model inference from UI presentation:
1. **Separation of Concerns:** The ML inference pipeline can be scaled independently of the UI.
2. **High Performance:** FastAPI is built on Starlette and Pydantic with asynchronous request handling.
3. **API Reusability:** Other client applications (mobile apps, IoT sensors, external microservices) can call the same `/predict` REST endpoint.

---

### Q28: How is the application cross-platform and accessible?
**Answer:** Because the application is web-based and containerized via Docker, it can run on any server (Windows, Linux, macOS, AWS/GCP) and be accessed by end-users via any standard web browser on desktop or mobile devices without installing any software locally.

---

### Q29: What were the most important sensor channels identified by the models?
**Answer:** Feature importance analysis revealed:
- `sensor_11` (Static pressure at HPC outlet)
- `sensor_9` (Physical core/fan rotational speed)
- `sensor_4` (Total temperature at LPT outlet)
- `sensor_12` (Ratio of fuel flow to static pressure)
These reflect thermodynamic reality: as compressor efficiency degrades, temperatures and pressure ratios change systematically to maintain required engine thrust.

---

### Q30: Why did you drop sensors 1, 5, 6, 10, 16, 18, and 19?
**Answer:** In the FD001 dataset, the engine operates under constant sea-level conditions. Variance calculations revealed these seven sensors have zero variance ($< 10^{-4}$) across all 20,631 records. Dropping them removes redundant dimensions, prevents multicollinearity, and speeds up model training.

---

### Q31: What are the main limitations of this project?
**Answer:** 
1. The FD001 subset considers only one operating condition (sea level) and one failure mode (HPC degradation).
2. The dataset comes from a physics-based simulation (C-MAPSS) rather than physical aircraft flight test recordings.
3. RUL clipping threshold (125 cycles) is a fixed heuristic rather than an adaptively detected changepoint.

---

### Q32: How could this system be improved in future work?
**Answer:** 
1. Expand to multi-regime subsets (FD002/FD004) incorporating regime-normalization techniques.
2. Implement Transformer models with self-attention mechanisms (e.g., DLformer).
3. Introduce Conformal Prediction to provide calibrated prediction intervals instead of point estimates.
4. Deploy model quantization (ONNX) for edge compute hardware directly on aircraft avionics.

---

### Q33: Why did Linear Regression perform so competitively compared to complex models?
**Answer:** Because our domain-guided feature engineering extracted 214 expressive rolling statistics and rate-of-change metrics. When high-quality linear and trend features are explicitly provided, linear models with standardized inputs can approximate smooth degradation trajectories remarkably well without overfitting.

---

### Q34: What is the significance of the 20-cycle failure threshold?
**Answer:** In commercial aviation logistics, maintenance crews require lead time (typically 15–25 flight cycles) to order replacement turbofan components, route the aircraft to a maintenance hub, and assign specialized mechanics. A 20-cycle warning window balances maintenance preparation time against premature parts replacement.

---

### Q35: What would happen if this model were deployed in a real airline fleet?
**Answer:** Unscheduled engine removals would decrease substantially, maintenance costs would decline due to proactive scheduling, flight safety would improve by preventing in-flight shutdowns, and parts inventory could be optimized based on anticipated component lifespans across the fleet.
