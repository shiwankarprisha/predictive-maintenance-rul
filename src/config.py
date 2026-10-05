"""
Configuration module for Predictive Maintenance System.
All paths, hyperparameters, feature definitions, and risk thresholds are centralized here.
"""

from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPORTS_DIR / "results"

# Ensure runtime directories exist
for directory in [PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset configuration
DEFAULT_SUBSET = "FD001"
RANDOM_SEED = 42

# Column specifications for NASA C-MAPSS dataset
INDEX_COLUMNS = ["engine_id", "cycle"]
SETTING_COLUMNS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
ALL_RAW_COLUMNS = INDEX_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS

# Constant or near-zero variance sensors in FD001
# (Sensor 1, 5, 10, 16, 18, 19 have variance ~ 0 across all cycles in FD001)
LOW_VARIANCE_SENSORS_FD001 = ["sensor_1", "sensor_5", "sensor_10", "sensor_16", "sensor_18", "sensor_19"]

# Informative sensors selected for feature engineering & modeling in FD001
INFORMATIVE_SENSORS_FD001 = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_6", "sensor_7",
    "sensor_8", "sensor_9", "sensor_11", "sensor_12", "sensor_13",
    "sensor_14", "sensor_15", "sensor_17", "sensor_20", "sensor_21"
]

# Engine-wise train/val/test split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Piece-wise linear RUL clipping threshold (standard in C-MAPSS prognostic literature)
# Degradation begins after an initial healthy plateau period
CLIP_RUL_MAX = 125  # Set to None or float for unclipped RUL

# Feature engineering parameters
ROLLING_WINDOWS = [5, 10, 20]

# LSTM Sequence modeling parameters
LSTM_SEQUENCE_LENGTH = 20
LSTM_HIDDEN_DIM = 64
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.2
LSTM_BATCH_SIZE = 64
LSTM_EPOCHS = 35
LSTM_LEARNING_RATE = 0.001

# Failure risk classification thresholds (Operating cycles remaining)
FAILURE_THRESHOLD = 20        # Critical threshold: failure within 20 cycles
MEDIUM_RISK_THRESHOLD = 50    # Warning threshold: between 21 and 50 cycles

RISK_LEVEL_HIGH = "HIGH RISK"
RISK_LEVEL_MEDIUM = "MEDIUM RISK"
RISK_LEVEL_LOW = "LOW RISK"

def get_risk_category(rul: float) -> str:
    """Translates numerical RUL into human-interpretable risk category."""
    if rul <= FAILURE_THRESHOLD:
        return RISK_LEVEL_HIGH
    elif rul <= MEDIUM_RISK_THRESHOLD:
        return RISK_LEVEL_MEDIUM
    else:
        return RISK_LEVEL_LOW

def get_maintenance_recommendation(risk_category: str, rul: float) -> str:
    """Generates actionable maintenance guidance based on risk level and RUL."""
    if risk_category == RISK_LEVEL_HIGH:
        return (
            f"URGENT: Impending failure detected (Estimated RUL: {rul:.1f} cycles). "
            f"Schedule immediate inspection, ground equipment, or initiate hot-section overhaul."
        )
    elif risk_category == RISK_LEVEL_MEDIUM:
        return (
            f"ATTENTION: Moderate wear detected (Estimated RUL: {rul:.1f} cycles). "
            f"Queue for maintenance within next 20-30 cycles; inspect high-wear sensor telemetry."
        )
    else:
        return (
            f"NORMAL: Engine operates within nominal health boundaries (Estimated RUL: {rul:.1f} cycles). "
            f"Continue routine monitoring; no maintenance intervention required."
        )
