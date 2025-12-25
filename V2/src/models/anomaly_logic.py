# src/models/anomaly_logic.py
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from tensorflow.keras.models import load_model


# Canonical checkpoint features (20 × 11)
LSTM_FEATURES = [
    "V_mean", "V_std", "V_min", "V_max", "V_range",
    "dV_dt_mean", "dV_dt_max",
    "T_mean", "T_max", "T_delta",
    "duration_s"
]

# Autoencoder feature contract (LOCKED)
AE_FEATURES = [
    "V_range",
    "V_std",
    "dV_dt_mean",
    "dV_dt_max",
    "T_mean",
    "T_delta",
    "duration_s"
]


# Isolation Forest feature contract (LOCKED)
IF_FEATURES = [
    "V_range",
    "V_std",
    "dV_dt_mean",
    "dV_dt_max",
    "T_mean",
    "T_delta",
    "duration_s"
]

# -------------------------------------------------
# Paths
# -------------------------------------------------
ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"

IF_MODEL_PATH = ARTIFACTS_DIR / "models" / "isolation_forest_anomaly.pkl"
AE_MODEL_PATH = ARTIFACTS_DIR / "models" / "anomaly_lstm_autoencoder.keras"
AE_SCALER_PATH = ARTIFACTS_DIR / "scalers" / "anomaly_ae_scaler.joblib"
AE_THRESH_PATH = ARTIFACTS_DIR / "thresholds" / "ae_thresholds.json"


# -------------------------------------------------
# Load artifacts (ONCE)
# -------------------------------------------------
_if_model = joblib.load(IF_MODEL_PATH)

_ae_model = load_model(AE_MODEL_PATH)
_ae_scaler = joblib.load(AE_SCALER_PATH)

with open(AE_THRESH_PATH, "r") as f:
    FUSION_CFG = json.load(f)


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def _fuse_zone(if_norm: float, ae_norm: float) -> str:
    if if_norm > FUSION_CFG["if_p99"] or ae_norm > FUSION_CFG["ae_p99"]:
        return "UNSAFE"
    elif if_norm > FUSION_CFG["if_p97"] or ae_norm > FUSION_CFG["ae_p97"]:
        return "WATCH"
    else:
        return "NORMAL"


def _minmax_norm(x, xmin, xmax):
    return (x - xmin) / (xmax - xmin + 1e-6)


# -------------------------------------------------
# ✅ PUBLIC API (USED BY UI)
# -------------------------------------------------
def detect_anomaly(sequence_20x11: np.ndarray, if_features_row):
    """
    sequence_20x11 : np.ndarray (20, 11)
    if_features_row : pandas.DataFrame (1, 11)
    """

    # ---------- Isolation Forest ----------
    if_input = if_features_row[IF_FEATURES]
    if_raw = float(_if_model.decision_function(if_input)[0])


    # Normalize IF score using observed bounds
    if_norm = _minmax_norm(
        if_raw,
        FUSION_CFG.get("if_min", -1.0),
        FUSION_CFG.get("if_max", 1.0),
    )

    # ---------- Autoencoder ----------
    # Select AE features ONLY (7)
    seq_df = (
        pd.DataFrame(sequence_20x11, columns=LSTM_FEATURES)
        [AE_FEATURES]
    )

    flat = seq_df.values.reshape(-1, len(AE_FEATURES))  # (20, 7)
    flat_scaled = _ae_scaler.transform(flat)
    WINDOW = 15  # MUST match AE training

    if flat_scaled.shape[0] < WINDOW:
        pad = np.repeat(
            flat_scaled[:1],
            WINDOW - flat_scaled.shape[0],
            axis=0
        )
        window_data = np.vstack([pad, flat_scaled])
    else:
        window_data = flat_scaled[-WINDOW:]

    seq_scaled = window_data.reshape(1, WINDOW, len(AE_FEATURES))

    recon = _ae_model.predict(seq_scaled, verbose=0)
    ae_error = float(np.mean((seq_scaled - recon) ** 2))

    ae_norm = _minmax_norm(
        ae_error,
        FUSION_CFG.get("ae_min", 0.0),
        FUSION_CFG.get("ae_max", FUSION_CFG["ae_p99"]),
    )

    # ---------- Fusion ----------
    zone = _fuse_zone(if_norm, ae_norm)
    anomaly_score = max(if_norm, ae_norm)
    return {
        "anomaly_zone": zone,
        "anomaly_score": anomaly_score,
        "if_score": if_raw,
        "ae_error": ae_error,
        "if_norm": if_norm,
        "ae_norm": ae_norm,
    }
