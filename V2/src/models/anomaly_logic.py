# src/models/anomaly_logic.py
import numpy as np
import pandas as pd
import joblib
import json
import torch
import torch.nn as nn
from pathlib import Path


# =================================================
# FEATURE CONTRACTS (LOCKED)
# =================================================

# Full checkpoint features (20 × 11)
LSTM_FEATURES = [
    "V_mean", "V_std", "V_min", "V_max", "V_range",
    "dV_dt_mean", "dV_dt_max",
    "T_mean", "T_max", "T_delta",
    "duration_s"
]

# AE + Isolation Forest features (7)
AE_FEATURES = [
    "V_range",
    "V_std",
    "dV_dt_mean",
    "dV_dt_max",
    "T_mean",
    "T_delta",
    "duration_s"
]

IF_FEATURES = AE_FEATURES.copy()

WINDOW = 15  # MUST match AE training


# =================================================
# PATHS
# =================================================
ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"

IF_MODEL_PATH = ARTIFACTS_DIR / "models" / "isolation_forest_anomaly.pkl"
AE_MODEL_PATH = ARTIFACTS_DIR / "models" / "anomaly_lstm_autoencoder.pt"
AE_SCALER_PATH = ARTIFACTS_DIR / "scalers" / "anomaly_ae_scaler.joblib"
AE_THRESH_PATH = ARTIFACTS_DIR / "thresholds" / "ae_thresholds.json"


# =================================================
# PYTORCH AUTOENCODER DEFINITION
# =================================================
class LSTMAutoEncoder(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64):
        super().__init__()

        self.encoder = nn.LSTM(
            input_dim, hidden_dim, batch_first=True
        )
        self.decoder = nn.LSTM(
            hidden_dim, input_dim, batch_first=True
        )

    def forward(self, x):
        enc_out, (h, _) = self.encoder(x)
        dec_input = h[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        recon, _ = self.decoder(dec_input)
        return recon


# =================================================
# LOAD ARTIFACTS (ONCE)
# =================================================
_if_model = joblib.load(IF_MODEL_PATH)

_ae_scaler = joblib.load(AE_SCALER_PATH)

_ae_model = LSTMAutoEncoder(input_dim=len(AE_FEATURES))
_ae_model.load_state_dict(torch.load(AE_MODEL_PATH, map_location="cpu"))
_ae_model.eval()

with open(AE_THRESH_PATH, "r") as f:
    FUSION_CFG = json.load(f)


# =================================================
# HELPERS
# =================================================
def _minmax_norm(x, xmin, xmax):
    return (x - xmin) / (xmax - xmin + 1e-6)


def _fuse_zone(if_norm, ae_norm):
    if if_norm > FUSION_CFG["if_p99"] or ae_norm > FUSION_CFG["ae_p99"]:
        return "UNSAFE"
    elif if_norm > FUSION_CFG["if_p97"] or ae_norm > FUSION_CFG["ae_p97"]:
        return "WATCH"
    else:
        return "NORMAL"


# =================================================
# PUBLIC API (USED BY UI)
# =================================================
def detect_anomaly(sequence_20x11: np.ndarray, if_features_row: pd.DataFrame):
    """
    sequence_20x11 : np.ndarray (20, 11)
    if_features_row : pd.DataFrame (1, 11)
    """

    # ---------- Isolation Forest ----------
    if_input = if_features_row[IF_FEATURES]
    if_raw = float(_if_model.decision_function(if_input)[0])

    if_norm = _minmax_norm(
        if_raw,
        FUSION_CFG.get("if_min", -1.0),
        FUSION_CFG.get("if_max", 1.0),
    )

    # ---------- Autoencoder ----------
    seq_df = pd.DataFrame(sequence_20x11, columns=LSTM_FEATURES)[AE_FEATURES]

    flat = seq_df.values
    flat_scaled = _ae_scaler.transform(flat)

    if flat_scaled.shape[0] < WINDOW:
        pad = np.repeat(flat_scaled[:1], WINDOW - flat_scaled.shape[0], axis=0)
        window = np.vstack([pad, flat_scaled])
    else:
        window = flat_scaled[-WINDOW:]

    x = torch.tensor(
        window[np.newaxis, :, :],
        dtype=torch.float32
    )

    with torch.no_grad():
        recon = _ae_model(x).numpy()

    ae_error = float(np.mean((x.numpy() - recon) ** 2))

    ae_norm = _minmax_norm(
        ae_error,
        FUSION_CFG.get("ae_min", 0.0),
        FUSION_CFG["ae_p99"],
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
