# src/models/anomaly_logic.py

import numpy as np
import joblib
import torch
import json
from pathlib import Path
import torch.nn as nn


# -------------------------------------------------
# LSTM Autoencoder
# -------------------------------------------------
class LSTMAutoEncoder(nn.Module):
    def __init__(self, input_dim=11, hidden_dim=64, latent_dim=32):
        super().__init__()

        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.latent = nn.Linear(hidden_dim, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        z = self.latent(h[-1])

        dec_input = self.decoder_input(z).unsqueeze(1)
        dec_input = dec_input.repeat(1, x.size(1), 1)

        x_hat, _ = self.decoder(dec_input)
        return x_hat


# -------------------------------------------------
# Load artifacts
# -------------------------------------------------
ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"

IF_MODEL_PATH = ARTIFACTS_DIR / "models" / "isolation_forest_anomaly.pkl"
AE_MODEL_PATH = ARTIFACTS_DIR / "models" / "anomaly_lstm_autoencoder.pt"
AE_SCALER_PATH = ARTIFACTS_DIR / "scalers" / "anomaly_ae_scaler.joblib"
AE_THRESH_PATH = ARTIFACTS_DIR / "thresholds" / "ae_thresholds.json"

_if_model = joblib.load(IF_MODEL_PATH)

_ae_model = LSTMAutoEncoder()
_ae_model.load_state_dict(torch.load(AE_MODEL_PATH, map_location="cpu"))
_ae_model.eval()

_ae_scaler = joblib.load(AE_SCALER_PATH)

with open(AE_THRESH_PATH, "r") as f:
    AE_THRESHOLDS = json.load(f)


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def _zone(score: float) -> str:
    if score < 1.0:
        return "NORMAL"
    elif score < 1.5:
        return "WATCH"
    else:
        return "UNSAFE"


# -------------------------------------------------
# ✅ PUBLIC API (USED BY UI)
# -------------------------------------------------
def detect_anomaly(sequence_20x11: np.ndarray, if_features_row):
    """
    sequence_20x11 : np.ndarray (20,11)
    if_features_row : pandas.DataFrame (1,11)
    """

    # Isolation Forest
    if_score = float(_if_model.decision_function(if_features_row)[0])

    # Autoencoder
    flat = sequence_20x11.reshape(-1, 11)
    flat_scaled = _ae_scaler.transform(flat)
    seq_scaled = flat_scaled.reshape(1, 20, 11)

    with torch.no_grad():
        recon = _ae_model(torch.tensor(seq_scaled, dtype=torch.float32)).numpy()

    ae_error = float(np.mean((seq_scaled - recon) ** 2))

    p95 = AE_THRESHOLDS["ae_p95"]
    p99 = AE_THRESHOLDS["ae_p99"]

    if ae_error <= p95:
        ae_score = ae_error / p95
    else:
        ae_score = 1.0 + (ae_error - p95) / (p99 - p95)

    final_score = max(ae_score, abs(if_score))

    return {
        "anomaly_score": final_score,
        "zone": _zone(final_score),
        "ae_error": ae_error,
        "if_score": if_score,
    }
