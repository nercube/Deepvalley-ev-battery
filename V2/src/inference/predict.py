# src/inference/predict.py

import numpy as np
import pandas as pd
from pathlib import Path
import streamlit as st

# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_PATH = Path(__file__).resolve().parents[2]

# -------------------------------------------------
# Feature contract (LOCKED)
# -------------------------------------------------
MODEL_FEATURES = [
    "V_range",
    "V_std",
    "dV_dt_mean",
    "dV_dt_max",
    "T_mean",
    "T_delta",
    "duration_s",
]

WINDOW = 15  # MUST match LSTM training

# -------------------------------------------------
# Lazy model loader (Streamlit-safe)
# -------------------------------------------------
@st.cache_resource(show_spinner="Loading ML models…")
def load_models():
    import torch
    import torch.nn as nn
    import joblib

    # -----------------------------
    # PyTorch LSTM definition
    # -----------------------------
    class SOHLSTM(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=128,
                batch_first=True
            )
            self.fc = nn.Linear(128, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    # -----------------------------
    # Load models
    # -----------------------------
    xgb_model = joblib.load(
        BASE_PATH / "artifacts/models/baseline_xgb_soh_model.joblib"
    )

    meta_model = joblib.load(
        BASE_PATH / "artifacts/models/meta_soh_model.pkl"
    )

    lstm_model = SOHLSTM(input_dim=len(MODEL_FEATURES))
    lstm_model.load_state_dict(
        torch.load(
            BASE_PATH / "artifacts/models/soh_lstm_model.pt",
            map_location="cpu",
        )
    )
    lstm_model.eval()

    return {
        "torch": torch,
        "xgb": xgb_model,
        "meta": meta_model,
        "lstm": lstm_model,
    }

# -------------------------------------------------
# Main inference function
# -------------------------------------------------
def predict_soh(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run frozen SOH inference (XGB + LSTM + Meta)
    TensorFlow-free, production-safe.
    """

    models = load_models()

    torch = models["torch"]
    xgb = models["xgb"]
    meta = models["meta"]
    lstm = models["lstm"]

    df = features_df.copy()

    # -----------------------------
    # Select features
    # -----------------------------
    X = df[MODEL_FEATURES].values.astype(np.float32)

    # -----------------------------
    # Safe normalization (no scaler)
    # -----------------------------
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)

    # -----------------------------
    # XGBoost (tabular)
    # -----------------------------
    soh_xgb = xgb.predict(X_norm)

    # -----------------------------
    # LSTM (sequence)
    # -----------------------------
    if len(X_norm) < WINDOW:
        pad = np.repeat(
            X_norm[:1],
            WINDOW - len(X_norm),
            axis=0
        )
        seq = np.vstack([pad, X_norm])
    else:
        seq = X_norm[-WINDOW:]

    seq = torch.tensor(
        seq[None, :, :],
        dtype=torch.float32
    )

    with torch.no_grad():
        soh_lstm = float(lstm(seq).item())

    # -----------------------------
    # Meta model
    # -----------------------------
    soh_meta = float(
        meta.predict([[soh_xgb[-1], soh_lstm]])[0]
    )

    # -----------------------------
    # Append outputs
    # -----------------------------
    df["soh_xgb"] = soh_xgb
    df["soh_lstm"] = soh_lstm
    df["soh_meta"] = soh_meta

    return df
