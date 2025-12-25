# src/inference/predict.py

import numpy as np
import pandas as pd
from pathlib import Path
import streamlit as st

# ------------------------------------------------------------------
# PATHS
# ------------------------------------------------------------------
BASE_PATH = Path(__file__).resolve().parents[2]

# ------------------------------------------------------------------
# FEATURE CONTRACT (LOCKED – MUST MATCH TRAINING)
# ------------------------------------------------------------------
MODEL_FEATURES = [
    "V_range",
    "V_std",
    "dV_dt_mean",
    "dV_dt_max",
    "T_mean",
    "T_delta",
    "duration_s",
]

WINDOW = 20  # MUST match SOH LSTM training

# ------------------------------------------------------------------
# LAZY MODEL LOADER (STREAMLIT SAFE)
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading ML models…")
def load_models():
    import torch
    import torch.nn as nn
    import joblib

    # -----------------------------
    # PyTorch LSTM definition
    # -----------------------------
    class SOHLSTM(nn.Module):
        def __init__(self, input_dim=7):
            super().__init__()
            self.lstm = nn.LSTM(
                input_dim,
                128,
                batch_first=True
            )
            self.fc = nn.Linear(128, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    # -----------------------------
    # Load artifacts
    # -----------------------------
    xgb_model = joblib.load(
        BASE_PATH / "artifacts/models/baseline_xgb_soh_model.joblib"
    )

    meta_model = joblib.load(
        BASE_PATH / "artifacts/models/meta_soh_model.pkl"
    )

    lstm_scaler = joblib.load(
        BASE_PATH / "artifacts/scalers/lstm_scaler.joblib"
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
        "scaler": lstm_scaler,
        "lstm": lstm_model,
    }

# ------------------------------------------------------------------
# MAIN INFERENCE FUNCTION
# ------------------------------------------------------------------
def predict_soh(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run frozen SOH models on checkpoint features (V2).
    """

    models = load_models()

    torch = models["torch"]
    xgb_model = models["xgb"]
    meta_model = models["meta"]
    lstm_scaler = models["scaler"]
    lstm_model = models["lstm"]

    df = features_df.copy()

    # -----------------------------
    # Feature selection
    # -----------------------------
    X = df[MODEL_FEATURES]

    # -----------------------------
    # XGBoost prediction
    # -----------------------------
    soh_xgb = xgb_model.predict(X)

    # -----------------------------
    # LSTM sequence preparation
    # -----------------------------
    X_scaled = lstm_scaler.transform(X)

    if len(X_scaled) < WINDOW:
        pad = np.repeat(
            X_scaled[:1],
            WINDOW - len(X_scaled),
            axis=0
        )
        seq = np.vstack([pad, X_scaled])
    else:
        seq = X_scaled[-WINDOW:]

    seq = torch.tensor(
        seq[np.newaxis, :, :],
        dtype=torch.float32
    )

    with torch.no_grad():
        soh_lstm = float(lstm_model(seq).item())

    # -----------------------------
    # Meta learner
    # -----------------------------
    meta_input = np.array([[soh_xgb[-1], soh_lstm]])
    soh_meta = float(meta_model.predict(meta_input)[0])

    # -----------------------------
    # Attach outputs
    # -----------------------------
    df["soh_xgb"] = soh_xgb
    df["soh_lstm"] = soh_lstm
    df["soh_meta"] = soh_meta

    return df
