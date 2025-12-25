import numpy as np
import pandas as pd
from pathlib import Path
import streamlit as st

BASE_PATH = Path(__file__).resolve().parents[2]

MODEL_FEATURES = [
    "V_range",
    "V_std",
    "dV_dt_mean",
    "dV_dt_max",
    "T_mean",
    "T_delta",
    "duration_s",
]

WINDOW = 20


@st.cache_resource(show_spinner="Loading ML models…")
def load_models():
    import torch
    import torch.nn as nn
    import joblib

    class SOHLSTM(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, 128, batch_first=True)
            self.fc = nn.Linear(128, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1])

    xgb = joblib.load(
        BASE_PATH / "artifacts/models/baseline_xgb_soh_model.joblib"
    )

    meta = joblib.load(
        BASE_PATH / "artifacts/models/meta_soh_model.pkl"
    )

    lstm = SOHLSTM(len(MODEL_FEATURES))
    lstm.load_state_dict(
        torch.load(
            BASE_PATH / "artifacts/models/soh_lstm_model.pt",
            map_location="cpu",
        )
    )
    lstm.eval()

    return {
        "torch": torch,
        "xgb": xgb,
        "meta": meta,
        "lstm": lstm,
    }


def _normalize(X: np.ndarray) -> np.ndarray:
    """Stateless, production-safe normalization"""
    return (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)


def predict_soh(features_df: pd.DataFrame) -> pd.DataFrame:
    models = load_models()

    torch = models["torch"]
    xgb = models["xgb"]
    meta = models["meta"]
    lstm = models["lstm"]

    df = features_df.copy()
    X = df[MODEL_FEATURES].values

    # XGBoost
    soh_xgb = xgb.predict(df[MODEL_FEATURES])

    # LSTM
    X_norm = _normalize(X)

    if len(X_norm) < WINDOW:
        pad = np.repeat(X_norm[:1], WINDOW - len(X_norm), axis=0)
        seq = np.vstack([pad, X_norm])
    else:
        seq = X_norm[-WINDOW:]

    seq = torch.tensor(seq[None, :, :], dtype=torch.float32)

    with torch.no_grad():
        soh_lstm = float(lstm(seq).item())

    # Meta
    soh_meta = float(meta.predict([[soh_xgb[-1], soh_lstm]])[0])

    df["soh_xgb"] = soh_xgb
    df["soh_lstm"] = soh_lstm
    df["soh_meta"] = soh_meta

    return df
