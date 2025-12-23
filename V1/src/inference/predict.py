import joblib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path


# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
BASE_PATH = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------
# LSTM ARCHITECTURE (MUST MATCH TRAINING)
# ------------------------------------------------------------------
class SOHLSTM(nn.Module):
    def __init__(self, input_dim=11):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 32),   # ✅ MUST BE 32
            nn.ReLU(),
            nn.Linear(32, 1)      # ✅ MUST BE 32 → 1
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

# ------------------------------------------------------------------
# LOAD ARTIFACTS (ONCE)
# ------------------------------------------------------------------
xgb_model = joblib.load(
    BASE_PATH / "artifacts/models/baseline_xgb_soh_model.joblib"
)

meta_model = joblib.load(
    BASE_PATH / "artifacts/models/meta_soh_model.pkl"
)

lstm_scaler = joblib.load(
    BASE_PATH / "artifacts/scalers/lstm_scaler.joblib"
)

# ---- LSTM MODEL (STATE_DICT → MODEL) ----
lstm_model = SOHLSTM()

state_dict = torch.load(
    BASE_PATH / "artifacts/models/soh_lstm_model.pt",
    map_location="cpu"
)

lstm_model.load_state_dict(state_dict)
lstm_model.eval()



# ------------------------------------------------------------------
# FEATURE CONTRACT (LOCKED)
# ------------------------------------------------------------------
CANONICAL_FEATURES = [
    "V_mean", "V_std", "V_min", "V_max", "V_range",
    "dV_dt_mean", "dV_dt_max",
    "T_mean", "T_max", "T_delta",
    "duration_s"
]


# ------------------------------------------------------------------
# MAIN INFERENCE FUNCTION
# ------------------------------------------------------------------
def predict_soh(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run frozen SOH models on canonical checkpoint features.

    Parameters
    ----------
    features_df : pd.DataFrame
        One row per window (canonical feature schema)

    Returns
    -------
    pd.DataFrame
        Input DF with appended:
        - soh_xgb
        - soh_lstm
        - soh_meta
    """

    df = features_df.copy()

    # -----------------------------
    # XGBOOST (TABULAR)
    # -----------------------------
    X_tab = df[CANONICAL_FEATURES]
    soh_xgb = xgb_model.predict(X_tab)

    # -----------------------------
    # LSTM (SEQUENCE)
    # -----------------------------
    X_scaled = lstm_scaler.transform(X_tab)

    # Ensure minimum 20 timesteps
    if X_scaled.shape[0] < 20:
        pad = np.repeat(
            X_scaled[:1],
            20 - X_scaled.shape[0],
            axis=0
        )
        X_seq = np.vstack([pad, X_scaled])
    else:
        X_seq = X_scaled[-20:]

    X_seq = torch.tensor(
        X_seq[np.newaxis, :, :],
        dtype=torch.float32
    )

    with torch.no_grad():
        soh_lstm = lstm_model(X_seq).cpu().numpy().item()

    # -----------------------------
    # META MODEL (STABILITY)
    # -----------------------------
    meta_input = np.array([[soh_xgb[-1], soh_lstm]])
    soh_meta = meta_model.predict(meta_input)[0]

    # -----------------------------
    # APPEND OUTPUTS
    # -----------------------------
    df["soh_xgb"] = soh_xgb
    df["soh_lstm"] = soh_lstm
    df["soh_meta"] = soh_meta

    return df
