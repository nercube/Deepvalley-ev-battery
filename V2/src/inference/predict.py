import numpy as np
import pandas as pd
from pathlib import Path
import streamlit as st


# ------------------------------------------------------------------
# PATHS
# ------------------------------------------------------------------
BASE_PATH = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------
# FEATURE CONTRACTS (LOCKED)
# ------------------------------------------------------------------

MODEL_FEATURES = [
    "V_range",
    "V_std",
    "dV_dt_mean",
    "dV_dt_max",
    "T_mean",
    "T_delta",
    "duration_s"
]

# ------------------------------------------------------------------
# LAZY MODEL LOADER (CRITICAL FOR DEPLOYMENT)
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading ML models…")
def load_models():
    """
    Load all ML models lazily.
    Runs once per Streamlit session.
    """

    import joblib
    from tensorflow.keras.models import load_model

    xgb_model = joblib.load(
        BASE_PATH / "artifacts/models/baseline_xgb_soh_model.joblib"
    )

    meta_model = joblib.load(
        BASE_PATH / "artifacts/models/meta_soh_model.pkl"
    )

    lstm_scaler = joblib.load(
        BASE_PATH / "artifacts/scalers/lstm_scaler.joblib"
    )

    lstm_model = load_model(
        BASE_PATH / "artifacts/models/soh_lstm_model.keras"
    )

    return {
        "xgb": xgb_model,
        "meta": meta_model,
        "scaler": lstm_scaler,
        "lstm": lstm_model
    }


# ------------------------------------------------------------------
# MAIN INFERENCE FUNCTION
# ------------------------------------------------------------------
def predict_soh(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run frozen SOH models on checkpoint features.
    """

    models = load_models()

    xgb_model = models["xgb"]
    meta_model = models["meta"]
    lstm_scaler = models["scaler"]
    lstm_model = models["lstm"]

    df = features_df.copy()

    # -----------------------------
    # SHARED FEATURES (MATCH TRAINING)
    # -----------------------------
    X_model = df[MODEL_FEATURES]

    # -----------------------------
    # XGBOOST
    # -----------------------------
    soh_xgb = xgb_model.predict(X_model)

    # -----------------------------
    # LSTM (SEQUENCE)
    # -----------------------------
    X_scaled = lstm_scaler.transform(X_model)

    if X_scaled.shape[0] < 20:
        pad = np.repeat(
            X_scaled[:1],
            20 - X_scaled.shape[0],
            axis=0
        )
        X_seq = np.vstack([pad, X_scaled])
    else:
        X_seq = X_scaled[-20:]

    X_seq = X_seq[np.newaxis, :, :]  # (1, 20, 7)

    soh_lstm = float(
        lstm_model.predict(X_seq, verbose=0)[0][0]
    )
    # -----------------------------
    # META MODEL
    # -----------------------------
    meta_input = np.array([[soh_xgb[-1], soh_lstm]])
    soh_meta = float(meta_model.predict(meta_input)[0])

    # -----------------------------
    # APPEND OUTPUTS
    # -----------------------------
    df["soh_xgb"] = soh_xgb
    df["soh_lstm"] = soh_lstm
    df["soh_meta"] = soh_meta

    return df
