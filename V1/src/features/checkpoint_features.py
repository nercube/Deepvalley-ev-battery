import numpy as np
import pandas as pd


def generate_checkpoint_features(window_df: pd.DataFrame, cell_cols: list) -> dict:
    """
    Generate canonical checkpoint features from a BMS time window.

    Parameters
    ----------
    window_df : pd.DataFrame
        Time-windowed BMS data for a single asset.
    cell_cols : list
        List of per-cell voltage column names.

    Returns
    -------
    dict
        Canonical checkpoint feature dictionary (LOCKED schema).
    """

    feats = {}

    # -----------------------------
    # Voltage features
    # -----------------------------
    cell_v = window_df[cell_cols]

    feats["V_mean"] = cell_v.mean().mean()
    feats["V_std"] = cell_v.stack().std()
    feats["V_min"] = cell_v.min().min()
    feats["V_max"] = cell_v.max().max()
    feats["V_range"] = feats["V_max"] - feats["V_min"]

    # -----------------------------
    # Voltage dynamics (sparse-safe)
    # -----------------------------
    if "pack_voltage" in window_df.columns and "timestamp" in window_df.columns:
        dv = window_df["pack_voltage"].diff()
        dt = window_df["timestamp"].diff().dt.total_seconds()
        dv_dt = dv / dt.replace(0, np.nan)

        feats["dV_dt_mean"] = dv_dt.mean()
        feats["dV_dt_max"] = dv_dt.abs().max()
    else:
        feats["dV_dt_mean"] = np.nan
        feats["dV_dt_max"] = np.nan

    # -----------------------------
    # Thermal features
    # -----------------------------
    temp_cols = [c for c in window_df.columns if "temp" in c]

    if temp_cols:
        temps = window_df[temp_cols]
        feats["T_mean"] = temps.mean().mean()
        feats["T_max"] = temps.max().max()
        feats["T_delta"] = feats["T_max"] - feats["T_mean"]
    else:
        feats["T_mean"] = np.nan
        feats["T_max"] = np.nan
        feats["T_delta"] = np.nan

    # -----------------------------
    # Usage / duration
    # -----------------------------
    if "timestamp" in window_df.columns:
        duration = (
            window_df["timestamp"].max() - window_df["timestamp"].min()
        ).total_seconds()
        feats["duration_s"] = max(duration, 1.0)
    else:
        feats["duration_s"] = np.nan

    return feats
