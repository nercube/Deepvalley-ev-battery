# src/features/windowing.py

import pandas as pd
from typing import List, Dict


import pandas as pd


def create_time_windows(
    df: pd.DataFrame,
    window_minutes: int = 30
):
    """
    Create time-based windows from BMS telemetry.

    Automatically detects timestamp column.
    """

    df = df.copy()

    # --------------------------------------------------
    # Detect timestamp column (robust for real BMS data)
    # --------------------------------------------------
    if "timestamp" in df.columns:
        time_col = "timestamp"
    elif "lastdata" in df.columns:
        time_col = "lastdata"
    elif "createdat" in df.columns:
        time_col = "createdat"
    elif "updatedat" in df.columns:
        time_col = "updatedat"
    else:
        raise KeyError(
            "No valid timestamp column found. "
            "Expected one of: timestamp, lastData, createdAt, updatedAt"
        )

    # Ensure datetime
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])

    # --------------------------------------------------
    # Asset grouping (macId normalized earlier)
    # --------------------------------------------------
    if "asset_id" not in df.columns:
        raise KeyError("asset_id column missing after normalization")

    windows = []

    for asset_id, g in df.groupby("asset_id"):
        g = g.sort_values(time_col)

        g["window_id"] = (
            (g[time_col] - g[time_col].min())
            .dt.total_seconds()
            // (window_minutes * 60)
        ).astype(int)

        for _, w in g.groupby("window_id"):
            if len(w) < 5:
                continue

            w = w.copy()
            w["window_start"] = w[time_col].min()
            windows.append(w)

    return windows
