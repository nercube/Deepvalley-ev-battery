import pandas as pd


def normalize_columns(df: pd.DataFrame):
    df = df.copy()

    # ----------------------------
    # Normalize column names
    # ----------------------------
    df.columns = [c.strip() for c in df.columns]

    # ----------------------------
    # Asset ID
    # ----------------------------
    if "macId" in df.columns:
        df["asset_id"] = df["macId"]
    elif "macid" in df.columns:
        df["asset_id"] = df["macid"]
    else:
        raise KeyError("macId column not found")

    # ----------------------------
    # Timestamp
    # ----------------------------
    if "lastData" in df.columns:
        df["timestamp"] = pd.to_datetime(df["lastData"], errors="coerce")
    elif "createdAt" in df.columns:
        df["timestamp"] = pd.to_datetime(df["createdAt"], errors="coerce")
    elif "updatedAt" in df.columns:
        df["timestamp"] = pd.to_datetime(df["updatedAt"], errors="coerce")
    else:
        raise KeyError(
            "No valid time column found (lastData / createdAt / updatedAt)"
        )

    # ----------------------------
    # Cell voltages
    # ----------------------------
    cell_cols = [c for c in df.columns if c.startswith("cell_voltages")]
    if not cell_cols:
        raise KeyError("No cell voltage columns found")

    for idx, col in enumerate(sorted(cell_cols)):
        df[f"cell_{idx}"] = df[col]

    # ----------------------------
    # Temperature normalization
    # ----------------------------
    temp_cols = [c for c in df.columns if c.startswith("temperature")]
    if temp_cols:
        df["T_mean"] = df[temp_cols].mean(axis=1)
        df["T_max"] = df[temp_cols].max(axis=1)
        df["T_delta"] = df[temp_cols].max(axis=1) - df[temp_cols].min(axis=1)

    return df, [f"cell_{i}" for i in range(len(cell_cols))]


def validate_schema(df: pd.DataFrame):
    required_columns = {
        "asset_id",
        "timestamp",
        "pack_voltage",
        "pack_current",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
