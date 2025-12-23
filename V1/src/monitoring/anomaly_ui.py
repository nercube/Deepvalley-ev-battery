# src/monitoring/anomaly_ui.py

import streamlit as st
import pandas as pd
import plotly.express as px

from src.validation.csv_schema import normalize_columns, validate_schema
from src.features.windowing import create_time_windows
from src.features.checkpoint_features import generate_checkpoint_features
from src.models.anomaly_logic import detect_anomaly
from src.utils.audit_logger import log_event


def anomaly_page():
    st.header("🛡 Anomaly & Safety Review")
    st.caption("Safety-only inference • Models frozen (v1.0)")

    uploaded = st.file_uploader(
        "Upload raw BMS CSV",
        type=["csv"],
        key="anomaly_csv"
    )

    if not uploaded:
        st.info("Upload a CSV to run anomaly detection")
        return

    try:
        df = pd.read_csv(uploaded)
        df, cell_cols = normalize_columns(df)
        validate_schema(df)
    except Exception as e:
        st.error(str(e))
        return

    if st.button("▶ Run Anomaly Detection"):
        log_event(action="ANOMALY_RUN", status="STARTED")

        windows = create_time_windows(df)

        if len(windows) < 20:
            st.error("At least 20 windows required")
            return

        feats = []
        for w in windows:
            row = generate_checkpoint_features(w, cell_cols)
            row["asset_id"] = w["asset_id"].iloc[0]
            row["window_start"] = w["window_start"].iloc[0]
            feats.append(row)

        feats_df = pd.DataFrame(feats)

        records = []
        for i in range(19, len(feats_df)):
            seq = feats_df.iloc[i-19:i+1].drop(
                columns=["asset_id", "window_start"]
            ).values

            if_row = feats_df.iloc[[i]].drop(
                columns=["asset_id", "window_start"]
            )

            res = detect_anomaly(seq, if_row)

            records.append({
                "asset_id": feats_df.iloc[i]["asset_id"],
                "window_start": feats_df.iloc[i]["window_start"],
                "anomaly_zone": res["zone"],
                "anomaly_score": res["anomaly_score"],
                "ae_error": res["ae_error"],
                "if_score": res["if_score"],
            })

        out = pd.DataFrame(records)
        render_anomaly_dashboard(out)


def render_anomaly_dashboard(df):
    c1, c2, c3 = st.columns(3)
    c1.metric("NORMAL", (df.anomaly_zone == "NORMAL").sum())
    c2.metric("WATCH", (df.anomaly_zone == "WATCH").sum())
    c3.metric("UNSAFE", (df.anomaly_zone == "UNSAFE").sum())

    fig = px.line(
        df,
        x="window_start",
        y="anomaly_score",
        color="anomaly_zone",
        title="Anomaly Severity Over Time"
    )
    fig.update_layout(template="plotly_dark")

    st.plotly_chart(fig, use_container_width=True)

    critical = df[df.anomaly_zone != "NORMAL"]
    if critical.empty:
        st.success("No WATCH or UNSAFE windows detected")
    else:
        st.dataframe(critical, use_container_width=True)
