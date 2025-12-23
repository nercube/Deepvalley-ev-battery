import streamlit as st
import pandas as pd
try:
    import plotly.express as px
except ModuleNotFoundError:
    px = None
if px is None:
    st.error("Plotly not available. Check requirements.txt")
    return

from src.validation.csv_schema import normalize_columns, validate_schema
from src.utils.audit_logger import log_event
from src.features.windowing import create_time_windows
from src.features.checkpoint_features import generate_checkpoint_features
from src.inference.predict import predict_soh


# =========================================================
# 📂 DATA INTAKE PAGE
# =========================================================
def data_intake_page():
    st.header("📂 Fleet BMS Data Intake")

    uploaded = st.file_uploader(
        "Upload raw BMS CSV",
        type=["csv"]
    )

    if not uploaded:
        return

    with st.spinner("Validating CSV…"):
        df = pd.read_csv(uploaded)

        try:
            df, cell_cols = normalize_columns(df)
            validate_schema(df)

            st.success("CSV validated successfully")

            log_event(
                action="CSV_VALIDATION",
                status="SUCCESS",
                details={
                    "rows": len(df),
                    "assets": df["asset_id"].nunique(),
                    "cell_channels": len(cell_cols)
                }
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Rows", len(df))
            c2.metric("Cell Channels", len(cell_cols))
            c3.metric("Assets", df["asset_id"].nunique())

            with st.expander("Preview normalized data"):
                st.dataframe(df.head(50))

        except Exception as e:
            log_event(
                action="CSV_VALIDATION",
                status="FAILURE",
                details={"error": str(e)}
            )
            st.error(str(e))


# =========================================================
# 📊 MONITORING PAGE
# =========================================================
def monitoring_page():
    st.header("📊 SOH Monitoring")
    st.caption("Trend & stability monitoring (no accuracy metrics)")

    uploaded = st.file_uploader(
        "Upload BMS CSV",
        type=["csv"],
        key="monitoring_csv"
    )

    if not uploaded:
        st.info("Upload a CSV to begin monitoring")
        return

    # ❗ ALWAYS normalize again
    df = pd.read_csv(uploaded)
    df, _ = normalize_columns(df)

    if st.button("▶ Run Monitoring"):
        log_event(action="MONITORING_RUN", status="STARTED")

        progress = st.progress(0)
        status = st.empty()

        # ----------------------------
        status.text("Creating time windows…")
        windows = create_time_windows(df)
        progress.progress(25)

        # ----------------------------
        status.text("Extracting checkpoint features…")
        feature_rows = []

        for i, w in enumerate(windows):
            cell_cols = [c for c in w.columns if c.startswith("cell_")]

            feats = generate_checkpoint_features(w, cell_cols)
            feats["asset_id"] = w["asset_id"].iloc[0]
            feats["window_start"] = w["window_start"].iloc[0]

            feature_rows.append(feats)

            if i % max(1, len(windows) // 20) == 0:
                progress.progress(25 + int(40 * i / len(windows)))

        features_df = pd.DataFrame(feature_rows)
        progress.progress(70)

        # ----------------------------
        status.text("Running model inference…")
        results = predict_soh(features_df)
        progress.progress(100)
        status.empty()

        log_event(
            action="MONITORING_RUN",
            status="SUCCESS",
            details={
                "windows": len(results),
                "avg_soh": round(results["soh_meta"].mean(), 3)
            }
        )

        render_monitoring_dashboard(results)


# =========================================================
# 📈 DASHBOARD
# =========================================================
def render_monitoring_dashboard(df):
    st.subheader("Summary")

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg SOH (Meta)", round(df["soh_meta"].mean(), 2))
    c2.metric("Stability", stability_label(df))
    c3.metric("Drift Risk", drift_label(df))

    st.markdown("---")
    st.subheader("SOH Trend")

    fig = px.line(
        df.sort_values("window_start"),
        x="window_start",
        y="soh_meta",
        markers=True,
        title="SOH Trend (Meta Model)"
    )

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Time",
        yaxis_title="SOH",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Model Agreement (XGB vs LSTM)"):
        fig2 = px.line(
            df.sort_values("window_start"),
            x="window_start",
            y=["soh_xgb", "soh_lstm"],
            title="XGB vs LSTM Agreement"
        )

        fig2.update_layout(
            template="plotly_dark",
            height=350
        )

        st.plotly_chart(fig2, use_container_width=True)


# =========================================================
# 🧠 LABEL HELPERS
# =========================================================
def stability_label(df):
    diff = (df["soh_xgb"] - df["soh_lstm"]).abs().mean()
    return "HIGH" if diff < 0.1 else "MEDIUM"


def drift_label(df):
    slope = df["soh_meta"].iloc[-1] - df["soh_meta"].iloc[0]
    return "ELEVATED" if slope < -0.15 else "LOW"

