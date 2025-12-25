# src/app/streamlit_admin.py

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.utils.ui_style import apply_style
from src.monitoring.monitoring_ui import data_intake_page, monitoring_page
from src.monitoring.anomaly_ui import anomaly_page


st.set_page_config(
    page_title="EV Battery ML Admin",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_style()

st.sidebar.title("EV Battery ML")
st.sidebar.caption("Admin Console")

page = st.sidebar.radio(
    "Navigation",
    [
        "📂 Data Intake",
        "📊 Monitoring",
        "🛡 Anomalies",
        "🧠 Training Readiness",
        "📦 Model Registry",
        "📜 Audit Logs"
    ]
)

st.sidebar.markdown("---")
st.sidebar.success("Models Frozen • v2.0")

if page == "📂 Data Intake":
    data_intake_page()

elif page == "📊 Monitoring":
    monitoring_page()

elif page == "🛡 Anomalies":
    anomaly_page()

else:
    st.info("Module under construction")
