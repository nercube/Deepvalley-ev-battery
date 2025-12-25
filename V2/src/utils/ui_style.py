import streamlit as st

def apply_style():
    st.markdown(
        """
        <style>
        body {
            background-color: #0E1117;
            color: #E6EDF3;
        }
        .stMetric {
            background-color: #161B22;
            padding: 16px;
            border-radius: 10px;
        }
        div[data-testid="stSidebar"] {
            background-color: #0E1117;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
