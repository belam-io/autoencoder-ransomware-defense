import streamlit as st
import pandas as pd
import os

st.title("📊 SIEM Anomaly Dashboard")

PARQUET_FILE = "dashboard/anomalies.parquet"

# Auto-refresh every 3 seconds
st.experimental_singleton.clear()
st_autorefresh = st.experimental_rerun  # Streamlit rerun mechanism

if os.path.exists(PARQUET_FILE):
    df = pd.read_parquet(PARQUET_FILE)
    st.write(f"Showing last {len(df)} anomalies")
    st.dataframe(df)
else:
    st.write("No anomalies detected yet.")
