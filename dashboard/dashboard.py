import streamlit as st
import pandas as pd
import os

st.title("📊 SIEM Anomaly Dashboard")

CSV_FILE = "dashboard/anomalies.csv"

if os.path.exists(CSV_FILE):
    df = pd.read_csv(CSV_FILE)
    st.write(f"Showing last {len(df)} anomalies")
    st.dataframe(df)
else:
    st.write("No anomalies detected yet.")


