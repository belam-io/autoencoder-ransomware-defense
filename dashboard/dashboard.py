import streamlit as st
import pandas as pd
from datetime import datetime
import time # Import time for sleeping
from api import event_data_queue # 🌟 IMPORT THE SHARED QUEUE

st.set_page_config(page_title="Anomaly Monitor", layout="wide")

st.title("🚨 Real-Time Transaction Anomaly Dashboard")

# History buffer (stored in Streamlit session state)
if "events" not in st.session_state:
    st.session_state.events = []

# Sidebar — thresholds
st.sidebar.header("Thresholds")
yellow_threshold = st.sidebar.slider("Suspicious Threshold", 0.0001, 0.02, 0.005)
red_threshold = st.sidebar.slider("Anomaly Threshold", 0.0001, 0.05, 0.01)

st.sidebar.info(f"""
🟢 Normal: score < {yellow_threshold}  
🟡 Suspicious: {yellow_threshold} ≤ score < {red_threshold}  
🔴 Anomaly: score ≥ {red_threshold}
""")

st.markdown("### Incoming Events")

# ----------------------------------------------------
# 🌟 FIX: Function to pull data from the shared API queue
# ----------------------------------------------------
def pull_new_events():
    global event_data_queue
    
    # Process all events currently in the queue
    while event_data_queue:
        new_event = event_data_queue.pop(0) # Get the oldest event
        
        # Merge raw event, score, and classified status
        st.session_state.events.append({
            "timestamp": datetime.fromtimestamp(new_event["timestamp"]).strftime("%H:%M:%S"),
            **new_event["event"],
            "anomaly_score": new_event["score"],
            "status": classify(new_event["score"])
        })


def classify(score):
    if score >= red_threshold:
        return "🔴 Anomaly"
    elif score >= yellow_threshold:
        return "🟡 Suspicious"
    else:
        return "🟢 Normal"

# ----------------------------------------------------
# 🌟 Auto-refresh loop
# ----------------------------------------------------
while True:
    
    # 1. Pull data from the API thread
    pull_new_events()
    
    # 2. Re-render the UI
    df = pd.DataFrame(st.session_state.events)

    with st.container():
        if not df.empty:
            df_display = df.sort_values(by="timestamp", ascending=False)
            
            # Color rows based on status
            def color_row(row):
                if "🔴" in row["status"]:
                    return ["background-color: #ffcccc"] * len(row)
                elif "🟡" in row["status"]:
                    return ["background-color: #fff4cc"] * len(row)
                else:
                    return ["background-color: #ccffcc"] * len(row)

            st.dataframe(df_display.style.apply(color_row, axis=1), use_container_width=True)
        else:
            st.info("Waiting for events...")

    # 3. Wait 1 second before refreshing
    time.sleep(1) 
    st.experimental_rerun()