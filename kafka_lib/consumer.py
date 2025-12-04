import json
from kafka import KafkaConsumer
import torch
import pandas as pd
import joblib
import sys
import torch.nn.functional as F
import numpy as np
from collections import deque
import os

from autoencoder.data import preprocess

try:
    from autoencoder.model import AutoEncoder
except ImportError:
    print("❌ Could not import AutoEncoder class. Ensure autoencoder/model.py exists.")
    sys.exit(1)

consumer = KafkaConsumer(
    "mpesa-c2b-transactions",
    bootstrap_servers=["kafka:9092"],
    value_deserializer=lambda x: json.loads(x.decode("utf-8")),
    auto_offset_reset="earliest",
    enable_auto_commit=True
)

# Parameters
BATCH_SIZE = 10
SLIDING_WINDOW_SIZE = 100
K_FACTOR = 1.5
CSV_FILE = "dashboard/anomalies.csv"
MAX_ENTRIES = 1000

# Sliding window for recent reconstruction errors
recent_errors = deque(maxlen=SLIDING_WINDOW_SIZE)

def preprocess_new_data(batch_size=BATCH_SIZE):
    messages = []
    for msg in consumer:
        messages.append(msg.value)
        if len(messages) >= batch_size:
            break
    if not messages:
        return None, None
    df = pd.DataFrame(messages)
    scaled_df, _, _ = preprocess(df, save_scaler=False)
    return df, scaled_df

def compute_adaptive_threshold(errors_window, k=K_FACTOR):
    if not errors_window:
        return 0.0
    mean_err = np.mean(errors_window)
    std_err = np.std(errors_window)
    return mean_err + k * std_err

def save_anomalies(anomaly_df):
    """Append anomalies to CSV only and keep it rolling."""
    if os.path.exists(CSV_FILE):
        all_anomalies = pd.read_csv(CSV_FILE)
        all_anomalies = pd.concat([all_anomalies, anomaly_df]).tail(MAX_ENTRIES)
    else:
        all_anomalies = anomaly_df
    all_anomalies.to_csv(CSV_FILE, index=False)
    print(f"💾 Saved {len(anomaly_df)} anomalies. Total stored: {len(all_anomalies)}")

def process_kafka_batch(batch_size=BATCH_SIZE):
    try:
        scaler = joblib.load("models/scaler.pkl")
        encoders = joblib.load("models/encoders.pkl")
    except:
        print("❌ Missing scaler.pkl or encoders.pkl.")
        return

    df_raw, df_scaled = preprocess_new_data(batch_size=batch_size)
    if df_scaled is None or df_scaled.empty:
        return

    input_dim = df_scaled.shape[1]
    model = AutoEncoder(input_dim)
    model.load_state_dict(torch.load("models/autoencoder.pth"))
    model.eval()

    X_new = torch.tensor(df_scaled.values, dtype=torch.float32)
    with torch.no_grad():
        reconstructed = model(X_new)
        batch_errors = F.mse_loss(reconstructed, X_new, reduction="none").mean(dim=1).numpy()

    recent_errors.extend(batch_errors)
    threshold = compute_adaptive_threshold(recent_errors, k=K_FACTOR)
    anomalies_idx = np.where(batch_errors > threshold)[0]

    if len(anomalies_idx) > 0:
        anomaly_df = df_raw.iloc[anomalies_idx].copy()
        anomaly_df["reconstruction_error"] = batch_errors[anomalies_idx]
        anomaly_df["threshold"] = threshold
        save_anomalies(anomaly_df)
        print(f"💥 {len(anomalies_idx)} anomalies detected in this batch.")

    print(f"Batch Errors: {batch_errors.tolist()}")
    print(f"Adaptive Threshold: {threshold:.6f}")
    print(f"Anomalies in batch: {len(anomalies_idx)}\n")

if __name__ == "__main__":
    print("📡 Listening for Kafka messages with adaptive threshold...")
    while True:
        process_kafka_batch()
