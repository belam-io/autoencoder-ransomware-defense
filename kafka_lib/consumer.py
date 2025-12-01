import json
from kafka import KafkaConsumer
import torch
import pandas as pd
import joblib
import requests

# 🌟 FIX PART 1: Import the model class from your module.py file
# This import relies on the autoencoder/__init__.py file existing.
from autoencoder.model import AutoEncoder 


MODEL_PATH = "models/autoencoder.pth"
SCALER_PATH = "models/scaler.pkl"
ENCODERS_PATH = "models/encoders.pkl"

# 🛑 CRITICAL: DEFINE YOUR MODEL DIMENSION HERE
# You previously confirmed 21 features. This MUST match the total columns
# output by your preprocessing script (data.py).
INPUT_DIM = 21 

# 🌟 FIX PART 2: Correct model loading logic 🌟

# 1. Instantiate the AutoEncoder class with its correct dimension
model = AutoEncoder(input_dim=INPUT_DIM)

# 2. Load the state dictionary (weights) from the file
model_state = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
model.load_state_dict(model_state)

# 3. Set to evaluation mode
model.eval()

scaler = joblib.load(SCALER_PATH)
encoders = joblib.load(ENCODERS_PATH)


def push_to_dashboard(event, score, severity):
    """
    Send the anomaly to Streamlit dashboard using the Docker service name.
    """
    try:
        requests.post(
            # 🌟 FIX PART 3: Use the Docker service name for inter-container communication
            "http://siem_dashboard:8501/add_event",
            json={
                "event": event,
                "score": score,
                "severity": severity
            },
            timeout=1.0 
        )
    except requests.exceptions.RequestException:
        # Dashboard may be closed or unreachable; ignore errors
        pass
    except Exception:
        # Other potential errors
        pass


# ----------------------------
# Kafka Consumer
# ----------------------------
consumer = KafkaConsumer(
    "mpesa-c2b-transactions",
    bootstrap_servers=["kafka:9092"],
    value_deserializer=lambda x: json.loads(x.decode("utf-8")),
    auto_offset_reset="earliest",
    enable_auto_commit=True
)


# ----------------------------
# Preprocessing for Each Event
# ----------------------------
def preprocess_event(event_dict):
    df = pd.DataFrame([event_dict])

    # Apply label encoding
    for col, encoder in encoders.items():
        # Ensure the column exists before transforming
        if col in df.columns:
            df[col] = encoder.transform(df[col].astype(str))
        
    # Apply scaling
    # NOTE: The data must have the exact same 21 columns as the model input
    scaled = scaler.transform(df)

    return torch.tensor(scaled, dtype=torch.float32)


# ----------------------------
# Anomaly Scoring
# ----------------------------
def score_event(event_tensor):
    with torch.no_grad():
        reconstructed = model(event_tensor)

    # Calculate Mean Squared Error (MSE) for the reconstruction loss
    loss = torch.mean((event_tensor - reconstructed) ** 2).item()
    return loss


# ----------------------------
# Main Loop
# ----------------------------
def start_consumer():
    print("📡 Listening for Kafka events... Dashboard mode enabled.\n")

    for msg in consumer:
        event = msg.value

        # Convert → Tensor
        try:
            event_tensor = preprocess_event(event)
        except ValueError as e:
            print(f"Skipping event due to preprocessing error: {e}")
            continue

        score = score_event(event_tensor)

        # Determine severity
        if score > 0.05:
            severity = "red"       # high anomaly
        elif score > 0.01:
            severity = "yellow"    # suspicious
        else:
            severity = "green"     # normal

        # Send to Streamlit dashboard
        push_to_dashboard(event, score, severity)


if __name__ == "__main__":
    start_consumer()