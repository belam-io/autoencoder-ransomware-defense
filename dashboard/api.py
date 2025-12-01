import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import threading
import time

# ----------------------------------------------------
# Global State Management
# This list will hold incoming events and will be accessed by the Streamlit app
# ----------------------------------------------------
event_data_queue = [] 

class AnomalyEvent(BaseModel):
    """Defines the data structure expected from the Kafka Consumer."""
    event: dict
    score: float
    severity: str # This is the color tag (red, yellow, green)

app = FastAPI()

@app.post("/add_event")
async def receive_event(event: AnomalyEvent):
    """
    API endpoint called by the Kafka Consumer (http://siem_dashboard:8501/add_event)
    """
    event_data_queue.append({
        "timestamp": time.time(), # Use raw timestamp for sorting
        "event": event.event,
        "score": event.score,
        "severity": event.severity
    })
    return {"message": "Event received successfully"}


def run_api():
    """Starts the FastAPI server."""
    # Run FastAPI on port 8000 (Streamlit will run on 8501)
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Start the API server in a separate thread so Streamlit can also run
# NOTE: We won't start the API here; we will start it in the docker-compose command.
# This structure is just for clarity.