import json
from kafka import KafkaProducer
import pandas as pd
from autoencoder.data import preprocess # Assuming this is correct

def dataset_producer(csv_path):

    producer = KafkaProducer(
        bootstrap_servers=['kafka:9092'],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    df = pd.read_csv(csv_path)
    # The 'preprocess' function must be able to find and load the necessary preprocessors if needed,
    # or ensure it can run standalone.
    scaled_df, _, _ = preprocess(df, save_scaler=False) 

    print(f"Starting to send {len(scaled_df)} events...") # Added log
    for _, row in scaled_df.iterrows():
        event = row.to_dict()
        producer.send("mpesa-c2b-transactions", event)
        # producer.flush() # Removed flush inside loop for performance, but you can keep it if you need immediate delivery assurance.
        print("Sent:", event, flush=True)

    producer.flush() # Flush all remaining messages at the end
    print("Finished sending all events.")


if __name__ == "__main__":
    # You may need to adjust the path based on where 'kafka/producer.py' is executed 
    # and where the 'data' directory is mounted in your Docker container.
    # Given the docker-compose setup, the root of the project is mounted to /app.
    # The path should be relative to /app.
    DATA_PATH = "data/kafka_mpesa_dataset.csv" 
    dataset_producer(DATA_PATH)