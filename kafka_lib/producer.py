import json
from kafka import KafkaProducer
import pandas as pd
from autoencoder.data import preprocess

def dataset_producer(csv_path):

    producer = KafkaProducer(
        bootstrap_servers=['kafka:9092'],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    df = pd.read_csv(csv_path)

    scaled_df, _, _ = preprocess(df, save_scaler=False) 

    print(f"Starting to send {len(scaled_df)} events...") 
    for _, row in scaled_df.iterrows():
        event = row.to_dict()
        producer.send("mpesa-c2b-transactions", event)
        print("Sent:", event, flush=True)

    producer.flush() 
    print("Finished sending all events.")


if __name__ == "__main__":

    DATA_PATH = "data/kafka_mpesa_dataset.csv" 
    dataset_producer(DATA_PATH)