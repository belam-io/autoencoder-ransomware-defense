from autoencoder.predict import predict
import pandas as pd

# Replace with your actual CSV path
incoming_data = pd.read_csv('data/input.csv')

anomalies, errors = predict(
    model_path='model/autoencoder(1).pth',
    input_data=incoming_data
)

