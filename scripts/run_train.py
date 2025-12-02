import torch
from autoencoder.data import split_data
from autoencoder.train import train_the_model

X_train, X_test = split_data('data/simulation_logs_with_timestamp.csv')
train_tensor = torch.tensor(X_train.values, dtype=torch.float32)


input_dim = train_tensor.shape[1]

model, errors, threshold, anomalies = train_the_model(
    train_tensor,
    input_dim,
    epochs=50,
    save_path="models/autoencoder.pth"
)
