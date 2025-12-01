import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from autoencoder.model import AutoEncoder
from autoencoder.utils import get_logger_helper

logger = get_logger_helper()

def train_the_model(
    data,
    input_dim,
    epochs=50,
    lr=1e-3,
    batch_size=32,
    save_path=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create DataLoader
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = AutoEncoder(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()

        running_loss = 0.0
        for batch in loader:
            batch = batch[0].to(device)

            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(loader)
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - start_time
        remaining = (epochs - epoch - 1) * epoch_time

        print(
            f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}, "
            f"Elapsed: {elapsed:.2f}s, ETA: {remaining:.2f}s"
        )

    # Compute reconstruction errors
    model.eval()
    with torch.no_grad():
        reconstructed = model(data.to(device))
        errors = torch.mean((data.to(device) - reconstructed) ** 2, dim=1)

    mean_err = errors.mean()
    std_err = errors.std()
    threshold = mean_err + 3 * std_err
    anomalies = errors > threshold


    if save_path:
        torch.save(model.state_dict(), save_path)

    return model, errors, threshold, anomalies
