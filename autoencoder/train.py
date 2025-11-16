import time
import torch
import torch.nn as nn
import torch.optim as optim
from autoencoder.model import AutoEncoder
from autoencoder.utils import get_logger_helper

logger = get_logger_helper()

def train_the_model(data, input_dim, epochs=50, lr=1e-3, save_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    model = AutoEncoder(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    start_time = time.time()  

    for epoch in range(epochs):
        epoch_start = time.time()  #

        model.train()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()

        elapsed = time.time() - start_time 
        epoch_time = time.time() - epoch_start  
        remaining = (epochs - epoch - 1) * epoch_time 

        print(
            f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}, "
            f"Elapsed: {elapsed:.2f}s, ETA: {remaining:.2f}s"
        )
        logger.info(
            f"Epoch {epoch+1}, Loss: {loss.item():.6f}, "
            f"Elapsed: {elapsed:.2f}s, ETA: {remaining:.2f}s"
        )

    model.eval()
    with torch.no_grad():
        reconstructed = model(data)
        errors = torch.mean((data - reconstructed) ** 2, dim=1)

    mean_err = errors.mean()
    std_err = errors.std()
    threshold = mean_err + 3 * std_err
    anomalies = errors > threshold

    logger.info(f"Threshold: {threshold.item():.6f}")

    if save_path:
        torch.save(model.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")

    return model, errors, threshold, anomalies
