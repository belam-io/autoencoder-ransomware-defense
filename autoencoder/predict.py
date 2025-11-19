import torch
from autoencoder.model import AutoEncoder
from autoencoder.utils import get_logger_helper

logger = get_logger_helper()

def load_model(model_path=None, input_dim=None):
    if model_path is None:
        model_path = '/model/autoencoder (1).pth'
    
    model = AutoEncoder(input_dim=input_dim)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def compute_error(model, input_tensor):
    with torch.no_grad():
        reconstructed = model(input_tensor)
        errors = torch.mean((input_tensor - reconstructed) ** 2, dim=1)
        return errors

def compute_threshold(errors):
    mean_err = errors.mean()
    std_err = errors.std()
    threshold = mean_err + 3 * std_err
    logger.info(f'Computed anomaly threshold: {threshold:.4f}')
    return threshold

def classify_anomaly(errors, threshold):
    return errors > threshold

def predict(model_path, input_data):
    tensor = torch.tensor(input_data.values, dtype=torch.float32)
    model = load_model(model_path=model_path, input_dim=tensor.shape[1])
    errors = compute_error(model=model, input_tensor=tensor)
    threshold = compute_threshold(errors=errors)
    anomalies = classify_anomaly(errors=errors, threshold=threshold)

    for x, is_anomaly in enumerate(anomalies):
        if is_anomaly:
            logger.warning(f'Anomaly detected at index {x}, error={errors[x].item()}')

    return anomalies, errors
