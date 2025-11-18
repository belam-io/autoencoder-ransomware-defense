from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

def preprocess(data: pd.DataFrame, save_scaler=True):
    if 'timestamp' in data.columns:
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data['hour'] = data['timestamp'].dt.hour
        data['day'] = data['timestamp'].dt.day
        data.drop(columns=['timestamp'], inplace=True)

    # Handle categorical features
    categorical_cols = data.select_dtypes(include=['object']).columns
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le

    # Numeric scaling
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data[numeric_cols])
    scaled_df = pd.DataFrame(scaled, columns=numeric_cols)

    # Save for inference
    if save_scaler:
        joblib.dump(scaler, "scaler.pkl")
        joblib.dump(label_encoders, "encoders.pkl")

    return scaled_df, label_encoders, scaler


def split_data(file_path: str, test_size: float = 0.1, random_state: int = 42):
    raw_data = load_data(file_path)
    processed_data, encoders, scaler = preprocess(raw_data)

    train_set, test_set = train_test_split(
        processed_data, test_size=test_size, random_state=random_state
    )

    return train_set, test_set
