from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

def preprocess(data: pd.DataFrame):
    if 'target' in data.columns:
        target = data['target']
        data = data.drop(columns=['target'])
    else:
        target = None

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    scaled_df = pd.DataFrame(scaled, columns=data.columns)
    return scaled_df, target

def split_data(file_path: str, test_size: float = 0.1, random_state: int = 42):
    raw_data = load_data(file_path)
    scaled_data, target = preprocess(raw_data)

    train_set, test_set = train_test_split(
        scaled_data, test_size=test_size, random_state=random_state
    )

    return train_set, test_set
