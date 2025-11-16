from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

def preprocess(data: pd.DataFrame):
    # Separate target if it exists
    if 'target' in data.columns:
        target = data['target']
        data = data.drop(columns=['target'])
    else:
        target = None

    # Select numeric columns
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    numerical_data = data[numeric_cols]

    # Scale numeric data
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(numerical_data)

    # Convert back to DataFrame
    scaled_df = pd.DataFrame(scaled, columns=numeric_cols)
    
    return scaled_df, target


def split_data(file_path: str, test_size: float = 0.1, random_state: int = 42):
    raw_data = load_data(file_path)
    scaled_data, target = preprocess(raw_data)

    train_set, test_set = train_test_split(
        scaled_data, test_size=test_size, random_state=random_state
    )

    return train_set, test_set
