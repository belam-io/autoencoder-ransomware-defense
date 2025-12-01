import os
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def load_data(file_path: str) -> pd.DataFrame:
    """Load CSV data."""
    return pd.read_csv(file_path)

def preprocess(data: pd.DataFrame, save_scaler=True):
    """Preprocess transaction data for autoencoder training."""
    if 'timestamp' in data.columns:
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data['hour'] = data['timestamp'].dt.hour
        data['day'] = data['timestamp'].dt.day
        data['day_of_week'] = data['timestamp'].dt.dayofweek
        data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)

    data.drop_duplicates(subset=['TxnID'], inplace=True)
    data = data.sort_values(['Msisdn', 'timestamp'])

    # User-level statistics
    data['user_avg_amount'] = data.groupby('Msisdn')['Amount'].expanding().mean().reset_index(level=0, drop=True)
    data['user_std_amount'] = data.groupby('Msisdn')['Amount'].expanding().std().reset_index(level=0, drop=True)
    data['user_std_amount'].fillna(0, inplace=True)  # avoid NaNs
    data['amount_dev'] = data['Amount'] - data['user_avg_amount']

    # Transaction counts
    data['txn_user_count'] = data.groupby('Msisdn').cumcount() + 1
    window_size = 5
    data['txn_last_5'] = data.groupby('Msisdn')['Amount'].rolling(window=window_size, min_periods=1).count().reset_index(level=0, drop=True)

    # Time-based features
    data['time_since_last_txn'] = data.groupby('Msisdn')['timestamp'].diff().dt.total_seconds()
    data['time_since_last_txn'] = data['time_since_last_txn'].fillna(999999)
    data['velocity'] = data['Amount'] / (data['time_since_last_txn'] + 1)

    # Bill-level features
    data['txn_bill_ref_count'] = data.groupby('BillRefNumber')['TxnID'].transform('count')
    data['txn_bill_ref_unique'] = data.groupby('BillRefNumber')['Msisdn'].transform('nunique')
    data['is_success'] = (data['StatusCode'] == 0).astype(int)
    data['bill_success_rate'] = data.groupby('BillRefNumber')['is_success'].transform('mean')

    # ShortCode-level features
    data['txn_shortcode_count'] = data.groupby('ShortCode')['TxnID'].transform('count')
    data['txn_shortcode_avg'] = data.groupby('ShortCode')['Amount'].transform('mean')
    data['txn_shortcode_std'] = data.groupby('ShortCode')['Amount'].transform('std').fillna(0)
    data['txn_success_ratio'] = data.groupby('ShortCode')['is_success'].transform('mean')

    # One-hot encoding
    onehot_cols = ['Response', 'StatusCode']
    data = pd.get_dummies(data, columns=onehot_cols, prefix=onehot_cols, drop_first=False)

    # Label encoding for remaining categorical columns
    categorical_cols = data.select_dtypes(include=['object']).columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le

    # Scaling numeric columns
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data[numeric_cols])
    scaled_df = pd.DataFrame(scaled, columns=numeric_cols)

    # Handle NaNs/Infs after scaling
    scaled_df = scaled_df.fillna(0)
    scaled_df = scaled_df.replace([float('inf'), -float('inf')], 0)

    # Save scalers/encoders
    if save_scaler:
        os.makedirs("models", exist_ok=True)
        joblib.dump(scaler, "models/scaler.pkl")
        joblib.dump(label_encoders, "models/encoders.pkl")

    return scaled_df, label_encoders, scaler

def split_data(file_path: str, test_size: float = 0.1, random_state: int = 42):
    """Load, preprocess, and split data into train/test sets."""
    raw_data = load_data(file_path)
    processed_data, encoders, scaler = preprocess(raw_data)

    train_set, test_set = train_test_split(
        processed_data, test_size=test_size, random_state=random_state
    )

    return train_set, test_set
