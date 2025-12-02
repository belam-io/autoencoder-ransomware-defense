import os
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

def preprocess(data: pd.DataFrame, save_scaler=True):
    if 'timestamp' in data.columns:
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data['hour'] = data['timestamp'].dt.hour
        data['day'] = data['timestamp'].dt.day
        data['day_of_week'] = data['timestamp'].dt.dayofweek
        data['is_weekend'] = data['day_of_week'].isin([5,6]).astype(int)

    if 'TxnID' in data.columns:
        data.drop_duplicates(subset=['TxnID'], inplace=True)

    sort_cols = [c for c in ['Msisdn', 'timestamp'] if c in data.columns]
    if sort_cols:
        data = data.sort_values(sort_cols)

    if 'Msisdn' in data.columns and 'Amount' in data.columns:
        data['user_avg_amount'] = data.groupby('Msisdn')['Amount'].expanding().mean().reset_index(level=0, drop=True)
        data['user_std_amount'] = data.groupby('Msisdn')['Amount'].expanding().std().reset_index(level=0, drop=True).fillna(0)
        data['amount_dev'] = data['Amount'] - data['user_avg_amount']
        data['txn_user_count'] = data.groupby('Msisdn').cumcount() + 1
        data['txn_last_5'] = data.groupby('Msisdn')['Amount'].rolling(window=5, min_periods=1).count().reset_index(level=0, drop=True)

    if 'timestamp' in data.columns and 'Msisdn' in data.columns:
        data['time_since_last_txn'] = data.groupby('Msisdn')['timestamp'].diff().dt.total_seconds().fillna(999999)
        data['velocity'] = data['Amount'] / (data['time_since_last_txn'] + 1)

    if 'BillRefNumber' in data.columns and 'TxnID' in data.columns:
        data['txn_bill_ref_count'] = data.groupby('BillRefNumber')['TxnID'].transform('count')
    if 'BillRefNumber' in data.columns and 'Msisdn' in data.columns:
        data['txn_bill_ref_unique'] = data.groupby('BillRefNumber')['Msisdn'].transform('nunique')

    if 'StatusCode' in data.columns:
        data['is_success'] = (data['StatusCode'] == 0).astype(int)

    if 'BillRefNumber' in data.columns and 'is_success' in data.columns:
        data['bill_success_rate'] = data.groupby('BillRefNumber')['is_success'].transform('mean')

    if 'ShortCode' in data.columns:
        if 'TxnID' in data.columns:
            data['txn_shortcode_count'] = data.groupby('ShortCode')['TxnID'].transform('count')
        if 'Amount' in data.columns:
            data['txn_shortcode_avg'] = data.groupby('ShortCode')['Amount'].transform('mean')
            data['txn_shortcode_std'] = data.groupby('ShortCode')['Amount'].transform('std').fillna(0)
        if 'is_success' in data.columns:
            data['txn_success_ratio'] = data.groupby('ShortCode')['is_success'].transform('mean')

    onehot_cols = [c for c in ['Response', 'StatusCode'] if c in data.columns]
    if onehot_cols:
        data = pd.get_dummies(data, columns=onehot_cols, prefix=onehot_cols, drop_first=False)

    drop_cols = ['TxnID','Msisdn','BillRefNumber','timestamp']
    data.drop(columns=[c for c in drop_cols if c in data.columns], inplace=True, errors='ignore')

    categorical_cols = data.select_dtypes(include=['object']).columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = data[col].fillna('MISSING').astype(str)
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    numeric_cols = data.select_dtypes(include=['int64','float64']).columns
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data[numeric_cols])
    scaled_df = pd.DataFrame(scaled, columns=numeric_cols).fillna(0).replace([float('inf'), -float('inf')],0)

    if save_scaler:
        os.makedirs("models", exist_ok=True)
        joblib.dump(scaler, "models/scaler.pkl")
        joblib.dump(label_encoders, "models/encoders.pkl")

    return scaled_df, label_encoders, scaler

def split_data(file_path: str, test_size: float=0.1, random_state:int=42):
    raw_data = load_data(file_path)
    processed_data, encoders, scaler = preprocess(raw_data)
    X_train, X_test = train_test_split(processed_data, test_size=test_size, random_state=random_state)
    return X_train, X_test
