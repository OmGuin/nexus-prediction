import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def load_datax(csv_path, target_column, test_size=0.2, random_state=42, log_transform = False):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=[target_column])
    y = df[[target_column]]
    if log_transform:
        y = np.log1p(y)

    #x_scaler = MinMaxScaler()
    #_scaled = x_scaler.fit_transform(X)

    #y_scaler = MinMaxScaler()
    #y = y_scaler.fit_transform(y) if log_transform else y.values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_val, y_train, y_val, log_transform

def load_data(csv_path, target_column, test_size=0.2, random_state=42):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=[target_column]).values.astype(np.float32)
    y = df[[target_column]].values.astype(np.float32).reshape(-1, 1)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)