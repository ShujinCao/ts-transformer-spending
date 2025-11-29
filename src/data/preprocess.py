import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_health_data(path):
    """
    Load a CSV containing health expenditure time series.
    Expected format:
        year, value
        2000, 4500
        2001, 4600
        ...
    """
    df = pd.read_csv(path)
    df = df.sort_values("year")
    df = df.reset_index(drop=True)
    return df


def interpolate_missing(df, column="value"):
    """Linear interpolation for missing health expenditure years."""
    df[column] = df[column].interpolate(method="linear")
    return df


def create_sliding_windows(values, input_len=30, forecast_horizon=5):
    """
    Convert a long 1D time series into (X, y) windows.
    X: sequences of length input_len
    y: next forecast_horizon values
    """
    X, y = [], []
    for i in range(len(values) - input_len - forecast_horizon + 1):
        X.append(values[i : i + input_len])
        y.append(values[i + input_len : i + input_len + forecast_horizon])
    return np.array(X), np.array(y)


def scale_data(train, test):
    """Scale using StandardScaler and apply same scaler to test."""
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train.reshape(-1, 1)).reshape(train.shape)
    test_scaled = scaler.transform(test.reshape(-1, 1)).reshape(test.shape)
    return train_scaled, test_scaled, scaler

