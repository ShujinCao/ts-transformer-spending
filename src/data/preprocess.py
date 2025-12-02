import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_multicountry_data(path: str) -> pd.DataFrame:
    """
    Load long-format health expenditure data:

        country, iso3, year, value

    and sort by (iso3, year).
    """
    df = pd.read_csv(path)
    df = df.sort_values(["iso3", "year"]).reset_index(drop=True)
    return df


def make_country_groups(df: pd.DataFrame):
    """
    Returns a dict: iso3 -> dataframe for that country.
    """
    groups = {iso: g.reset_index(drop=True) for iso, g in df.groupby("iso3")}
    return groups


def interpolate_country(df: pd.DataFrame) -> pd.DataFrame:
    """
    Linear interpolation of missing 'value' for a single country.
    Assumes rows are sorted by year.
    """
    df = df.copy()
    df["value"] = df["value"].interpolate(method="linear")
    return df


def create_multiseries_windows(country_df: pd.DataFrame,
                               input_len: int,
                               forecast_horizon: int):
    """
    Given a dataframe for one country (year, value),
    returns (X, y) numpy arrays of sliding windows.

    X: (num_windows, input_len)
    y: (num_windows, forecast_horizon)
    """
    values = country_df["value"].values.astype(float)

    X, y = [], []
    L = len(values)

    for i in range(L - input_len - forecast_horizon + 1):
        X.append(values[i: i + input_len])
        y.append(values[i + input_len: i + input_len + forecast_horizon])

    return np.array(X), np.array(y)


def build_multicountry_dataset(df: pd.DataFrame,
                               input_len: int,
                               forecast_horizon: int):
    """
    Build dataset across all countries.

    Returns:
        X_all: (N, input_len)
        y_all: (N, forecast_horizon)
        country_ids: (N,) array of iso3 codes for each sample
    """
    X_all, y_all, cid_all = [], [], []

    countries = make_country_groups(df)

    for cid, subdf in countries.items():
        subdf = interpolate_country(subdf)

        if len(subdf) < input_len + forecast_horizon:
            continue

        X_c, y_c = create_multiseries_windows(
            subdf, input_len, forecast_horizon
        )

        X_all.append(X_c)
        y_all.append(y_c)
        cid_all.extend([cid] * len(X_c))

    if not X_all:
        raise ValueError("No valid windows generated. Check data length/config.")

    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    country_ids = np.array(cid_all)

    return X_all, y_all, country_ids


def scale_global(X_train: np.ndarray, X_val: np.ndarray):
    """
    Scale input windows globally using StandardScaler.

    X_train, X_val: (N, input_len)
    """
    scaler = StandardScaler()
    train_flat = X_train.reshape(-1, 1)
    val_flat = X_val.reshape(-1, 1)

    train_scaled = scaler.fit_transform(train_flat).reshape(X_train.shape)
    val_scaled = scaler.transform(val_flat).reshape(X_val.shape)

    return train_scaled, val_scaled, scaler

