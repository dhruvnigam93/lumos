import numpy as np
import pandas as pd


class MinMaxScaler:
    """Min-max scaler that stores parameters for reproducible transforms."""

    def __init__(self):
        self.mins = None
        self.maxs = None
        self.fitted = False

    def fit(self, data: pd.DataFrame | np.ndarray) -> "MinMaxScaler":
        arr = np.asarray(data, dtype=np.float64)
        self.mins = arr.min(axis=0)
        self.maxs = arr.max(axis=0)
        self.fitted = True
        return self

    def transform(self, data: pd.DataFrame | np.ndarray) -> np.ndarray:
        arr = np.asarray(data, dtype=np.float64)
        denom = self.maxs - self.mins
        denom[denom == 0] = 1.0
        return (arr - self.mins) / denom

    def fit_transform(self, data: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.fit(data).transform(data)

    def get_params(self) -> dict:
        return {"mins": self.mins.tolist(), "maxs": self.maxs.tolist()}

    @classmethod
    def from_params(cls, params: dict) -> "MinMaxScaler":
        scaler = cls()
        scaler.mins = np.array(params["mins"])
        scaler.maxs = np.array(params["maxs"])
        scaler.fitted = True
        return scaler


class LogMinMaxScaler:
    """Log-transform followed by min-max scaling. Useful for skewed distributions."""

    def __init__(self):
        self.inner = MinMaxScaler()

    def fit(self, data: pd.DataFrame | np.ndarray) -> "LogMinMaxScaler":
        arr = np.asarray(data, dtype=np.float64)
        self.inner.fit(np.log1p(np.abs(arr)))
        return self

    def transform(self, data: pd.DataFrame | np.ndarray) -> np.ndarray:
        arr = np.asarray(data, dtype=np.float64)
        return self.inner.transform(np.log1p(np.abs(arr)))

    def fit_transform(self, data: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.fit(data).transform(data)

    def get_params(self) -> dict:
        return self.inner.get_params()

    @classmethod
    def from_params(cls, params: dict) -> "LogMinMaxScaler":
        scaler = cls()
        scaler.inner = MinMaxScaler.from_params(params)
        return scaler


def build_sequences(
    df: pd.DataFrame,
    user_col: str,
    day_col: str,
    feature_cols: list[str],
    lookback: int,
    pad_value: float = 0.0,
) -> dict[int, np.ndarray]:
    """Build fixed-length sequences per user from a flat DataFrame.

    Args:
        df: Input DataFrame sorted by user and day.
        user_col: Column identifying users.
        day_col: Column identifying time steps.
        feature_cols: Feature columns to include in sequences.
        lookback: Desired sequence length (left-padded if shorter).
        pad_value: Value used for padding.

    Returns:
        Dictionary mapping user_id -> np.ndarray of shape [lookback, n_features].
    """
    sequences = {}
    for uid, group in df.groupby(user_col):
        group = group.sort_values(day_col)
        values = group[feature_cols].values.astype(np.float32)
        if len(values) >= lookback:
            sequences[uid] = values[-lookback:]
        else:
            pad = np.full((lookback - len(values), len(feature_cols)), pad_value, dtype=np.float32)
            sequences[uid] = np.concatenate([pad, values], axis=0)
    return sequences
