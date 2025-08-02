"""
Module for preprocessing time series data for change point analysis.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional


def calculate_returns(series: pd.Series, method: str = 'log') -> pd.Series:
    """
    Calculate returns from a price series.

    Args:
        series: Pandas Series of prices.
        method: Method to calculate returns ('log' or 'simple').

    Returns:
        pd.Series: Series of returns.
    """
    if method == 'log':
        return np.log(series).diff().dropna()
    elif method == 'simple':
        return series.pct_change().dropna()
    else:
        raise ValueError("method must be either 'log' or 'simple'")


def prepare_time_series(
    df: pd.DataFrame,
    date_col: str = 'Date',
    value_col: str = 'Price',
    freq: str = 'D',
    fill_method: Optional[str] = 'ffill'
) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    """
    Prepare time series data for change point analysis.

    Args:
        df: DataFrame containing the time series data.
        date_col: Name of the date column.
        value_col: Name of the value column.
        freq: Frequency for date range ('D' for daily, 'M' for monthly, etc.).
        fill_method: Method to fill missing values ('ffill', 'bfill', 'interpolate', or None).

    Returns:
        Tuple containing:
            - DatetimeIndex of the time series
            - Numpy array of the time series values
    """
    # Ensure date is the index
    ts = df.set_index(date_col)[value_col].copy()
    
    # Create complete date range
    date_range = pd.date_range(start=ts.index.min(), end=ts.index.max(), freq=freq)
    ts = ts.reindex(date_range)
    
    # Handle missing values
    if fill_method == 'ffill':
        ts = ts.ffill()
    elif fill_method == 'bfill':
        ts = ts.bfill()
    elif fill_method == 'interpolate':
        ts = ts.interpolate(method='linear')
    elif fill_method is not None:
        raise ValueError(f"Unsupported fill_method: {fill_method}")
    
    return ts.index, ts.values


def create_rolling_features(
    series: pd.Series,
    windows: list = [7, 30, 90],
    stats: list = ['mean', 'std']
) -> pd.DataFrame:
    """
    Create rolling window features from a time series.

    Args:
        series: Input time series.
        windows: List of window sizes for rolling calculations.
        stats: List of statistics to calculate for each window.

    Returns:
        pd.DataFrame: DataFrame containing the rolling features.
    """
    features = {}
        
    for window in windows:
        for stat in stats:
            if stat == 'mean':
                features[f'rolling_{window}d_mean'] = series.rolling(window=window).mean()
            elif stat == 'std':
                features[f'rolling_{window}d_std'] = series.rolling(window=window).std()
            elif stat == 'min':
                features[f'rolling_{window}d_min'] = series.rolling(window=window).min()
            elif stat == 'max':
                features[f'rolling_{window}d_max'] = series.rolling(window=window).max()
    
    return pd.DataFrame(features).dropna()
