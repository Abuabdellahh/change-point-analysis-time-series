"""
Module for loading and validating the Brent oil price dataset.
"""
import pandas as pd
from pathlib import Path


def load_brent_data(filepath: str) -> pd.DataFrame:
    """
    Load and validate the Brent oil price dataset.

    Args:
        filepath: Path to the CSV file containing Brent oil price data.

    Returns:
        pd.DataFrame: DataFrame containing the loaded data with proper data types.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the required columns are missing from the dataset.
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    # Load the data
    df = pd.read_csv(filepath)
    
    # Validate required columns
    required_columns = {'Date', 'Price'}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by date to ensure chronological order
    df = df.sort_values('Date').reset_index(drop=True)
    
    return df


def load_event_data(filepath: str) -> pd.DataFrame:
    """
    Load and validate the event dataset.

    Args:
        filepath: Path to the CSV file containing event data.

    Returns:
        pd.DataFrame: DataFrame containing the loaded event data.
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Event data file not found: {filepath}")

    # Load the data
    df = pd.read_csv(filepath)
    
    # Validate required columns
    required_columns = {'event_date', 'event_type', 'description', 'impact'}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns in event data: {missing_columns}")
    
    # Convert event_date to datetime
    df['event_date'] = pd.to_datetime(df['event_date'])
    
    return df
