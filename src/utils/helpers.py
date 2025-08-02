"""
Helper functions for the change point analysis project.
"""
from typing import Union, List, Dict, Any, Optional
import os
import json
import yaml
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_directory_structure(base_dir: str) -> None:
    """
    Create the project directory structure if it doesn't exist.
    
    Args:
        base_dir: Base directory for the project.
    """
    dirs = [
        'data/raw',
        'data/processed',
        'data/external',
        'notebooks',
        'src',
        'dashboard/backend',
        'dashboard/frontend/src/components',
        'dashboard/frontend/src/pages',
        'dashboard/frontend/src/services',
        'docs',
        'tests'
    ]
    
    for dir_path in dirs:
        full_path = os.path.join(base_dir, dir_path)
        os.makedirs(full_path, exist_ok=True)
        
    logger.info("Project directory structure created")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML or JSON file.
    
    Args:
        config_path: Path to the configuration file.
        
    Returns:
        dict: Configuration dictionary.
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
            config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            config = json.load(f)
        else:
            raise ValueError("Unsupported config file format. Use .yaml, .yml, or .json")
    
    return config


def save_pickle(obj: Any, filepath: Union[str, Path]) -> None:
    """
    Save an object to a pickle file.
    
    Args:
        obj: Object to save.
        filepath: Path to save the object to.
    """
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    logger.info(f"Object saved to {filepath}")


def load_pickle(filepath: Union[str, Path]) -> Any:
    """
    Load an object from a pickle file.
    
    Args:
        filepath: Path to the pickle file.
        
    Returns:
        The unpickled object.
    """
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    logger.info(f"Object loaded from {filepath}")
    return obj


def save_dataframe(
    df: pd.DataFrame, 
    filepath: Union[str, Path], 
    index: bool = False,
    **kwargs
) -> None:
    """
    Save a pandas DataFrame to a file.
    
    Args:
        df: DataFrame to save.
        filepath: Path to save the DataFrame to.
        index: Whether to write row names.
        **kwargs: Additional arguments passed to the pandas to_* method.
    """
    filepath = Path(filepath)
    
    if filepath.suffix.lower() == '.csv':
        df.to_csv(filepath, index=index, **kwargs)
    elif filepath.suffix.lower() == '.parquet':
        df.to_parquet(filepath, index=index, **kwargs)
    elif filepath.suffix.lower() == '.feather':
        df.to_feather(filepath, **kwargs)
    elif filepath.suffix.lower() == '.xlsx':
        df.to_excel(filepath, index=index, **kwargs)
    else:
        raise ValueError("Unsupported file format. Use .csv, .parquet, .feather, or .xlsx")
    
    logger.info(f"DataFrame saved to {filepath}")


def load_dataframe(
    filepath: Union[str, Path],
    **kwargs
) -> pd.DataFrame:
    """
    Load a pandas DataFrame from a file.
    
    Args:
        filepath: Path to the file to load.
        **kwargs: Additional arguments passed to the pandas read_* function.
        
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if filepath.suffix.lower() == '.csv':
        df = pd.read_csv(filepath, **kwargs)
    elif filepath.suffix.lower() == '.parquet':
        df = pd.read_parquet(filepath, **kwargs)
    elif filepath.suffix.lower() == '.feather':
        df = pd.read_feather(filepath, **kwargs)
    elif filepath.suffix.lower() == '.xlsx':
        df = pd.read_excel(filepath, **kwargs)
    else:
        raise ValueError("Unsupported file format. Use .csv, .parquet, .feather, or .xlsx")
    
    logger.info(f"DataFrame loaded from {filepath}")
    return df


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path: Path to the project root directory.
    """
    return Path(__file__).parent.parent.parent


def setup_logging(
    log_file: Optional[Union[str, Path]] = None,
    log_level: int = logging.INFO
) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_file: Optional path to a log file. If None, logs will only be printed to console.
        log_level: Logging level (default: logging.INFO).
    """
    handlers = [logging.StreamHandler()]
    
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    logger.info("Logging configured")


def time_series_train_test_split(
    X: np.ndarray,
    y: np.ndarray = None,
    test_size: float = 0.2,
    shuffle: bool = False
) -> tuple:
    """
    Split time series data into training and testing sets.
    
    Args:
        X: Feature matrix.
        y: Target vector (optional).
        test_size: Proportion of the dataset to include in the test split.
        shuffle: Whether to shuffle the data before splitting.
        
    Returns:
        tuple: X_train, X_test, y_train, y_test (if y is provided)
               or X_train, X_test (if y is None)
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    n_train = n_samples - n_test
    
    if shuffle:
        indices = np.random.permutation(n_samples)
        X = X[indices]
        if y is not None:
            y = y[indices]
    
    X_train, X_test = X[:n_train], X[n_train:]
    
    if y is not None:
        y_train, y_test = y[:n_train], y[n_train:]
        return X_train, X_test, y_train, y_test
    else:
        return X_train, X_test


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate regression metrics.
    
    Args:
        y_true: True values.
        y_pred: Predicted values.
        
    Returns:
        dict: Dictionary of metrics.
    """
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        mean_absolute_percentage_error
    )
    
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred),
        'mape': mean_absolute_percentage_error(y_true, y_pred) * 100  # as percentage
    }
    
    return metrics
