"""
Visualization functions for time series analysis and change point detection.
"""
from typing import List, Optional, Union, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from matplotlib.dates import DateFormatter


def plot_series(
    dates: Union[pd.DatetimeIndex, np.ndarray],
    values: np.ndarray,
    title: str = "Time Series",
    xlabel: str = "Date",
    ylabel: str = "Value",
    figsize: Tuple[int, int] = (12, 6),
    **kwargs
) -> plt.Figure:
    """
    Plot a time series with proper date formatting.
    
    Args:
        dates: Array of dates or datetime indices.
        values: Array of values to plot.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        figsize: Figure size (width, height).
        **kwargs: Additional arguments passed to plt.plot().
        
    Returns:
        matplotlib.figure.Figure: The figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the time series
    ax.plot(dates, values, **kwargs)
    
    # Format x-axis for dates if dates are datetime objects
    if isinstance(dates, (pd.DatetimeIndex, pd.Series)):
        ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate()
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_changepoints(
    dates: Union[pd.DatetimeIndex, np.ndarray],
    values: np.ndarray,
    changepoints: Union[List[int], np.ndarray],
    changepoint_probs: Optional[np.ndarray] = None,
    title: str = "Change Point Detection",
    xlabel: str = "Date",
    ylabel: str = "Value",
    figsize: Tuple[int, int] = (12, 8),
    **kwargs
) -> plt.Figure:
    """
    Plot time series with detected change points.
    
    Args:
        dates: Array of dates or datetime indices.
        values: Array of time series values.
        changepoints: Indices of detected change points.
        changepoint_probs: Optional array of change point probabilities.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        figsize: Figure size (width, height).
        **kwargs: Additional arguments passed to plt.plot().
        
    Returns:
        matplotlib.figure.Figure: The figure object.
    """
    if changepoint_probs is not None:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, 
            figsize=figsize, 
            gridspec_kw={'height_ratios': [3, 1]},
            sharex=True
        )
    else:
        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = None
    
    # Plot the time series
    ax1.plot(dates, values, 'k-', alpha=0.8, label='Observed', **kwargs)
    
    # Add change points
    for i, cp in enumerate(changepoints):
        if i == 0:
            label = 'Change Points'
        else:
            label = None
        
        ax1.axvline(
            dates[cp], 
            color='r', 
            linestyle='--', 
            alpha=0.7,
            label=label
        )
    
    ax1.set_title(title)
    ax1.set_ylabel(ylabel)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot change point probabilities if provided
    if changepoint_probs is not None:
        ax2.bar(dates, changepoint_probs, width=1.0, alpha=0.7, color='b')
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel('Probability')
        ax2.set_ylim(0, 1.1)
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis for dates if dates are datetime objects
        if isinstance(dates, (pd.DatetimeIndex, pd.Series)):
            ax2.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    else:
        ax1.set_xlabel(xlabel)
        if isinstance(dates, (pd.DatetimeIndex, pd.Series)):
            ax1.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    
    plt.tight_layout()
    return fig


def plot_posterior(
    trace: az.InferenceData,
    var_names: Union[str, List[str]] = None,
    kind: str = "hist",
    figsize: Tuple[int, int] = (12, 6),
    **kwargs
) -> plt.Figure:
    """
    Plot posterior distributions of model parameters.
    
    Args:
        trace: ArviZ InferenceData object containing the trace.
        var_names: List of variable names to plot. If None, plot all variables.
        kind: Type of plot ('hist', 'kde', or 'trace').
        figsize: Figure size (width, height).
        **kwargs: Additional arguments passed to arviz.plot_posterior().
        
    Returns:
        matplotlib.figure.Figure: The figure object.
    """
    if kind == 'trace':
        axes = az.plot_trace(
            trace,
            var_names=var_names,
            figsize=figsize,
            **kwargs
        )
    else:
        axes = az.plot_posterior(
            trace,
            var_names=var_names,
            kind=kind,
            figsize=figsize,
            **kwargs
        )
    
    plt.tight_layout()
    return plt.gcf()


def plot_rolling_statistics(
    dates: Union[pd.DatetimeIndex, np.ndarray],
    series: np.ndarray,
    windows: List[int] = [30, 90, 365],
    title: str = "Rolling Statistics",
    figsize: Tuple[int, int] = (14, 10),
    **kwargs
) -> plt.Figure:
    """
    Plot rolling mean and standard deviation of a time series.
    
    Args:
        dates: Array of dates or datetime indices.
        series: Time series data.
        windows: List of window sizes for rolling calculations.
        title: Plot title.
        figsize: Figure size (width, height).
        **kwargs: Additional arguments passed to plt.plot().
        
    Returns:
        matplotlib.figure.Figure: The figure object.
    """
    n_windows = len(windows)
    fig, axes = plt.subplots(n_windows * 2, 1, figsize=figsize, sharex=True)
    
    if n_windows == 1:
        axes = [axes]
    
    for i, window in enumerate(windows):
        # Calculate rolling statistics
        rolling_mean = pd.Series(series).rolling(window=window).mean()
        rolling_std = pd.Series(series).rolling(window=window).std()
        
        # Plot rolling mean
        axes[2*i].plot(dates, rolling_mean, 'b-', label=f'{window}-day Rolling Mean', **kwargs)
        axes[2*i].set_ylabel('Mean')
        axes[2*i].set_title(f'{window}-day Rolling Statistics')
        axes[2*i].grid(True, alpha=0.3)
        
        # Plot rolling standard deviation
        axes[2*i + 1].plot(dates, rolling_std, 'r-', label=f'{window}-day Rolling Std', **kwargs)
        axes[2*i + 1].set_ylabel('Std Dev')
        axes[2*i + 1].grid(True, alpha=0.3)
    
    # Format x-axis for dates if dates are datetime objects
    if isinstance(dates, (pd.DatetimeIndex, pd.Series)):
        plt.gca().xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
        plt.gcf().autofmt_xdate()
    
    plt.xlabel('Date')
    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    
    return fig


def plot_autocorrelation(
    series: np.ndarray,
    lags: int = 50,
    title: str = "Autocorrelation and Partial Autocorrelation",
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """
    Plot autocorrelation and partial autocorrelation functions.
    
    Args:
        series: Time series data.
        lags: Number of lags to plot.
        title: Plot title.
        figsize: Figure size (width, height).
        
    Returns:
        matplotlib.figure.Figure: The figure object.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot ACF
    pd.plotting.autocorrelation_plot(series, ax=ax1)
    ax1.set_title('Autocorrelation')
    
    # Plot PACF
    pd.plotting.autocorrelation_plot(
        series.diff().dropna() if pd.Series(series).isna().any() else series,
        ax=ax2
    )
    ax2.set_title('Partial Autocorrelation')
    
    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    
    return fig
