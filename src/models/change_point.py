"""
Bayesian Change Point Detection for time series data using PyMC3.
"""
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pymc3 as pm
import theano.tensor as tt
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
from scipy import stats


class BayesianChangePoint:
    """
    Bayesian Change Point Detection for time series data.
    
    This class implements a Bayesian approach to detect change points in time series
    data, allowing for changes in mean, volatility, or both.
    """
    
    def __init__(
        self,
        n_changepoints: int = 1,
        change_type: str = 'mean',
        model_type: str = 'normal',
        **kwargs
    ):
        """
        Initialize the Bayesian Change Point model.
        
        Args:
            n_changepoints: Number of change points to detect.
            change_type: Type of change to detect ('mean', 'volatility', or 'both').
            model_type: Type of likelihood model ('normal', 'studentt', 'laplace').
            **kwargs: Additional arguments for the model.
        ""
        self.n_changepoints = n_changepoints
        self.change_type = change_type.lower()
        self.model_type = model_type.lower()
        self.model = None
        self.trace = None
        self.training_data = None
        self._validate_parameters()
        
    def _validate_parameters(self):
        """Validate model parameters."""
        if self.change_type not in ['mean', 'volatility', 'both']:
            raise ValueError("change_type must be 'mean', 'volatility', or 'both'")
            
        if self.model_type not in ['normal', 'studentt', 'laplace']:
            raise ValueError("model_type must be 'normal', 'studentt', or 'laplace'")
            
        if not isinstance(self.n_changepoints, int) or self.n_changepoints < 1:
            raise ValueError("n_changepoints must be a positive integer")
    
    def _build_model(self, data: np.ndarray) -> pm.Model:
        """
        Build the PyMC3 model for change point detection.
        
        Args:
            data: 1D array of time series data.
                
        Returns:
            pm.Model: The PyMC3 model.
        """
        n = len(data)
        
        with pm.Model() as model:
            # Uniform prior on change point positions
            tau = pm.DiscreteUniform(
                'tau', 
                lower=1, 
                upper=n-1, 
                shape=self.n_changepoints,
                testval=np.linspace(1, n-1, self.n_changepoints+2)[1:-1].astype(int)
            )
            
            # Sort change points to ensure they're in order
            tau_sorted = pm.Deterministic('tau_sorted', tt.sort(tau))
            
            # Priors for mean and standard deviation
            if self.change_type in ['mean', 'both']:
                mu = pm.Normal('mu', mu=0, sigma=10, shape=self.n_changepoints + 1)
            else:
                mu = pm.Normal('mu', mu=0, sigma=10)
                
            if self.change_type in ['volatility', 'both']:
                sigma = pm.HalfNormal('sigma', sigma=1, shape=self.n_changepoints + 1)
            else:
                sigma = pm.HalfNormal('sigma', sigma=1)
            
            # Piecewise constant mean and sigma
            if self.change_type == 'mean':
                mu_piecewise = mu[0] * tt.ones(n)
                for i in range(self.n_changepoints):
                    mu_piecewise = tt.set_subtensor(
                        mu_piecewise[tau_sorted[i]:], 
                        mu[i+1]
                    )
                sigma_piecewise = sigma * tt.ones(n)
                
            elif self.change_type == 'volatility':
                mu_piecewise = mu * tt.ones(n)
                sigma_piecewise = sigma[0] * tt.ones(n)
                for i in range(self.n_changepoints):
                    sigma_piecewise = tt.set_subtensor(
                        sigma_piecewise[tau_sorted[i]:], 
                        sigma[i+1]
                    )
                    
            else:  # both mean and volatility change
                mu_piecewise = mu[0] * tt.ones(n)
                sigma_piecewise = sigma[0] * tt.ones(n)
                for i in range(self.n_changepoints):
                    mu_piecewise = tt.set_subtensor(
                        mu_piecewise[tau_sorted[i]:], 
                        mu[i+1]
                    )
                    sigma_piecewise = tt.set_subtensor(
                        sigma_piecewise[tau_sorted[i]:], 
                        sigma[i+1]
                    )
            
            # Likelihood
            if self.model_type == 'normal':
                obs = pm.Normal(
                    'obs', 
                    mu=mu_piecewise, 
                    sigma=sigma_piecewise, 
                    observed=data
                )
            elif self.model_type == 'studentt':
                nu = pm.Exponential('nu', 1/10.)
                obs = pm.StudentT(
                    'obs',
                    nu=nu,
                    mu=mu_piecewise,
                    sigma=sigma_piecewise,
                    observed=data
                )
            else:  # laplace
                obs = pm.Laplace(
                    'obs',
                    mu=mu_piecewise,
                    b=sigma_piecewise,
                    observed=data
                )
            
            # Store deterministic variables
            pm.Deterministic('mu_piecewise', mu_piecewise)
            pm.Deterministic('sigma_piecewise', sigma_piecewise)
            
        return model
    
    def fit(
        self, 
        data: Union[np.ndarray, pd.Series],
        tune: int = 2000,
        draws: int = 2000,
        chains: int = 2,
        **kwargs
    ) -> az.InferenceData:
        """
        Fit the Bayesian Change Point model to the data.
        
        Args:
            data: 1D array or pandas Series of time series data.
            tune: Number of tuning samples.
            draws: Number of posterior samples to draw.
            chains: Number of MCMC chains to run.
            **kwargs: Additional arguments passed to pm.sample().
            
        Returns:
            arviz.InferenceData: The trace and observed data.
        """
        if isinstance(data, pd.Series):
            data = data.values
            
        if not isinstance(data, np.ndarray) or data.ndim != 1:
            raise ValueError("data must be a 1D array or pandas Series")
            
        self.training_data = data
        self.model = self._build_model(data)
        
        with self.model:
            self.trace = pm.sample(
                tune=tune,
                draws=draws,
                chains=chains,
                return_inferencedata=True,
                **kwargs
            )
            
        return self.trace
    
    def plot_changepoints(self, dates: Optional[np.ndarray] = None, **kwargs):
        """
        Plot the time series with change points.
        
        Args:
            dates: Optional array of dates corresponding to the time series data.
            **kwargs: Additional arguments passed to the plotting function.
            
        Returns:
            matplotlib.figure.Figure: The figure object.
        """
        if self.trace is None or self.training_data is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
            
        # Get posterior samples of change point locations
        tau_samples = self.trace.posterior['tau_sorted'].values
        
        # Flatten samples from all chains
        tau_flat = tau_samples.reshape(-1, tau_samples.shape[-1])
        
        # Calculate posterior probabilities of change points
        n_samples = len(tau_flat)
        n_points = len(self.training_data)
        
        # Create a grid of points and count how many times each point is a change point
        grid = np.arange(n_points)
        cp_probs = np.zeros(n_points)
        
        for i in range(self.n_changepoints):
            cp_probs += np.bincount(
                tau_flat[:, i], 
                minlength=n_points
            )
            
        cp_probs = cp_probs / n_samples
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot the time series
        if dates is not None:
            x = dates
        else:
            x = np.arange(len(self.training_data))
            
        ax1.plot(x, self.training_data, 'k-', alpha=0.8, label='Observed')
        
        # Add mean posterior change points
        tau_mean = np.mean(tau_flat, axis=0).astype(int)
        
        for i, tau in enumerate(tau_mean):
            if i == 0:
                label = 'Change Points'
            else:
                label = None
                
            ax1.axvline(
                x[tau], 
                color='r', 
                linestyle='--', 
                alpha=0.7,
                label=label
            )
        
        ax1.set_ylabel('Value')
        ax1.set_title('Time Series with Change Points')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot change point probabilities
        ax2.bar(x, cp_probs, width=1.0, alpha=0.7, color='b')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Probability')
        ax2.set_title('Change Point Probability')
        ax2.set_ylim(0, 1.1)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def get_changepoint_summary(self) -> pd.DataFrame:
        """
        Get a summary of the detected change points.
        
        Returns:
            pd.DataFrame: Summary statistics for each change point.
        """
        if self.trace is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
            
        # Get posterior samples of change point locations
        tau_samples = self.trace.posterior['tau_sorted'].values
        
        # Flatten samples from all chains
        tau_flat = tau_samples.reshape(-1, tau_samples.shape[-1])
        
        # Calculate summary statistics
        summary = []
        
        for i in range(self.n_changepoints):
            cp_samples = tau_flat[:, i]
            
            summary.append({
                'changepoint': i + 1,
                'mean': np.mean(cp_samples),
                'std': np.std(cp_samples),
                'hdi_3%': np.percentile(cp_samples, 1.5),
                'hdi_97%': np.percentile(cp_samples, 98.5),
                'probability': np.mean(cp_samples > 0)  # Probability of being a change point
            })
            
        return pd.DataFrame(summary)
    
    def get_parameter_summary(self) -> pd.DataFrame:
        """
        Get a summary of the model parameters.
        
        Returns:
            pd.DataFrame: Summary statistics for model parameters.
        """
        if self.trace is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
            
        return az.summary(self.trace, var_names=['mu', 'sigma'])


def detect_changepoints(
    data: Union[np.ndarray, pd.Series],
    n_changepoints: int = 1,
    change_type: str = 'both',
    model_type: str = 'normal',
    **kwargs
) -> Tuple[az.InferenceData, pd.DataFrame]:
    """
    Convenience function to detect change points in time series data.
    
    Args:
        data: 1D array or pandas Series of time series data.
        n_changepoints: Number of change points to detect.
        change_type: Type of change to detect ('mean', 'volatility', or 'both').
        model_type: Type of likelihood model ('normal', 'studentt', 'laplace').
        **kwargs: Additional arguments passed to the model's fit method.
        
    Returns:
        tuple: (trace, summary) where trace is the arviz.InferenceData and
            summary is a DataFrame with change point statistics.
    """
    model = BayesianChangePoint(
        n_changepoints=n_changepoints,
        change_type=change_type,
        model_type=model_type
    )
    
    trace = model.fit(data, **kwargs)
    summary = model.get_changepoint_summary()
    
    return trace, summary
