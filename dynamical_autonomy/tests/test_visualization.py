"""
Tests for the visualization functions.
"""

import os
import numpy as np
import pandas as pd
import pytest
import tempfile
import matplotlib.pyplot as plt
from dynamical_autonomy.visualization import (
    plot_time_series,
    plot_granger_causality_matrix,
    plot_granger_causality_network
)


@pytest.fixture
def sample_time_series():
    """Create sample time series data for testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 3
    
    # Create time index
    time_index = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')
    
    # Create random data
    data = np.random.randn(n_samples, n_features)
    
    # Create DataFrame with time index
    df = pd.DataFrame(
        data, 
        index=time_index,
        columns=[f'var_{i}' for i in range(n_features)]
    )
    
    return df


@pytest.fixture
def sample_fitted_data(sample_time_series):
    """Create sample fitted data for testing."""
    # Create a simple fitted dataset (with some noise)
    np.random.seed(43)
    fitted_data = sample_time_series.copy()
    noise = np.random.randn(*fitted_data.shape) * 0.1
    fitted_data = fitted_data + noise
    
    return fitted_data


@pytest.fixture
def sample_causality_matrix():
    """Create sample Granger causality matrix for testing."""
    np.random.seed(44)
    n_features = 4
    
    # Create a random causality matrix
    F = np.random.rand(n_features, n_features)
    
    # Set diagonal to zero (no self-causality)
    np.fill_diagonal(F, 0)
    
    return F


def test_plot_time_series(sample_time_series, sample_fitted_data):
    """Test that plot_time_series creates a figure without errors."""
    # Test with only data
    fig = plot_time_series(sample_time_series)
    assert isinstance(fig, plt.Figure), "Should return a matplotlib Figure"
    plt.close(fig)
    
    # Test with data and fitted data
    fig = plot_time_series(sample_time_series, sample_fitted_data)
    assert isinstance(fig, plt.Figure), "Should return a matplotlib Figure"
    plt.close(fig)
    
    # Test with custom title
    title = "Custom Time Series Plot"
    fig = plot_time_series(sample_time_series, title=title)
    assert isinstance(fig, plt.Figure), "Should return a matplotlib Figure"
    assert fig.axes[0].get_title() == title, "Title should be set correctly"
    plt.close(fig)


def test_plot_granger_causality_matrix(sample_causality_matrix):
    """Test that plot_granger_causality_matrix creates a figure without errors."""
    # Test with default parameters
    fig = plot_granger_causality_matrix(sample_causality_matrix)
    assert isinstance(fig, plt.Figure), "Should return a matplotlib Figure"
    plt.close(fig)
    
    # Test with custom labels
    labels = [f"Variable {i}" for i in range(sample_causality_matrix.shape[0])]
    fig = plot_granger_causality_matrix(sample_causality_matrix, labels=labels)
    assert isinstance(fig, plt.Figure), "Should return a matplotlib Figure"
    plt.close(fig)
    
    # Test with custom colormap and title
    cmap = "Blues"
    title = "Custom Causality Matrix"
    fig = plot_granger_causality_matrix(sample_causality_matrix, cmap=cmap, title=title)
    assert isinstance(fig, plt.Figure), "Should return a matplotlib Figure"
    assert fig.axes[0].get_title() == title, "Title should be set correctly"
    plt.close(fig)


def test_plot_granger_causality_network(sample_causality_matrix):
    """Test that plot_granger_causality_network creates a figure without errors."""
    # Test with default parameters
    fig = plot_granger_causality_network(sample_causality_matrix)
    assert isinstance(fig, plt.Figure), "Should return a matplotlib Figure"
    plt.close(fig)
    
    # Test with custom labels and threshold
    labels = [f"Variable {i}" for i in range(sample_causality_matrix.shape[0])]
    threshold = 0.3
    fig = plot_granger_causality_network(sample_causality_matrix, labels=labels, threshold=threshold)
    assert isinstance(fig, plt.Figure), "Should return a matplotlib Figure"
    plt.close(fig)
    
    # Test with HTML output
    with tempfile.TemporaryDirectory() as temp_dir:
        html_output = os.path.join(temp_dir, "network.html")
        fig = plot_granger_causality_network(
            sample_causality_matrix, 
            html_output=html_output
        )
        assert isinstance(fig, plt.Figure), "Should return a matplotlib Figure"
        assert os.path.exists(html_output), "HTML file should be created"
        plt.close(fig)


def test_visualization_with_numpy_arrays():
    """Test that visualization functions work with NumPy arrays."""
    np.random.seed(45)
    
    # Create NumPy array time series
    data = np.random.randn(100, 3)
    
    # Test plot_time_series with NumPy array
    fig = plot_time_series(data)
    assert isinstance(fig, plt.Figure), "Should return a matplotlib Figure"
    plt.close(fig)
    
    # Create causality matrix
    F = np.random.rand(4, 4)
    np.fill_diagonal(F, 0)
    
    # Test plot_granger_causality_matrix with NumPy array
    fig = plot_granger_causality_matrix(F)
    assert isinstance(fig, plt.Figure), "Should return a matplotlib Figure"
    plt.close(fig)
    
    # Test plot_granger_causality_network with NumPy array
    fig = plot_granger_causality_network(F)
    assert isinstance(fig, plt.Figure), "Should return a matplotlib Figure"
    plt.close(fig)


if __name__ == "__main__":
    pytest.main(["-v", __file__]) 