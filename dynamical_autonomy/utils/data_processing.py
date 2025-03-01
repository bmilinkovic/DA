"""
Utility functions for loading and saving time series data.
"""

import numpy as np
import pandas as pd
import os


def load_time_series(file_path, format='csv', transpose=False):
    """
    Load time series data from a file.
    
    Parameters:
    file_path (str): Path to the file containing the time series data.
    format (str): Format of the file ('csv', 'npy', or 'pkl').
    transpose (bool): Whether to transpose the data (rows/columns swap).
    
    Returns:
    numpy.ndarray: Time series data.
    """
    if format.lower() == 'csv':
        data = pd.read_csv(file_path).values
    elif format.lower() == 'npy':
        data = np.load(file_path)
    elif format.lower() == 'pkl':
        data = pd.read_pickle(file_path).values
    else:
        raise ValueError(f"Unsupported format: {format}. Choose 'csv', 'npy', or 'pkl'.")
    
    if transpose:
        data = data.T
    
    return data


def save_time_series(data, file_path, format='csv', transpose=False):
    """
    Save time series data to a file.
    
    Parameters:
    data (numpy.ndarray): Time series data to save.
    file_path (str): Path where the data will be saved.
    format (str): Format to save the data ('csv', 'npy', or 'pkl').
    transpose (bool): Whether to transpose the data before saving.
    
    Returns:
    bool: True if saved successfully.
    """
    if transpose:
        data = data.T
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    if format.lower() == 'csv':
        pd.DataFrame(data).to_csv(file_path, index=False)
    elif format.lower() == 'npy':
        np.save(file_path, data)
    elif format.lower() == 'pkl':
        pd.DataFrame(data).to_pickle(file_path)
    else:
        raise ValueError(f"Unsupported format: {format}. Choose 'csv', 'npy', or 'pkl'.")
    
    return True 