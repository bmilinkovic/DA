"""
Tests for the data processing utility functions.
"""

import os
import numpy as np
import pandas as pd
import pytest
import tempfile
from dynamical_autonomy.utils.data_processing import load_time_series, save_time_series


@pytest.fixture
def sample_data():
    """Create sample time series data for testing."""
    # Create a simple time series dataset
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


def test_save_load_csv(sample_data):
    """Test saving and loading time series data in CSV format."""
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, 'test_data.csv')
        
        # Save data
        save_time_series(sample_data, file_path, format='csv')
        
        # Check file exists
        assert os.path.exists(file_path), f"File {file_path} was not created"
        
        # Load data
        loaded_data = load_time_series(file_path, format='csv')
        
        # Check data integrity
        pd.testing.assert_frame_equal(sample_data, loaded_data)
        
        # Test with transpose
        save_time_series(sample_data, file_path, format='csv', transpose=True)
        loaded_data_transposed = load_time_series(file_path, format='csv', transpose=True)
        pd.testing.assert_frame_equal(sample_data, loaded_data_transposed)


def test_save_load_numpy(sample_data):
    """Test saving and loading time series data in NumPy format."""
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, 'test_data.npy')
        
        # Convert to numpy array for saving
        np_data = sample_data.values
        
        # Save data
        save_time_series(np_data, file_path, format='npy')
        
        # Check file exists
        assert os.path.exists(file_path), f"File {file_path} was not created"
        
        # Load data
        loaded_data = load_time_series(file_path, format='npy')
        
        # Check data integrity
        np.testing.assert_array_equal(np_data, loaded_data)
        
        # Test with transpose
        save_time_series(np_data, file_path, format='npy', transpose=True)
        loaded_data_transposed = load_time_series(file_path, format='npy', transpose=True)
        np.testing.assert_array_equal(np_data, loaded_data_transposed)


def test_save_load_pickle(sample_data):
    """Test saving and loading time series data in pickle format."""
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, 'test_data.pkl')
        
        # Save data
        save_time_series(sample_data, file_path, format='pkl')
        
        # Check file exists
        assert os.path.exists(file_path), f"File {file_path} was not created"
        
        # Load data
        loaded_data = load_time_series(file_path, format='pkl')
        
        # Check data integrity
        pd.testing.assert_frame_equal(sample_data, loaded_data)


def test_unsupported_format(sample_data):
    """Test that unsupported formats raise appropriate errors."""
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, 'test_data.xyz')
        
        # Test save with unsupported format
        with pytest.raises(ValueError, match="Unsupported format"):
            save_time_series(sample_data, file_path, format='xyz')
        
        # Test load with unsupported format
        with pytest.raises(ValueError, match="Unsupported format"):
            load_time_series(file_path, format='xyz')


def test_directory_creation():
    """Test that save_time_series creates directories if they don't exist."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a nested path that doesn't exist
        nested_dir = os.path.join(temp_dir, 'nested', 'dir')
        file_path = os.path.join(nested_dir, 'test_data.csv')
        
        # Save data to the nested path
        data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        save_time_series(data, file_path, format='csv')
        
        # Check that the directory was created
        assert os.path.exists(nested_dir), "Nested directory was not created"
        assert os.path.exists(file_path), "File was not created in the nested directory"


if __name__ == "__main__":
    pytest.main(["-v", __file__]) 