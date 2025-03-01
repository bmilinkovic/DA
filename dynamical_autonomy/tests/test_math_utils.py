"""
Tests for the math utility functions.
"""

import numpy as np
import pytest
from dynamical_autonomy.utils.math_utils import corr_rand


def test_corr_rand_dimensions():
    """Test that corr_rand outputs have the correct dimensions."""
    n = 5  # Dimension of the correlation matrix
    
    # Generate random correlation matrix
    C = corr_rand(n)
    
    # Check dimensions
    assert C.shape == (n, n), f"Correlation matrix should be {n}x{n}"
    
    # Check that it's a valid correlation matrix
    assert np.allclose(np.diag(C), 1.0), "Diagonal elements should be 1.0"
    assert np.allclose(C, C.T), "Correlation matrix should be symmetric"
    
    # Check that eigenvalues are positive (positive definite)
    eigvals = np.linalg.eigvals(C)
    assert np.all(eigvals > -1e-10), "Correlation matrix should be positive semi-definite"
    
    # Check that off-diagonal elements are in [-1, 1]
    mask = ~np.eye(n, dtype=bool)  # Mask to select off-diagonal elements
    assert np.all(C[mask] >= -1.0) and np.all(C[mask] <= 1.0), "Correlation values should be in [-1, 1]"


def test_corr_rand_with_multi_information():
    """Test that corr_rand can generate matrices with specified multi-information."""
    n = 4  # Dimension of the correlation matrix
    g = 0.5  # Target multi-information
    
    # Generate random correlation matrix with specified multi-information
    C = corr_rand(n, g=g)
    
    # Check dimensions
    assert C.shape == (n, n), f"Correlation matrix should be {n}x{n}"
    
    # Check that it's a valid correlation matrix
    assert np.allclose(np.diag(C), 1.0), "Diagonal elements should be 1.0"
    assert np.allclose(C, C.T), "Correlation matrix should be symmetric"
    
    # Calculate multi-information
    # Multi-information = -0.5 * log(det(C))
    det_C = np.linalg.det(C)
    calculated_g = -0.5 * np.log(det_C)
    
    # Check that multi-information is close to the target
    assert np.isclose(calculated_g, g, atol=0.1), f"Multi-information should be close to {g}"


def test_corr_rand_with_variance_exponent():
    """Test that corr_rand works with different variance exponents."""
    n = 3  # Dimension of the correlation matrix
    
    # Generate matrices with different variance exponents
    C1 = corr_rand(n, vexp=1)  # More uniform eigenvalues
    C2 = corr_rand(n, vexp=3)  # More varied eigenvalues
    
    # Check that both are valid correlation matrices
    assert np.allclose(np.diag(C1), 1.0), "Diagonal elements should be 1.0"
    assert np.allclose(np.diag(C2), 1.0), "Diagonal elements should be 1.0"
    
    # Calculate eigenvalues
    eigvals1 = np.linalg.eigvals(C1)
    eigvals2 = np.linalg.eigvals(C2)
    
    # Calculate variance of eigenvalues
    var1 = np.var(eigvals1)
    var2 = np.var(eigvals2)
    
    # Higher vexp should generally lead to more variance in eigenvalues
    # This is a probabilistic test, so it might not always pass
    # We'll use a high number of retries to make it more reliable
    max_retries = 10
    for _ in range(max_retries):
        if var2 > var1:
            break
        
        # Retry if the test fails
        C1 = corr_rand(n, vexp=1)
        C2 = corr_rand(n, vexp=3)
        eigvals1 = np.linalg.eigvals(C1)
        eigvals2 = np.linalg.eigvals(C2)
        var1 = np.var(eigvals1)
        var2 = np.var(eigvals2)
    
    # Assert that higher vexp leads to more variance in eigenvalues
    # This might still fail occasionally due to randomness
    assert var2 > var1, "Higher variance exponent should lead to more variance in eigenvalues"


def test_corr_rand_edge_cases():
    """Test edge cases for corr_rand."""
    # Test with n=1 (should return [[1]])
    C = corr_rand(1)
    assert C.shape == (1, 1), "1x1 correlation matrix should have shape (1, 1)"
    assert C[0, 0] == 1.0, "1x1 correlation matrix should be [[1]]"
    
    # Test with n=2 and g=0 (should be close to identity)
    C = corr_rand(2, g=0)
    assert np.allclose(C, np.eye(2), atol=0.1), "Correlation matrix with g=0 should be close to identity"
    
    # Test with very high multi-information (should have high correlations)
    n = 3
    g = 2.0  # High multi-information
    C = corr_rand(n, g=g)
    
    # Calculate average absolute correlation
    mask = ~np.eye(n, dtype=bool)  # Mask to select off-diagonal elements
    avg_abs_corr = np.mean(np.abs(C[mask]))
    
    # High multi-information should lead to high correlations
    assert avg_abs_corr > 0.5, "High multi-information should lead to high correlations"


if __name__ == "__main__":
    test_corr_rand_dimensions()
    test_corr_rand_with_multi_information()
    test_corr_rand_with_variance_exponent()
    test_corr_rand_edge_cases()
    print("All tests passed!") 