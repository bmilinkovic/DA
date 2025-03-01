"""
Tests for the mdare function.
"""

import numpy as np
import pytest
from dynamical_autonomy.models import mdare


def test_mdare_output_dimensions():
    """Test that mdare outputs have the correct dimensions."""
    # Example inputs
    n, r = 2, 2  # 2 variables, 2 states
    A = np.array([[0.9, 0.1], [0.1, 0.8]])  # State transition matrix (r x r)
    C = np.array([[1, 0], [0, 1]])  # Observation matrix (n x r)
    Q = np.eye(r)  # Process noise covariance (r x r)
    R = np.eye(n)  # Observation noise covariance (n x n)
    
    # Call mdare
    K, V, residual, L, P = mdare(A, C, Q, R)
    
    # Check output dimensions
    assert K.shape == (r, n), f"Kalman gain should be {r}x{n}"
    assert V.shape == (n, n), f"Innovations covariance should be {n}x{n}"
    assert P.shape == (r, r), f"DARE solution should be {r}x{r}"
    
    # Check other properties
    assert isinstance(residual, float), "Residual should be a float"
    assert np.all(np.abs(L) < 1), "Eigenvalues should lie within the unit circle"


def test_mdare_solve_stable_system():
    """Test that mdare correctly solves a stable system."""
    # Example stable system
    A = np.array([[0.5, 0.2], [0.1, 0.6]])
    C = np.array([[1.0, 0.0], [0.0, 1.0]])
    Q = np.eye(2)
    R = 0.1 * np.eye(2)
    
    # Call mdare
    K, V, residual, L, P = mdare(A, C, Q, R)
    
    # Check if solution is valid
    assert K is not None, "Kalman gain should not be None"
    assert V is not None, "Innovations covariance should not be None"
    assert residual >= 0, "Residual should be non-negative"
    assert residual < 1e-6, "Residual should be small for an exact solution"
    
    # Check if P is symmetric and positive-definite
    assert np.allclose(P, P.T), "P should be symmetric"
    assert np.all(np.linalg.eigvals(P) > 0), "P should be positive-definite"


def test_mdare_unstable_system():
    """Test that mdare returns error indicators for an unstable system."""
    # Example unstable system (eigenvalues outside unit circle)
    A = np.array([[1.5, 0.1], [0.1, 1.2]])
    C = np.array([[1.0, 0.0], [0.0, 1.0]])
    Q = np.eye(2)
    R = np.eye(2)
    
    # Call mdare
    K, V, residual, L, P = mdare(A, C, Q, R)
    
    # Check error indicators
    assert residual < 0, "Residual should be negative for an unstable system"
    assert K is None, "Kalman gain should be None for an unstable system"
    assert V is None, "Innovations covariance should be None for an unstable system"
    assert P is None, "DARE solution should be None for an unstable system"


if __name__ == "__main__":
    test_mdare_output_dimensions()
    test_mdare_solve_stable_system()
    test_mdare_unstable_system()
    print("All tests passed!") 