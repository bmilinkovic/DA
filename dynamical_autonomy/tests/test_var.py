"""
Tests for the VAR model class.
"""

import numpy as np
import pytest
from dynamical_autonomy.models import VAR


def test_var_initialization():
    """Test that VAR model initializes correctly."""
    # Test with default parameters
    model = VAR()
    assert model.order == 1, "Default order should be 1"
    assert model.fit_intercept is True, "Default fit_intercept should be True"
    
    # Test with custom parameters
    model = VAR(order=2, fit_intercept=False)
    assert model.order == 2, "Order should be set to 2"
    assert model.fit_intercept is False, "fit_intercept should be set to False"


def test_var_fit_predict():
    """Test that VAR model can fit and predict data."""
    # Generate synthetic VAR(1) data
    n_samples = 200
    n_features = 3
    
    # Define VAR(1) parameters
    A = np.array([
        [0.6, 0.2, 0.1],
        [0.1, 0.5, 0.2],
        [0.1, 0.1, 0.7]
    ])
    
    # Generate data
    np.random.seed(42)
    data = np.zeros((n_samples, n_features))
    noise = np.random.normal(0, 0.1, (n_samples, n_features))
    
    for t in range(1, n_samples):
        data[t] = A @ data[t-1] + noise[t]
    
    # Fit VAR model
    model = VAR(order=1)
    model.fit(data)
    
    # Check coefficients shape
    assert model.coef_.shape == (n_features, n_features), "Coefficient matrix should be n_features x n_features"
    
    if model.fit_intercept:
        assert model.intercept_.shape == (n_features,), "Intercept should be a vector of length n_features"
    
    # Test prediction
    predictions = model.predict(data)
    assert predictions.shape == (n_samples - model.order, n_features), "Predictions shape is incorrect"
    
    # Test one-step prediction
    one_step = model.predict_step(data[-1:])
    assert one_step.shape == (1, n_features), "One-step prediction shape is incorrect"


def test_var_granger_causality():
    """Test Granger causality calculation."""
    # Generate synthetic data with known causality
    n_samples = 300
    n_features = 3
    
    # x1 causes x2, x2 causes x3, x3 does not cause x1 or x2
    np.random.seed(42)
    data = np.zeros((n_samples, n_features))
    noise = np.random.normal(0, 0.1, (n_samples, n_features))
    
    for t in range(1, n_samples):
        # x1 depends only on its past
        data[t, 0] = 0.7 * data[t-1, 0] + noise[t, 0]
        # x2 depends on x1 and its past
        data[t, 1] = 0.3 * data[t-1, 0] + 0.5 * data[t-1, 1] + noise[t, 1]
        # x3 depends on x2 and its past
        data[t, 2] = 0.4 * data[t-1, 1] + 0.4 * data[t-1, 2] + noise[t, 2]
    
    # Fit VAR model
    model = VAR(order=1)
    model.fit(data)
    
    # Calculate Granger causality
    F = model.granger_causality()
    
    # Check shape
    assert F.shape == (n_features, n_features), "Granger causality matrix should be n_features x n_features"
    
    # Check expected causality patterns
    # Diagonal elements should be zero (no self-causality)
    assert np.allclose(np.diag(F), 0), "Diagonal elements should be zero"
    
    # x1 should cause x2
    assert F[0, 1] > 0.1, "x1 should cause x2"
    
    # x2 should cause x3
    assert F[1, 2] > 0.1, "x2 should cause x3"
    
    # x3 should not cause x1 or x2
    assert F[2, 0] < 0.1, "x3 should not cause x1"
    assert F[2, 1] < 0.1, "x3 should not cause x2"


def test_var_fit_with_exog():
    """Test VAR model with exogenous variables."""
    # Generate synthetic data
    n_samples = 200
    n_features = 2
    n_exog = 1
    
    np.random.seed(42)
    data = np.random.normal(0, 1, (n_samples, n_features))
    exog = np.random.normal(0, 1, (n_samples, n_exog))
    
    # Fit model with exogenous variables
    model = VAR(order=1)
    model.fit(data, exog=exog)
    
    # Check that exog coefficients exist
    assert hasattr(model, 'exog_coef_'), "Model should have exog_coef_ attribute"
    assert model.exog_coef_.shape == (n_features, n_exog), "Exog coefficients shape is incorrect"
    
    # Test prediction with exog
    predictions = model.predict(data, exog=exog)
    assert predictions.shape == (n_samples - model.order, n_features), "Predictions shape is incorrect"


if __name__ == "__main__":
    test_var_initialization()
    test_var_fit_predict()
    test_var_granger_causality()
    test_var_fit_with_exog()
    print("All tests passed!") 