"""
Vector Autoregressive (VAR) model functions for time series analysis.
"""

import numpy as np
from statsmodels.tsa.api import VAR
from scipy.linalg import solve_discrete_are

from .state_space import mdare


def generate_var_data(A, V, n_obs, n_vars=None, n_trials=1):
    """
    Generate time series data from a VAR model.

    Parameters:
    A (numpy.ndarray): VAR coefficient matrix, shape (n, n, p), where:
        - n is the number of variables
        - p is the model order
    V (numpy.ndarray): Residual covariance matrix, shape (n, n)
    n_obs (int): Number of observations (time points)
    n_vars (int, optional): Number of variables. If None, inferred from A.
    n_trials (int, optional): Number of trials. Default is 1.

    Returns:
    numpy.ndarray: Generated time series data, shape (n_obs, n_vars, n_trials) or (n_obs, n_vars) if n_trials=1

    Raises:
    ValueError: If V is not symmetric or positive definite.
    """
    n, _, p = A.shape
    if n_vars is None:
        n_vars = n
    else:
        assert n == n_vars, "Mismatch between A and n_vars"
    assert V.shape == (n_vars, n_vars), "Mismatch between V and n_vars"

    data = np.zeros((n_obs, n_vars, n_trials))

    for trial in range(n_trials):
        noise = np.random.multivariate_normal(mean=np.zeros(n_vars), cov=V, size=n_obs)
        for t in range(p, n_obs):
            for lag in range(1, p + 1):
                data[t, :, trial] += np.dot(A[:, :, lag - 1], data[t - lag, :, trial])
            data[t, :, trial] += noise[t]

    # If only one trial, remove the last dimension
    if n_trials == 1:
        data = data[:, :, 0]
        
    return data


def demean_data(data):
    """
    Demean the time series data along the time axis.

    Parameters:
    data (numpy.ndarray): Time series data, shape (n_obs, n_vars, n_trials) or (n_obs, n_vars)

    Returns:
    numpy.ndarray: Demeaned time series data, same shape as input.
    """
    if data.ndim == 2:
        # Shape (n_obs, n_vars)
        mean_data = np.mean(data, axis=0, keepdims=True)
        return data - mean_data
    else:
        # Shape (n_obs, n_vars, n_trials)
        mean_data = np.mean(data, axis=0, keepdims=True)
        return data - mean_data


def estimate_var_order(data, max_order, criterion='aic'):
    """
    Estimate the optimal VAR model order using an information criterion.

    Parameters:
    data (numpy.ndarray): Time series data, shape (n_obs, n_vars)
    max_order (int): Maximum model order to consider
    criterion (str): Information criterion to use ('aic', 'hqic', 'bic')

    Returns:
    int: Optimal model order based on the specified criterion.

    Raises:
    ValueError: If the criterion is invalid or if the model fails to estimate.
    """
    # Check for NaNs or infinities in the data
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Input data contains NaNs or infinities.")

    # Check rank of covariance matrix
    cov_matrix = np.cov(data.T)
    if np.linalg.matrix_rank(cov_matrix) < data.shape[1]:
        raise ValueError("Covariance matrix is not full rank. Ensure data is well-conditioned.")

    model = VAR(data)
    try:
        selections = model.select_order(maxlags=max_order)
        if not hasattr(selections, criterion):
            raise ValueError(f"Invalid criterion '{criterion}'. Must be one of 'aic', 'hqic', or 'bic'.")
        return getattr(selections, criterion)
    except np.linalg.LinAlgError:
        raise ValueError("Covariance matrix is not positive definite. Ensure data is well-conditioned.")


def fit_var_model(data, order):
    """
    Fit a VAR model to the data with the specified order.

    Parameters:
    data (numpy.ndarray): Time series data, shape (n_obs, n_vars)
    order (int): VAR model order

    Returns:
    VARResults: Fitted VAR model results.
    """
    model = VAR(data)
    return model.fit(maxlags=order)


def generate_random_var(n_vars, actual_order, connectivity=None, decay_factor=0.5, spectral_radius=0.9):
    """
    Generate a random stable VAR model given parameters.

    Parameters:
    n_vars (int): Number of variables
    actual_order (int): Actual model order
    connectivity (numpy.ndarray, optional): Connectivity matrix (binary), shape (n_vars, n_vars)
        If None, full connectivity is assumed.
    decay_factor (float, optional): Decay factor for coefficients. Default is 0.5.
    spectral_radius (float, optional): Desired spectral radius (must be < 1 for stability). Default is 0.9.

    Returns:
    numpy.ndarray: VAR coefficient matrix A, shape (n, n, p)
    """
    if connectivity is None:
        connectivity = np.ones((n_vars, n_vars))
        
    A = np.zeros((n_vars, n_vars, actual_order))
    for lag in range(actual_order):
        coeff = np.random.uniform(-1, 1, size=(n_vars, n_vars)) * connectivity
        coeff *= decay_factor ** lag
        A[:, :, lag] = coeff

    total_matrix = sum(A[:, :, lag] for lag in range(actual_order))
    radius = max(abs(np.linalg.eigvals(total_matrix)))
    if radius >= spectral_radius:
        A *= spectral_radius / radius

    return A


def var_to_pwcgc(A, V):
    """
    Compute pairwise conditional Granger causality from a VAR model.

    Parameters:
    A (numpy.ndarray): VAR coefficients matrix, shape (n, n, p)
    V (numpy.ndarray): Residual covariance matrix, shape (n, n)

    Returns:
    numpy.ndarray: Pairwise conditional Granger causality matrix, shape (n, n)
    """
    n, _, p = A.shape
    DV = np.diag(V)
    LDV = np.log(DV)

    F = np.full((n, n), np.nan)
    for y in range(n):
        r = [i for i in range(n) if i != y]
        _, VR, rep = vardare(A, V, y, r)
        if rep < 0:
            continue
        F[r, y] = np.log(np.diag(VR)) - LDV[r]
    return F


def vardare(A, V, y, r):
    """
    Solve DARE for the reduced VAR model.

    Parameters:
    A (numpy.ndarray): VAR coefficients matrix, shape (n, n, p)
    V (numpy.ndarray): Residual covariance matrix, shape (n, n)
    y (int): Index of the dependent variable
    r (list): Indices of the conditioning variables

    Returns:
    tuple: Kalman gain matrix (K), innovations covariance matrix (VR), DARE report (rep)
    """
    n_y = 1  # Single dependent variable
    n_r = len(r)
    p = A.shape[2]  # Model order
    p_ny = p * n_y
    p_ny1 = p_ny - n_y

    # Construct State-Space Matrices
    A_ss = np.block([
        [A[y, y, :].reshape(n_y, p_ny)],
        [np.eye(p_ny1), np.zeros((p_ny1, n_y))]
    ])
    C_ss = A[r, y, :].reshape(n_r, p_ny)
    Q_ss = np.block([
        [V[y, y].reshape(n_y, n_y), np.zeros((n_y, p_ny1))],
        [np.zeros((p_ny1, n_y)), np.zeros((p_ny1, p_ny1))]
    ])
    S_ss = np.zeros((p_ny, n_r))  # Ensure S has correct shape
    R_ss = V[np.ix_(r, r)]

    return mdare(A_ss, C_ss, Q_ss, R_ss, S_ss) 