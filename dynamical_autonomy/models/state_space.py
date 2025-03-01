"""
State-space model functions for dynamical systems analysis.
"""

import numpy as np
from scipy.linalg import schur, solve


def mdare(A, C, Q, R, S=None):
    """
    Solve the Discrete Algebraic Riccati Equation (DARE) using the generalized Schur decomposition.

    Parameters:
    ----------
    A : ndarray
        State transition matrix (r x r).
    C : ndarray
        Observation matrix (n x r).
    Q : ndarray
        Process noise covariance matrix (r x r).
    R : ndarray
        Measurement noise covariance matrix (n x n).
    S : ndarray, optional
        Cross-covariance matrix (r x n). Defaults to None (assumes zero).

    Returns:
    -------
    K : ndarray
        Kalman gain matrix (r x n).
    V : ndarray
        Innovations covariance matrix (n x n).
    residual : float
        Relative residual of the DARE solution.
    L : ndarray
        Stabilizing eigenvalues of the DARE solution.
    P : ndarray
        Solution to the DARE (r x r).
    """
    # Validate input dimensions
    r, r1 = A.shape
    assert r1 == r, "A must be a square matrix"
    n, r1 = C.shape
    assert r1 == r, "C must have the same number of columns as A"
    assert Q.shape == (r, r), "Q must be r x r"
    assert R.shape == (n, n), "R must be n x n"
    S = np.zeros((r, n)) if S is None else S
    assert S.shape == (r, n), "S must be r x n"

    # Construct extended matrices for generalized Schur decomposition
    H = np.block([
        [A.T, np.zeros((r, r)), C.T],
        [-Q, np.eye(r), -S],
        [S.T, np.zeros((n, r)), R]
    ])
    J = np.block([
        [np.eye(r), np.zeros((r, r + n))],
        [np.zeros((r, r)), A, np.zeros((r, n))],
        [np.zeros((n, r + r)), -C, np.zeros((n, n))]
    ])

    # Perform Schur decomposition (QZ algorithm)
    T, Z = schur(J[: 2*r , : 2*r], H[: 2*r , : 2*r], output="real")

    # Extract eigenvalues and sort for stabilization
    eigvals = np.diag(T) / np.diag(Z)
    stable_indices = np.abs(eigvals) < 1
    if not np.all(stable_indices[:r]) or np.any(stable_indices[r:]):
        return None, None, -1, eigvals, None

    # Solve for P using stabilizing invariant subspace
    Z1, Z2 = Z[:r, :r], Z[r:, :r]
    P = np.linalg.solve(Z1.T, Z2.T).T
    P = (P + P.T) / 2  # Enforce symmetry

    # Compute Kalman gain
    U = A @ P @ C.T + S
    V = C @ P @ C.T + R
    K = U @ np.linalg.inv(V)

    # Compute residual for accuracy
    APA = A @ P @ A.T - P
    UK = U @ K.T
    residual = np.linalg.norm(APA - UK + Q) / (
        1 + np.linalg.norm(APA) + np.linalg.norm(UK) + np.linalg.norm(Q)
    )

    return K, V, residual, eigvals[stable_indices], P


def tsdata_to_ss(observations, past_future_horizon, model_order):
    """
    Estimate an innovations form state-space model from an empirical observation
    time series using Larimore's Canonical Correlations Analysis (CCA) state
    space-subspace algorithm.

    Parameters:
    observations : ndarray of shape (num_variables, num_time_steps, num_realizations)
        Observation process time series.
    past_future_horizon : int or list or tuple of two ints
        Past and future horizons for canonical correlations.
        If a scalar is provided, past_horizon = future_horizon = past_future_horizon.
        If a list or tuple of two integers is provided, past_horizon = past_future_horizon[0], future_horizon = past_future_horizon[1].
    model_order : int
        State-space model order (should be an integer between 0 and num_variables * min(past_horizon, future_horizon))

    Returns:
    state_transition : ndarray
        Estimated state transition matrix (A).
    observation_matrix : ndarray
        Estimated observation matrix (C).
    kalman_gain : ndarray
        Estimated Kalman gain matrix (K).
    innovations_covariance : ndarray
        Estimated innovations covariance matrix (V).
    state_estimates : ndarray
        Estimated state process time series (Z).
    innovations : ndarray
        Estimated innovations process time series (E).
    """
    # Ensure that inputs are numpy arrays
    observations = np.asarray(observations)

    # Get dimensions of observations
    num_variables, num_time_steps, num_realizations = observations.shape

    # Validate past_future_horizon and set past_horizon and future_horizon
    def is_positive_int(x):
        return isinstance(x, int) and x >= 1

    if is_positive_int(past_future_horizon):
        past_horizon = past_future_horizon
        future_horizon = past_future_horizon
    elif isinstance(past_future_horizon, (list, tuple)) and len(past_future_horizon) == 2 and all(is_positive_int(x) for x in past_future_horizon):
        past_horizon, future_horizon = past_future_horizon
    else:
        raise ValueError('Past/future horizon must be a scalar or a 2-element list/tuple of positive integers.')

    # Validate model_order
    rmax = num_variables * min(past_horizon, future_horizon)
    if not (isinstance(model_order, int) and 0 < model_order <= rmax):
        raise ValueError('Model order must be a positive integer <= num_variables * min(past_horizon, future_horizon) = %d' % rmax)

    # Remove the mean across the time dimension for each variable and realization
    def demean(data):
        """
        Remove the mean across the time dimension for each variable and realization.
        """
        mean_data = np.mean(data, axis=1, keepdims=True)
        return data - mean_data

    observations = demean(observations)  # No constant term

    # Compute various indices needed for slicing
    m_minus_p = num_time_steps - past_horizon
    m_minus_p_plus1 = m_minus_p + 1
    m_minus_f = num_time_steps - future_horizon
    m_p_f = m_minus_p_plus1 - future_horizon + 1

    total_samples_M = num_realizations * m_minus_p
    total_samples_M1 = num_realizations * m_minus_p_plus1
    total_samples_Mh = num_realizations * m_p_f

    # Prepare future observations X_future
    X_future = np.zeros((num_variables, future_horizon, m_p_f, num_realizations))
    for k in range(future_horizon):
        start_idx = past_horizon + k
        end_idx = m_minus_f + k
        X_future[:, k, :, :] = observations[:, start_idx:end_idx, :]

    # Reshape X_future to shape (num_variables * future_horizon, total_samples_Mh)
    X_future = X_future.reshape(num_variables * future_horizon, total_samples_Mh)

    # Prepare past observations X_past_full
    X_past_full = np.zeros((num_variables, past_horizon, m_minus_p_plus1, num_realizations))
    for k in range(past_horizon - 1):
        start_idx = past_horizon - k
        end_idx = num_time_steps - k
        X_past_full[:, k, :, :] = observations[:, start_idx:end_idx, :]

    # Reshape X_past_full to shape (num_variables * past_horizon, total_samples_M1)
    X_past_full = X_past_full.reshape(num_variables * past_horizon, total_samples_M1)

    # Extract X_past for m_p_f time steps
    X_past = X_past_full[:, :total_samples_Mh]

    # Compute Cholesky factors for future and past observations
    # Compute covariance matrices
    Wf_cov = (X_future @ X_future.T) / total_samples_Mh
    Wp_cov = (X_past @ X_past.T) / total_samples_Mh

    # Check if covariance matrices are positive definite
    try:
        Wf_cholesky = np.linalg.cholesky(Wf_cov)
    except np.linalg.LinAlgError:
        raise ValueError('Forward weight matrix not positive definite')

    try:
        Wp_cholesky = np.linalg.cholesky(Wp_cov)
    except np.linalg.LinAlgError:
        raise ValueError('Backward weight matrix not positive definite')

    # Compute the 'OH' estimate: regress future on past
    beta_regression = np.linalg.lstsq(X_past.T, X_future.T, rcond=None)[0].T
    if not np.all(np.isfinite(beta_regression)):
        raise ValueError('Subspace regression failed.')

    # Perform SVD of CCA-weighted OH estimate
    beta_weighted = np.linalg.solve(Wf_cholesky, beta_regression @ Wp_cholesky)
    U_svd, singular_values, Vh_svd = np.linalg.svd(beta_weighted, full_matrices=False)
    if not np.all(np.isfinite(singular_values)):
        raise ValueError('SVD failed.')

    # Singular values
    sval = np.diag(singular_values)

    # Extract right singular vectors (V) from Vh_svd
    V_svd = Vh_svd.T  # Shape: (n * p, min(n * f, n * p))

    # Take the first 'model_order' singular vectors
    V_r = V_svd[:, :model_order]  # Shape: (n * p, model_order)

    # Compute sqrt of singular values
    sqrt_sval = np.sqrt(singular_values[:model_order])
    diag_sqrt_sval = np.diag(sqrt_sval)  # Shape: (model_order, model_order)

    # Compute state estimates Z
    Z_temp = diag_sqrt_sval @ V_r.T @ Wp_cholesky  # Resulting shape: (model_order, total_samples_M1)
    state_estimates = Z_temp.reshape(model_order, m_minus_p_plus1, num_realizations)
    if not np.all(np.isfinite(state_estimates)):
        raise ValueError('Kalman states estimation failed.')

    # Calculate model parameters by regression
    # Reshape observations and state estimates for regression
    observations_reshaped = observations[:, past_horizon:num_time_steps, :].reshape(num_variables, total_samples_M)
    Z_reshaped = state_estimates[:, 0:m_minus_p, :].reshape(model_order, total_samples_M)

    # Compute observation matrix C
    observation_matrix = np.linalg.lstsq(Z_reshaped.T, observations_reshaped.T, rcond=None)[0].T
    if not np.all(np.isfinite(observation_matrix)):
        raise ValueError('Observation matrix estimation failed.')

    # Compute innovations E and covariance V
    innovations = observations_reshaped - observation_matrix @ Z_reshaped
    innovations_covariance = (innovations @ innovations.T) / total_samples_M

    # Compute state transition matrix A and Kalman gain K by regression
    Z_past = state_estimates[:, 0:m_minus_p, :].reshape(model_order, total_samples_M)
    Z_future = state_estimates[:, 1:m_minus_p_plus1, :].reshape(model_order, total_samples_M)

    # Regression equation Z_{t+1} = A * Z_t + K * E_t
    # Combine Z_past and innovations for regression
    Z_past_E = np.vstack([Z_past, innovations])

    # Solve for [A, K]
    AK = np.linalg.lstsq(Z_past_E.T, Z_future.T, rcond=None)[0].T
    state_transition = AK[:, :model_order]
    kalman_gain = AK[:, model_order:]

    return state_transition, observation_matrix, kalman_gain, innovations_covariance, state_estimates, innovations


def var_to_ss(VARA, V, report=None):
    """
    Convert VAR coefficients to state-space form.

    Parameters:
    VARA (numpy.ndarray): VAR coefficients matrix, shape (n, n, p)
    V (numpy.ndarray): Residual covariance matrix, shape (n, n)
    report (callable, optional): Function to report progress.

    Returns:
    tuple:
        - A (numpy.ndarray): State transition matrix.
        - C (numpy.ndarray): Observation matrix.
        - K (numpy.ndarray): Kalman gain matrix.
        - V (numpy.ndarray): Innovations covariance matrix.
    """
    if report:
        report({'status': 'Starting VAR to state-space conversion'})

    n, _, p = VARA.shape
    
    if report:
        report({'status': f'VAR model: {n} variables, order {p}'})
    
    np1 = n * p
    np2 = n * (p - 1)
    
    A = np.zeros((np1, np1))
    # Fill the first n rows of A with the VAR coefficients
    for k in range(p):
        A[:n, n*k:n*(k+1)] = VARA[:, :, k]
    # Fill the rest of A with the identity matrix (companion form)
    if p > 1:
        A[n:np1, :np2] = np.eye(np2)
    
    # Observation matrix: selects the first n states
    C = np.zeros((n, np1))
    C[:, :n] = np.eye(n)
    
    # Process noise covariance
    Q = np.zeros((np1, np1))
    Q[:n, :n] = V
    
    if report:
        report({'status': 'Created state-space matrices'})

    # Solve DARE to find Kalman gain and innovations covariance
    K, V_ss, _, _, _ = mdare(A, C, Q, np.zeros((n, n)))
    
    if report:
        report({'status': 'DARE solved successfully'})
    
    return A, C, K, V_ss


def ss_info(A, C, K, V, report=None):
    """
    Compute information-theoretic quantities for a state-space model.

    Parameters:
    A (numpy.ndarray): State transition matrix.
    C (numpy.ndarray): Observation matrix.
    K (numpy.ndarray): Kalman gain matrix.
    V (numpy.ndarray): Innovations covariance matrix.
    report (callable, optional): Function to report progress.

    Returns:
    tuple:
        - P (numpy.ndarray): State covariance.
        - H (float): Entropy rate.
        - AIS (float): Active information storage.
        - TE (float): Transfer entropy.
    """
    if report:
        report({'status': 'Computing information-theoretic quantities'})
    
    n, m = C.shape
    
    # State covariance satisfies: P = A*P*A' + K*V*K'
    # We can solve this using the Lyapunov equation: P = A*P*A' + Q
    # Where Q = K*V*K'
    Q = K @ V @ K.T
    P = solve_discrete_lyapunov(A, Q)
    
    # Entropy rate
    H = 0.5 * (n * (1 + np.log(2 * np.pi)) + np.log(np.linalg.det(V)))
    
    # Active information storage (AIS)
    AIS = 0.5 * np.log(np.linalg.det(C @ P @ C.T)) - 0.5 * np.log(np.linalg.det(V))
    
    # Transfer entropy (TE) - simplified computation
    TE = 0.5 * np.log(np.linalg.det(C @ P @ C.T)) - 0.5 * np.log(np.linalg.det(C @ P @ C.T - C @ K @ V @ K.T @ C.T))
    
    if report:
        report({'status': 'Information-theoretic quantities computed'})
    
    return P, H, AIS, TE 