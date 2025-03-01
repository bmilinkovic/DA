"""
Core functions for calculating dynamical autonomy metrics.
"""

import numpy as np
from numpy.linalg import inv
from scipy.linalg import solve_discrete_lyapunov, cholesky
import time
from concurrent.futures import ProcessPoolExecutor


def compute_dynamical_autonomy(A, Q, C, R_X, L):
    """
    Compute Dynamical Autonomy (DA) and related metrics for a VAR system.

    Parameters:
    A (numpy.ndarray): State transition matrix
    Q (numpy.ndarray): Process noise covariance
    C (numpy.ndarray): Observation matrix for microscopic variables (X)
    R_X (numpy.ndarray): Observation noise covariance for X
    L (numpy.ndarray): Coarse-graining matrix for macroscopic variables (Y)

    Returns:
    dict: Dynamical autonomy metrics including AIS(Y), TE(X->Y), DA, and PID atoms
    """
    # Step 1: Solve for Sigma_Z (state covariance)
    Sigma_Z = solve_discrete_lyapunov(A, Q)

    # Step 2: Compute Sigma_X (microscopic covariance)
    Sigma_X = C @ Sigma_Z @ C.T + R_X

    # Step 3: Compute Sigma_Y (macroscopic covariance)
    Sigma_Y = L @ Sigma_X @ L.T

    # Step 4: Compute conditional covariance Sigma_Y | Y_{t-}
    Sigma_Yt_Ytm1 = L @ (C @ A @ Sigma_Z @ C.T) @ L.T  # Cross-covariance between Y_t and Y_{t-}
    Sigma_Ytm1 = L @ (C @ A @ Sigma_Z @ A.T @ C.T + C @ Q @ C.T) @ L.T  # Covariance of Y_{t-}
    Sigma_Yt_given_Ytm1 = Sigma_Y - Sigma_Yt_Ytm1 @ inv(Sigma_Ytm1) @ Sigma_Yt_Ytm1.T

    # Step 5: Compute conditional covariance Sigma_Y | Y_{t-}, X_{t-}
    Sigma_Xtm1 = C @ A @ Sigma_Z @ A.T @ C.T + C @ Q @ C.T  # Covariance of X_{t-}

    # Cross-covariances between Y_{t-} and X_{t-}
    Sigma_Ytm1_Xtm1 = L @ Sigma_Xtm1  # Cross-covariance between Y_{t-} and X_{t-}
    Sigma_Xtm1_Ytm1 = Sigma_Ytm1_Xtm1.T  # Cross-covariance between X_{t-} and Y_{t-}

    # Joint covariance of [Y_{t-}, X_{t-}]
    Sigma_Ytm1_Xtm1_joint = np.block([
        [Sigma_Ytm1, Sigma_Ytm1_Xtm1],
        [Sigma_Xtm1_Ytm1, Sigma_Xtm1]
    ])

    # Cross-covariance of Y_t and [Y_{t-}, X_{t-}]
    Sigma_Yt_Ytm1_Xtm1 = np.hstack([Sigma_Yt_Ytm1, L @ Sigma_Xtm1])

    # Conditional covariance of Y_t | Y_{t-}, X_{t-}
    Sigma_Yt_given_Ytm1_Xtm1 = Sigma_Y - Sigma_Yt_Ytm1_Xtm1 @ inv(Sigma_Ytm1_Xtm1_joint) @ Sigma_Yt_Ytm1_Xtm1.T

    # Compute determinants
    det_Y = np.linalg.det(Sigma_Y)
    det_Y_given_Ytm1 = np.linalg.det(Sigma_Yt_given_Ytm1)
    det_Y_given_Ytm1_Xtm1 = np.linalg.det(Sigma_Yt_given_Ytm1_Xtm1)

    # Calculate AIS(Y) and TE(X -> Y)
    AIS_Y = 0.5 * np.log(det_Y / det_Y_given_Ytm1)
    TE_X_to_Y = 0.5 * np.log(det_Y_given_Ytm1 / det_Y_given_Ytm1_Xtm1)

    # Calculate DA
    DA = AIS_Y - TE_X_to_Y

    # PID atoms
    I_Yt_Ytm1 = AIS_Y
    I_Yt_Xtm1 = 0.5 * np.log(det_Y / det_Y_given_Ytm1_Xtm1)
    I_red = min(I_Yt_Ytm1, I_Yt_Xtm1)

    I_unq_Ytm1 = I_Yt_Ytm1 - I_red
    I_unq_Xtm1 = TE_X_to_Y - (I_Yt_Xtm1 - I_red)
    I_syn = TE_X_to_Y - (I_Yt_Xtm1 - I_red)
    DA_excl = I_unq_Ytm1 + I_syn - I_unq_Xtm1

    return {
        "AIS_Y": AIS_Y,
        "TE_X_to_Y": TE_X_to_Y,
        "DA": DA,
        "I_red": I_red,
        "I_unq_Ytm1": I_unq_Ytm1,
        "I_unq_Xtm1": I_unq_Xtm1,
        "I_syn": I_syn,
        "DA_excl": DA_excl
    }


def define_macroscopic_process(X, L):
    """
    Define a macroscopic process as a linear combination of X.
    
    Parameters:
    X (numpy.ndarray): Microscopic process time series data
    L (numpy.ndarray): Coarse-graining matrix for macroscopic variables
    
    Returns:
    numpy.ndarray: Macroscopic process time series
    """
    return L @ X


def compute_covariances(Y, X, lags=1):
    """
    Compute conditional covariance for DA and DI.

    Parameters:
    - Y: Macroscopic process (1xT)
    - X: Microscopic process (n_vars x T)
    - lags: Number of past lags to consider

    Returns:
    - cov_Y_cond: Conditional covariance of Y_t | Y_{past}
    - cov_Y_joint: Conditional covariance of Y_t | X_{past}, Y_{past}
    """
    Y = Y.flatten()  # Ensure Y is a 1D array
    T = Y.shape[0]   # Total time steps

    # Create lagged regressors
    Y_past = np.vstack([Y[i:T - lags + i] for i in range(lags)])  # (lags, T-lags)
    X_past = np.hstack([X[:, i:T - lags + i] for i in range(lags)])  # (n_vars * lags, T-lags)

    Y_t = Y[lags:]  # Current Y values (T-lags,)

    # Conditional covariance given Y_past (for AIS)
    Beta_Y = np.linalg.lstsq(Y_past.T, Y_t, rcond=None)[0]
    Y_residual = Y_t - Beta_Y @ Y_past
    cov_Y_cond = np.var(Y_residual)

    # Conditional covariance given X_past and Y_past (for DI)
    XY_past = np.vstack((X_past, Y_past))  # Combine lagged X and Y (n_vars * lags + lags, T-lags)
    Beta_XY = np.linalg.lstsq(XY_past.T, Y_t, rcond=None)[0]
    Y_residual_joint = Y_t - Beta_XY @ XY_past
    cov_Y_joint = np.var(Y_residual_joint)

    return cov_Y_cond, cov_Y_joint


def compute_da_and_di(Y, X, lags=1):
    """
    Compute Dynamical Autonomy (DA) and Dynamical Independence (DI).
    
    Parameters:
    Y (numpy.ndarray): Macroscopic process time series (1xT)
    X (numpy.ndarray): Microscopic process time series (n_vars x T)
    lags (int): Number of past lags to consider
    
    Returns:
    tuple: DA and DI values
    """
    cov_Y_cond, cov_Y_joint = compute_covariances(Y, X, lags)
    DI = np.log(cov_Y_cond / cov_Y_joint)
    DA = -np.log(cov_Y_cond)
    return DA, DI


def orthonormalize(L):
    """
    Orthonormalizes the columns of a given matrix using QR decomposition.

    Parameters:
    L (numpy.ndarray): Input matrix to be orthonormalized.

    Returns:
    numpy.ndarray: Orthonormalized matrix with the same dimensions as L.
    """
    Q, _ = np.linalg.qr(L)
    return Q


def compute_dd_gradient(L, H):
    """
    Compute the gradient of spectral dynamical dependence (DD).

    Parameters:
    L (numpy.ndarray): Orthonormal subspace basis (n x m).
    H (numpy.ndarray): Transfer function (n x n x h), where `h` is the number of frequency points.

    Returns:
    tuple:
        - G (numpy.ndarray): Gradient matrix of the same shape as L.
        - mG (float): Magnitude of the gradient.
    """
    n, m, h = H.shape[0], L.shape[1], H.shape[2]
    g = np.zeros((n, m, h), dtype=np.float64)

    for k in range(h):
        Hk = H[:, :, k]
        HLk = Hk.T @ L
        g[:, :, k] = np.real((Hk @ HLk) @ np.linalg.inv(HLk.T @ HLk))  # Gradient/2

    # Integrate frequency-domain derivative (trapezoidal rule)
    G = np.sum(g[:, :, :-1] + g[:, :, 1:], axis=2) / (h - 1) - 2 * L

    mG = np.sqrt(np.sum(G**2))  # Magnitude of gradient
    return G, mG


def compute_dd(L, H):
    """
    Compute the spectral dynamical dependence (DD).

    Parameters:
    L (numpy.ndarray): Orthonormal subspace basis (n x m).
    H (numpy.ndarray): Transfer function (n x n x h), where `h` is the number of frequency points.

    Returns:
    tuple:
        - D (float): Dynamical dependence value.
        - d (numpy.ndarray): Frequency-dependent dynamical dependence values (h,).
    """
    h = H.shape[2]
    d = np.zeros(h, dtype=np.float64)

    for k in range(h):
        Qk = H[:, :, k].T @ L
        d[k] = np.sum(np.log(np.diag(cholesky(Qk.T @ Qk))))

    D = np.sum(d[:-1] + d[1:]) / (h - 1)  # Trapezoidal integration
    return D, d


def optimize_dds(H, L0, max_iters=1000, gdsig0=0.1, gdls=(2.0, 0.5), tol=(1e-6, 1e-6, 1e-6), record_history=False):
    """
    Gradient-based optimization of dynamical dependence to find the optimal coarse-graining matrix.

    Parameters:
    H (numpy.ndarray): Transfer function (n x n x h).
    L0 (numpy.ndarray): Initial orthonormal coarse-graining matrix (n x m).
    max_iters (int): Maximum number of iterations.
    gdsig0 (float): Initial gradient descent step size.
    gdls (float or tuple): Learning rate factors (increase, decrease).
    tol (float or tuple): Convergence tolerances (step size, DD value, gradient norm).
    record_history (bool): Whether to record optimization history.

    Returns:
    tuple:
        - dd (float): Optimized dynamical dependence value.
        - L (numpy.ndarray): Optimal coarse-graining matrix (n x m).
        - converged (int): Convergence status (0: not converged, 1: step size, 2: DD value, 3: gradient norm).
        - sig (float): Final step size.
        - iters (int): Number of iterations performed.
        - history (numpy.ndarray): Optimization history (optional).
    """
    # Initialize parameters
    if isinstance(gdls, (float, int)):
        increase_factor, decrease_factor = gdls, 1 / gdls
    else:
        increase_factor, decrease_factor = gdls

    if isinstance(tol, (float, int)):
        tol_sig, tol_dd, tol_grad = tol, tol, tol
    else:
        tol_sig, tol_dd, tol_grad = tol

    # Initial DD calculation
    L = L0.copy()
    dd, _ = compute_dd(L, H)
    
    # Initialize variables
    sig = gdsig0
    converged = 0
    
    # Record history if needed
    if record_history:
        history = np.zeros((max_iters + 1, 2))
        history[0, 0] = dd
        history[0, 1] = 0
    
    # Optimization loop
    for i in range(max_iters):
        # Compute gradient
        G, mG = compute_dd_gradient(L, H)
        
        # Check gradient convergence
        if mG < tol_grad:
            converged = 3
            break
        
        # Record gradient norm
        if record_history:
            history[i, 1] = mG
        
        # Line search
        new_L = orthonormalize(L + sig * G)
        new_dd, _ = compute_dd(new_L, H)
        
        # Adaptive step size
        if new_dd > dd:
            # Success - increase step size
            dd_change = new_dd - dd
            dd = new_dd
            L = new_L
            sig *= increase_factor
            
            # Check convergence criteria
            if sig < tol_sig:
                converged = 1
                break
            if dd_change < tol_dd:
                converged = 2
                break
        else:
            # Failure - decrease step size
            sig *= decrease_factor
            
            # Check convergence criterion
            if sig < tol_sig:
                converged = 1
                break
        
        # Record history
        if record_history:
            history[i + 1, 0] = dd
    
    # Prepare return values
    iters = i + 1
    if record_history:
        history = history[:iters + 1]
        return dd, L, converged, sig, iters, history
    else:
        return dd, L, converged, sig, iters 