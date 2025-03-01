"""
Utility mathematical functions for dynamical autonomy calculations.
"""

import numpy as np


def corr_rand(n, g=None, vexp=2, tol=np.sqrt(np.finfo(float).eps), maxretries=1000):
    """
    Generate a random correlation matrix with a given multi-information.
    
    Parameters:
    n (int): Dimensionality of the correlation matrix
    g (float, optional): Desired multi-information. If None, a random value is chosen.
    vexp (float, optional): Variance exponent for eigenvalues
    tol (float, optional): Tolerance for multi-information accuracy
    maxretries (int, optional): Maximum number of retries
    
    Returns:
    numpy.ndarray: Random correlation matrix
    """
    # Inner functions
    def random_orthogonal_matrix(n):
        """Generate a random orthogonal matrix using QR decomposition."""
        X = np.random.randn(n, n)
        Q, _ = np.linalg.qr(X)
        return Q
    
    def multi_information(V, v):
        """Calculate multi-information of covariance matrix V with variances v."""
        n = V.shape[0]
        # Compute log determinant of covariance
        logdet_V = np.log(np.linalg.det(V))
        # Compute log determinant of diagonal covariance (product of variances)
        logdet_diag = np.sum(np.log(v))
        # Multi-information is the KL divergence between joint and independent distributions
        return 0.5 * (logdet_diag - logdet_V)
    
    # Choose random multi-information if not specified
    if g is None:
        g = 0.5 * np.random.rand() * n  # Random multi-information
    
    # Generate random correlation matrix
    retry = 0
    while retry < maxretries:
        # Generate random orthogonal matrix
        Q = random_orthogonal_matrix(n)
        
        # Generate random eigenvalues with desired distribution
        lmax = 1.0
        lmin = 1.0 / n  # Ensure minimum eigenvalue prevents singularity
        v = lmin + (lmax - lmin) * np.random.rand(n) ** vexp
        v = v / np.mean(v)  # Normalize to have mean 1
        
        # Form covariance matrix from eigenvalues and eigenvectors
        V = Q @ np.diag(v) @ Q.T
        
        # Convert to correlation matrix
        d = np.sqrt(np.diag(V))
        D_inv = np.diag(1.0 / d)
        C = D_inv @ V @ D_inv
        
        # Calculate actual multi-information
        ga = multi_information(C, np.ones(n))
        
        # Check if multi-information is close enough to desired value
        if np.abs(ga - g) < tol:
            return C
        
        retry += 1
    
    # If we reach here, we couldn't achieve the desired multi-information
    # Return the last correlation matrix generated
    return C 