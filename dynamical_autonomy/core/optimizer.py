"""
Optimization functions for dynamical autonomy metrics.
"""

import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor
from .dynamical_autonomy import compute_dd, compute_dd_gradient, orthonormalize


def opt_gd_dds_mruns(H, L0, niters, gdes, gdsig0, gdls, gdtol, hist=False, parallel=True):
    """
    Run multiple gradient descent optimizations with different initializations.

    Parameters:
    H (numpy.ndarray): Transfer function (n x n x h).
    L0 (numpy.ndarray): Initial coarse-graining matrix (n x m).
    niters (int): Number of random initializations.
    gdes (int): Maximum number of gradient descent iterations.
    gdsig0 (float): Initial step size.
    gdls (tuple): Learning rate factors (increase, decrease).
    gdtol (tuple): Convergence tolerances.
    hist (bool): Whether to record optimization history.
    parallel (bool): Whether to run in parallel.

    Returns:
    tuple:
        - DD (list): List of optimized dynamical dependence values.
        - L (list): List of optimized coarse-graining matrices.
        - conv (list): List of convergence status.
        - sig (list): List of final step sizes.
        - iters (list): List of iteration numbers.
        - history (list): List of optimization histories (optional).
    """
    n, m = L0.shape
    
    def single_run(k):
        """
        Run a single optimization with random initialization.
        
        Parameters:
        k (int): Run index.
        
        Returns:
        tuple: Optimization results.
        """
        # Generate random initialization
        if k == 0:
            L = L0.copy()  # Use provided initial guess for the first run
        else:
            # Generate random orthonormal matrix
            L = np.random.randn(n, m)
            L = orthonormalize(L)
        
        # Run optimization
        if hist:
            dd, Lopt, conv, sig, iters, history = optimize_dds(
                H, L, max_iters=gdes, gdsig0=gdsig0, gdls=gdls, tol=gdtol, record_history=True
            )
            return dd, Lopt, conv, sig, iters, history
        else:
            dd, Lopt, conv, sig, iters = optimize_dds(
                H, L, max_iters=gdes, gdsig0=gdsig0, gdls=gdls, tol=gdtol, record_history=False
            )
            return dd, Lopt, conv, sig, iters
    
    # Run optimizations in parallel or sequential
    if parallel:
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(single_run, range(niters)))
    else:
        results = [single_run(k) for k in range(niters)]
    
    # Unpack results
    if hist:
        DD, L, conv, sig, iters, history = zip(*results)
        return DD, L, conv, sig, iters, history
    else:
        DD, L, conv, sig, iters = zip(*results)
        return DD, L, conv, sig, iters


def optimize_steepest_descent(H, L0, manifold_type="Grassmann"):
    """
    Optimize using steepest descent on the specified manifold.
    
    Parameters:
    H (numpy.ndarray): Transfer function.
    L0 (numpy.ndarray): Initial orthonormal subspace basis.
    manifold_type (str): Type of manifold ('Grassmann' or 'Stiefel').
    
    Returns:
    tuple: Optimized DD value and coarse-graining matrix.
    """
    try:
        import pymanopt
        from pymanopt.manifolds import Grassmann, Stiefel
        from pymanopt.optimizers import SteepestDescent
    except ImportError:
        raise ImportError("This function requires pymanopt package. Please install with 'pip install pymanopt'.")
    
    n, m = L0.shape
    
    # Choose manifold type
    if manifold_type == "Grassmann":
        manifold = Grassmann(n, m)
    elif manifold_type == "Stiefel":
        manifold = Stiefel(n, m)
    else:
        raise ValueError(f"Invalid manifold type: {manifold_type}")
    
    # Define cost and gradient functions
    def cost(L):
        dd, _ = compute_dd(L, H)
        return -dd  # Negative because we want to maximize DD
    
    def gradient(L):
        G, _ = compute_dd_gradient(L, H)
        return -G  # Negative because we want to maximize DD
    
    # Create optimization problem
    problem = pymanopt.Problem(manifold, cost, gradient=gradient)
    
    # Run optimization
    optimizer = SteepestDescent()
    Lopt = optimizer.run(problem, initial_point=L0).point
    
    # Compute final DD value
    dd_opt, _ = compute_dd(Lopt, H)
    
    return dd_opt, Lopt


def optimize_conjugate_gradient(H, L0, manifold_type="Grassmann"):
    """
    Optimize using conjugate gradient on the specified manifold.
    
    Parameters:
    H (numpy.ndarray): Transfer function.
    L0 (numpy.ndarray): Initial orthonormal subspace basis.
    manifold_type (str): Type of manifold ('Grassmann' or 'Stiefel').
    
    Returns:
    tuple: Optimized DD value and coarse-graining matrix.
    """
    try:
        import pymanopt
        from pymanopt.manifolds import Grassmann, Stiefel
        from pymanopt.optimizers import ConjugateGradient
    except ImportError:
        raise ImportError("This function requires pymanopt package. Please install with 'pip install pymanopt'.")
    
    n, m = L0.shape
    
    # Choose manifold type
    if manifold_type == "Grassmann":
        manifold = Grassmann(n, m)
    elif manifold_type == "Stiefel":
        manifold = Stiefel(n, m)
    else:
        raise ValueError(f"Invalid manifold type: {manifold_type}")
    
    # Define cost and gradient functions
    def cost(L):
        dd, _ = compute_dd(L, H)
        return -dd  # Negative because we want to maximize DD
    
    def gradient(L):
        G, _ = compute_dd_gradient(L, H)
        return -G  # Negative because we want to maximize DD
    
    # Create optimization problem
    problem = pymanopt.Problem(manifold, cost, gradient=gradient)
    
    # Run optimization
    optimizer = ConjugateGradient()
    Lopt = optimizer.run(problem, initial_point=L0).point
    
    # Compute final DD value
    dd_opt, _ = compute_dd(Lopt, H)
    
    return dd_opt, Lopt


def optimize_trust_regions(H, L0, manifold_type="Grassmann"):
    """
    Optimize using trust regions on the specified manifold.
    
    Parameters:
    H (numpy.ndarray): Transfer function.
    L0 (numpy.ndarray): Initial orthonormal subspace basis.
    manifold_type (str): Type of manifold ('Grassmann' or 'Stiefel').
    
    Returns:
    tuple: Optimized DD value and coarse-graining matrix.
    """
    try:
        import pymanopt
        from pymanopt.manifolds import Grassmann, Stiefel
        from pymanopt.optimizers import TrustRegions
    except ImportError:
        raise ImportError("This function requires pymanopt package. Please install with 'pip install pymanopt'.")
    
    n, m = L0.shape
    
    # Choose manifold type
    if manifold_type == "Grassmann":
        manifold = Grassmann(n, m)
    elif manifold_type == "Stiefel":
        manifold = Stiefel(n, m)
    else:
        raise ValueError(f"Invalid manifold type: {manifold_type}")
    
    # Define cost and gradient functions
    def cost(L):
        dd, _ = compute_dd(L, H)
        return -dd  # Negative because we want to maximize DD
    
    def gradient(L):
        G, _ = compute_dd_gradient(L, H)
        return -G  # Negative because we want to maximize DD
    
    # Create optimization problem
    problem = pymanopt.Problem(manifold, cost, gradient=gradient)
    
    # Run optimization
    optimizer = TrustRegions()
    Lopt = optimizer.run(problem, initial_point=L0).point
    
    # Compute final DD value
    dd_opt, _ = compute_dd(Lopt, H)
    
    return dd_opt, Lopt


def project_to_stiefel(L):
    """
    Project a matrix to the Stiefel manifold using QR decomposition.
    
    Parameters:
    L (numpy.ndarray): Input matrix.
    
    Returns:
    numpy.ndarray: Projected matrix on Stiefel manifold.
    """
    return orthonormalize(L)


def optimize_pso(H, L0, manifold_type="Grassmann"):
    """
    Optimize using particle swarm optimization with projection to manifold.
    
    Parameters:
    H (numpy.ndarray): Transfer function.
    L0 (numpy.ndarray): Initial orthonormal subspace basis.
    manifold_type (str): Type of manifold ('Grassmann' or 'Stiefel').
    
    Returns:
    tuple: Optimized DD value and coarse-graining matrix.
    """
    try:
        import pyswarms as ps
    except ImportError:
        raise ImportError("This function requires pyswarms package. Please install with 'pip install pyswarms'.")
    
    n, m = L0.shape
    
    # Define cost function
    def cost(L_flat):
        # Reshape to matrix form
        L_reshaped = np.reshape(L_flat, (len(L_flat) // n // m, n, m))
        n_particles = L_reshaped.shape[0]
        costs = np.zeros(n_particles)
        
        # Compute cost for each particle
        for i in range(n_particles):
            # Project to Stiefel manifold
            L_proj = project_to_stiefel(L_reshaped[i])
            dd, _ = compute_dd(L_proj, H)
            costs[i] = -dd  # Negative because PSO minimizes
        
        return costs
    
    # Initialize PSO
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    n_particles = 10
    dimensions = n * m
    initial_pos = L0.flatten()[np.newaxis, :]
    
    # Create optimizer
    optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=dimensions, options=options, init_pos=initial_pos)
    
    # Run optimization
    best_cost, best_pos = optimizer.optimize(cost, iters=100)
    
    # Reshape and project best position
    Lopt = project_to_stiefel(np.reshape(best_pos, (n, m)))
    
    # Compute final DD value
    dd_opt, _ = compute_dd(Lopt, H)
    
    return dd_opt, Lopt


def opt_gd_dds_mruns_all_optimizers(H, L0, niters, gdes, gdsig0, gdls, gdtol, hist=False, parallel=True):
    """
    Run multiple optimizations with different optimizers.
    
    Parameters:
    H (numpy.ndarray): Transfer function.
    L0 (numpy.ndarray): Initial orthonormal subspace basis.
    niters (int): Number of random initializations.
    gdes (int): Maximum number of gradient descent iterations.
    gdsig0 (float): Initial step size.
    gdls (tuple): Learning rate factors.
    gdtol (tuple): Convergence tolerances.
    hist (bool): Whether to record optimization history.
    parallel (bool): Whether to run in parallel.
    
    Returns:
    tuple: Results from different optimizers.
    """
    n, m = L0.shape
    results = {}
    
    # Custom optimizer
    print("Running custom gradient descent...")
    gd_results = opt_gd_dds_mruns(H, L0, niters, gdes, gdsig0, gdls, gdtol, hist, parallel)
    results['custom_gd'] = gd_results
    
    def single_run(k):
        """
        Run a single optimization with all optimizers.
        
        Parameters:
        k (int): Run index.
        
        Returns:
        dict: Results from different optimizers.
        """
        run_results = {}
        
        # Generate random initialization
        if k == 0:
            L = L0.copy()
        else:
            L = np.random.randn(n, m)
            L = orthonormalize(L)
        
        try:
            print(f"Run {k}: Steepest descent (Grassmann)...")
            sd_dd, sd_L = optimize_steepest_descent(H, L, "Grassmann")
            run_results['sd_grass'] = (sd_dd, sd_L)
        except Exception as e:
            print(f"Error in steepest descent (Grassmann): {e}")
        
        try:
            print(f"Run {k}: Steepest descent (Stiefel)...")
            sd_dd_stiefel, sd_L_stiefel = optimize_steepest_descent(H, L, "Stiefel")
            run_results['sd_stiefel'] = (sd_dd_stiefel, sd_L_stiefel)
        except Exception as e:
            print(f"Error in steepest descent (Stiefel): {e}")
        
        try:
            print(f"Run {k}: Conjugate gradient...")
            cg_dd, cg_L = optimize_conjugate_gradient(H, L)
            run_results['cg'] = (cg_dd, cg_L)
        except Exception as e:
            print(f"Error in conjugate gradient: {e}")
        
        try:
            print(f"Run {k}: Trust regions...")
            tr_dd, tr_L = optimize_trust_regions(H, L)
            run_results['tr'] = (tr_dd, tr_L)
        except Exception as e:
            print(f"Error in trust regions: {e}")
        
        try:
            print(f"Run {k}: PSO...")
            pso_dd, pso_L = optimize_pso(H, L)
            run_results['pso'] = (pso_dd, pso_L)
        except Exception as e:
            print(f"Error in PSO: {e}")
        
        return run_results
    
    # Run alternative optimizers
    if parallel:
        with ProcessPoolExecutor() as executor:
            alt_results = list(executor.map(single_run, range(niters)))
    else:
        alt_results = [single_run(k) for k in range(niters)]
    
    # Reorganize results
    for optimizer in ['sd_grass', 'sd_stiefel', 'cg', 'tr', 'pso']:
        optimizer_results = []
        for run_result in alt_results:
            if optimizer in run_result:
                optimizer_results.append(run_result[optimizer])
        
        if optimizer_results:
            results[optimizer] = list(zip(*optimizer_results))
    
    return results 