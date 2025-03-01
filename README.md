# Dynamical Autonomy (DA)

A Python package for analyzing autonomy in dynamical systems. This package provides tools for measuring and analyzing dynamical autonomy, dynamical independence, and related metrics in time series data.

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/dynamical_autonomy.git
cd dynamical_autonomy

# Create and activate conda environment
conda create -n dynamical_autonomy python=3.8
conda activate dynamical_autonomy

# Install the package in development mode
pip install -e .
```

## Package Structure

The package is organized as follows:

- `dynamical_autonomy/` - Main package directory
  - `core/` - Core functionality for dynamical autonomy calculations
  - `models/` - Time series models (VAR, state-space models)
  - `utils/` - Utility functions for data preprocessing and numerical computations
  - `visualization/` - Visualization tools for networks and time series
  - `tests/` - Unit tests
  - `notebooks/` - Example Jupyter notebooks

## Jupyter Notebooks

The package includes the following tutorials and examples in the `notebooks/` directory:

1. `01_var_and_granger_causality.ipynb` - Introduction to Vector Autoregression (VAR) models and Granger causality analysis
2. `02_dynamical_autonomy_calculations.ipynb` - How to compute dynamical autonomy metrics for time series data
3. `03_optimizing_dynamical_dependence.ipynb` - Methods for optimizing dynamical dependence to find optimal coarse-graining matrices
4. `04_real_world_application.ipynb` - A practical application of dynamical autonomy analysis to economic data

To run these notebooks:

```bash
conda activate dynamical_autonomy
cd dynamical_autonomy
jupyter notebook
```

Then navigate to the `notebooks/` directory and open the desired notebook.

## Core Features

### VAR Models and Granger Causality

- Fit Vector Autoregression (VAR) models to multivariate time series
- Compute and visualize Granger causality matrices and networks
- Generate synthetic VAR data with specified causal structures

### Dynamical Autonomy Measures

- Compute dynamical autonomy (DA) and dynamical independence (DI)
- Calculate active information storage (AIS) and transfer entropy (TE)
- Perform partial information decomposition to quantify unique, redundant, and synergistic information

### Optimization Tools

- Optimize coarse-graining matrices to find maximally autonomous macro-variables
- Compare different optimization algorithms (gradient descent, conjugate gradient, trust regions, particle swarm)
- Visualize optimization progress and results

### Visualization

- Plot time series with optional fitted data
- Create Granger causality heatmaps
- Generate interactive causal network visualizations

## Usage Example

```python
import numpy as np
from dynamical_autonomy.models import generate_var_data
from dynamical_autonomy.core import compute_dynamical_autonomy

# Generate VAR model data
A = np.array([
    [0.7, 0.1, 0.0],
    [0.3, 0.5, 0.0],
    [0.0, 0.4, 0.6]
]).reshape(3, 3, 1)  # 3 variables, order 1
Q = np.eye(3) * 0.1   # Residual covariance matrix
n_obs = 1000          # Number of observations

# Generate data from the model
data = generate_var_data(A, Q, n_obs)

# Define parameters for dynamical autonomy calculation
C = np.eye(3)        # Observation matrix
R_X = np.zeros((3, 3))  # Observation noise covariance
L = np.array([[1, -1, 0]])  # Coarse-graining matrix

# Compute dynamical autonomy metrics
da_metrics = compute_dynamical_autonomy(A[:,:,0], Q, C, R_X, L)
print(f"Dynamical Autonomy: {da_metrics['DA']}")
```

## Requirements

- Python 3.7+
- NumPy
- SciPy
- Matplotlib
- statsmodels
- networkx
- pyvis
- seaborn
- pandas
- pytest

## License

Open source license.
