Python 3.13.12 (tags/v3.13.12:1cbe481, Feb  3 2026, 18:22:25) [MSC v.1944 64 bit (AMD64)] on win32
Enter "help" below or click "Help" above for more information.
>>> import numpy as np
... import pandas as pd
... 
... # =================================================================
... # STEP 4: SHARED UTILITIES (utils/numerical.py & utils/stats.py)
... # =================================================================
... 
... def make_grid(min_val, max_val, n, grid_type='linear'):
...     """Generates state space grids for assets or capital."""
...     if grid_type == 'linear':
...         return np.linspace(min_val, max_val, n)
...     elif grid_type == 'curved':
...         # Denser at the bottom to capture curvature in the Value Function
...         return np.linspace(0, 1, n)**2 * (max_val - min_val) + min_val
... 
... def validate_markov(P):
...     """Ensures transition matrix P is mathematically sound."""
...     if not np.allclose(P.sum(axis=1), 1):
...         raise ValueError("Rows of transition matrix must sum to 1.")
...     if np.any(P < 0):
...         raise ValueError("Probabilities cannot be negative.")
...     return True
... 
... def simulate_markov(P, states, T, seed=42):
...     """Simulates a path of shock values based on transition matrix P."""
...     np.random.seed(seed)
...     n_states = len(states)
...     out_indices = np.zeros(T, dtype=int)
...     
...     current_idx = 0 
...     for t in range(T):
...         out_indices[t] = current_idx
...         current_idx = np.random.choice(n_states, p=P[current_idx])
...     
...     return out_indices, states[out_indices]
... 
... def get_summary_stats(series, name="Variable"):
    """Calculates the 4 required moments for the assignment."""
    mu = np.mean(series)
    std = np.std(series)
    # Lag-1 Autocorrelation
    autocorr = np.corrcoef(series[:-1], series[1:])[0, 1] if std > 0 else 1.0
    # Coefficient of Variation
    cv = std / mu if mu != 0 else np.nan

    return {
        "Variable": name,
        "Mean": round(mu, 4),
        "Std Dev": round(std, 4),
        "Autocorr (L1)": round(autocorr, 4),
        "CV": round(cv, 4)
    }

# =================================================================
# STEP 5: GENERIC VFI ENGINE (solvers/vfi.py)
# =================================================================

def solve_vfi(u_matrix, P, beta, tol=1e-6, max_iter=1000):
    """
    Generic Stochastic VFI Solver using NumPy Broadcasting.
    
    Args:
        u_matrix: 3D Array [State_idx, Choice_idx, Shock_idx] 
                  Pre-computed utility for all combinations.
        P: Transition matrix [Shock_idx, Shock_next_idx]
        beta: Discount factor
    """
    n_states, n_choices, n_shocks = u_matrix.shape
    V = np.zeros((n_states, n_shocks)) # Initial guess: Value of zero
    
    for i in range(max_iter):
        # 1. Compute Expected Value: E[V | z]
        # EV shape: (n_choices, n_shocks) -> Choices become next period's states
        EV = V @ P.T 
        
        # 2. Bellman Equation: Utility(s, s', z) + beta * E[V(s', z') | z]
        # u_matrix: (S, C, Z)
        # EV[None, :, :]: (1, C, Z) - Broadcasts EV across all current states
        rhs = u_matrix + beta * EV[None, :, :]
        
        # 3. Maximize over choices (axis 1)
        V_new = np.max(rhs, axis=1)
        policy_indices = np.argmax(rhs, axis=1)
        
        # 4. Check convergence (Supremum Norm)
        error = np.max(np.abs(V_new - V))
        V = V_new
        
        if error < tol:
            return V, policy_indices, True, i
            
    return V, policy_indices, False, max_iter

# =================================================================
# VERIFICATION SUITE (Smoke Test)
# =================================================================

if __name__ == "__main__":
    print("--- Running Verification Suite ---")
    
    # Setup a dummy 'Cake Eating' problem
    beta = 0.95
    grid = make_grid(0.1, 5.0, 50)
    shocks = np.array([0.9, 1.1])
    P = np.array([[0.8, 0.2], [0.2, 0.8]])
    
    # Pre-compute u_matrix: u = log(s + z - s_next)
    n = len(grid)
    u_mat = np.full((n, n, 2), -1e10)
    for z_idx in range(2):
        for i in range(n):
            for j in range(n):
                c = grid[i] + shocks[z_idx] - grid[j]
                if c > 0:
                    u_mat[i, j, z_idx] = np.log(c)

    # Solve
    V_star, pol, conv, iters = solve_vfi(u_mat, P, beta)
    
    if conv:
        print(f"✅ SUCCESS: VFI converged in {iters} iterations.")
        # Calculate stats on a dummy simulation
        _, shock_path = simulate_markov(P, shocks, 100)
        stats = get_summary_stats(shock_path, "Exogenous Income")
        print(f"✅ STATS: Mean={stats['Mean']}, CV={stats['CV']}")
    else:
