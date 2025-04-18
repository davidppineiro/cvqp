"""
Tests for the CVQP solver.
"""

import sys
import os
# Add the parent directory to the path so we can import the cvqp package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import scipy as sp
import cvxpy as cp

from cvqp import CVQP, CVQPParams, CVQPConfig


def verify_solutions(cvxpy_sol, cvqp_sol, cvxpy_obj, cvqp_obj, test_name=""):
    """Verify that CVXPY and CVQP solutions match within tolerance.
    
    Args:
        cvxpy_sol: Solution vector from CVXPY
        cvqp_sol: Solution vector from CVQP
        cvxpy_obj: Objective value from CVXPY
        cvqp_obj: Objective value from CVQP
        test_name: Name of the test for display purposes
    """
    # Check objective values
    rel_gap = abs(cvxpy_obj - cvqp_obj) / (1 + abs(cvxpy_obj))
    print(f"{test_name} CVXPY objective: {cvxpy_obj:.6f}")
    print(f"{test_name} CVQP objective: {cvqp_obj:.6f}")
    print(f"{test_name} Relative objective gap: {rel_gap:.6f}")
    
    # Check if solutions are close (allowing for some tolerance)
    assert rel_gap <= 1e-2, f"Objective values differ: CVXPY={cvxpy_obj}, CVQP={cvqp_obj}"
    
    # Check if solutions are similar
    sol_diff = np.linalg.norm(cvxpy_sol - cvqp_sol)
    sol_norm = np.linalg.norm(cvxpy_sol)
    rel_sol_diff = sol_diff / (1.0 + sol_norm)
    print(f"{test_name} Relative solution difference: {rel_sol_diff:.6f}")
    
    assert rel_sol_diff <= 1e-2, "Solutions differ significantly"


def test_projection():
    """Test the projection function and compare with CVXPY solution."""
    from cvqp import proj_sum_largest
    
    # Create a sample vector where sum of 2 largest elements (6 + 5 = 11) exceeds alpha = 7
    z = np.array([6.0, 2.0, 5.0, 4.0, 1.0])
    k = 2
    alpha = 7.0

    # Apply projection using our implementation
    result = proj_sum_largest(z, k, alpha)

    # Check feasibility
    assert sum(sorted(result, reverse=True)[:k]) <= alpha + 1e-8
    
    # Solve the same projection problem using CVXPY with sum_largest atom
    x = cp.Variable(z.shape)
    objective = cp.Minimize(cp.sum_squares(x - z))
    constraints = [cp.sum_largest(x, k) <= alpha]
    
    # Solve with CVXPY
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=False)
    
    # Check if solutions are close
    cvxpy_proj = x.value
    our_proj = result
    
    # Check if the projections are close
    proj_diff = np.linalg.norm(cvxpy_proj - our_proj)
    print(f"Projection difference: {proj_diff:.6f}")
    
    # Check if solutions are close (allowing for some tolerance)
    assert proj_diff <= 1e-3, "Projections differ significantly"
    
    # Verify that CVXPY solution satisfies the constraint
    cvxpy_sorted = np.sort(cvxpy_proj)[::-1]
    cvxpy_sum_k_largest = np.sum(cvxpy_sorted[:k])
    print(f"CVXPY sum of {k} largest elements: {cvxpy_sum_k_largest:.6f} (limit: {alpha})")
    assert cvxpy_sum_k_largest <= alpha + 1e-6, "CVXPY solution violates constraint"
    
    
def test_cvqp_dense():
    """Test the CVQP solver with dense matrices and compare with CVXPY."""
    m, d = 100, 10
    np.random.seed(0)
    
    # Generate parameters with dense matrices
    P = np.eye(d)  # Dense identity matrix
    q = np.ones(d) * -0.1 + np.random.randn(d) * 0.05
    A = np.random.randn(m, d) * 0.2 + 0.1
    B = np.eye(d)
    l = -np.ones(d)
    u = np.ones(d)
    beta = 0.9
    kappa = 0.1
    
    params = CVQPParams(
        P=P, q=q, A=A, B=B, l=l, u=u, beta=beta, kappa=kappa
    )
    
    # Solve with CVQP solver
    cvqp = CVQP(params, CVQPConfig(verbose=False, max_iter=1000))
    results = cvqp.solve()
    
    # Check that result is approximately feasible
    assert results.problem_status == "optimal"
    assert all(results.x >= params.l - 1e-6)
    assert all(results.x <= params.u + 1e-6)
    
    # Check CVaR constraint
    sorted_vals = np.sort(params.A @ results.x)
    k = int(m * (1 - params.beta))
    cvar = np.mean(sorted_vals[-k:])
    assert cvar <= params.kappa + 1e-3
    
    # Compare with CVXPY solution using cvar atom
    x_cvxpy = cp.Variable(d)
    
    # Objective
    objective = cp.Minimize(0.5 * cp.quad_form(x_cvxpy, P) + q @ x_cvxpy)
    
    # Constraints using CVXPY's cvar atom
    constraints = [
        cp.cvar(A @ x_cvxpy, beta) <= kappa,  # CVaR constraint with beta
        l <= B @ x_cvxpy,                     # Lower bounds
        B @ x_cvxpy <= u                      # Upper bounds
    ]
    
    # Solve with CVXPY
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=False)
    
    # Check if CVXPY found a solution
    if prob.status == cp.OPTIMAL:
        # Calculate objective value for CVXPY solution
        cvxpy_obj = 0.5 * np.dot(x_cvxpy.value, P @ x_cvxpy.value) + q @ x_cvxpy.value
        cvqp_obj = results.objval[-1]
        
        # Verify that solutions match
        verify_solutions(x_cvxpy.value, results.x, cvxpy_obj, cvqp_obj, "Dense:")


def test_cvqp_sparse():
    """Test the CVQP solver with sparse matrices and compare with CVXPY."""
    m, d = 100, 10
    np.random.seed(0)
    
    # Generate parameters with sparse matrices
    P = sp.sparse.eye(d)  # Sparse identity matrix
    q = np.ones(d) * -0.1 + np.random.randn(d) * 0.05
    A = np.random.randn(m, d) * 0.2 + 0.1
    B = sp.sparse.eye(d)
    l = -np.ones(d)
    u = np.ones(d)
    beta = 0.9
    kappa = 0.1
    
    params = CVQPParams(
        P=P, q=q, A=A, B=B, l=l, u=u, beta=beta, kappa=kappa
    )
    
    # Solve with CVQP solver
    cvqp = CVQP(params, CVQPConfig(verbose=False, max_iter=1000))
    results = cvqp.solve()
    
    # Check that result is approximately feasible
    assert results.problem_status == "optimal"
    assert all(results.x >= params.l - 1e-6)
    assert all(results.x <= params.u + 1e-6)
    
    # Check CVaR constraint
    sorted_vals = np.sort(params.A @ results.x)
    k = int(m * (1 - params.beta))
    cvar = np.mean(sorted_vals[-k:])
    assert cvar <= params.kappa + 1e-3
    
    # Compare with CVXPY solution
    x_cvxpy = cp.Variable(d)
    
    # Objective - Convert sparse P to dense for CVXPY
    P_dense = P.toarray() if sp.sparse.issparse(P) else P
    objective = cp.Minimize(0.5 * cp.quad_form(x_cvxpy, P_dense) + q @ x_cvxpy)
    
    # Convert sparse B to dense for constraints
    B_dense = B.toarray() if sp.sparse.issparse(B) else B
    
    # Constraints using CVXPY's cvar atom
    constraints = [
        cp.cvar(A @ x_cvxpy, beta) <= kappa,  # CVaR constraint
        l <= B_dense @ x_cvxpy,               # Lower bounds
        B_dense @ x_cvxpy <= u                # Upper bounds
    ]
    
    # Solve with CVXPY
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=False)
    
    # Check if CVXPY found a solution
    if prob.status == cp.OPTIMAL:
        # Calculate objective value for CVXPY solution
        P_dense = P.toarray() if sp.sparse.issparse(P) else P
        cvxpy_obj = 0.5 * np.dot(x_cvxpy.value, P_dense @ x_cvxpy.value) + q @ x_cvxpy.value
        cvqp_obj = results.objval[-1]
        
        # Verify that solutions match
        verify_solutions(x_cvxpy.value, results.x, cvxpy_obj, cvqp_obj, "Sparse:")


def test_cvqp_linear():
    """Test CVQP with a linear cost (P=None) and compare with CVXPY."""
    m, d = 80, 8
    np.random.seed(1)
    
    # Generate parameters with a linear objective (no quadratic term)
    P = None
    q = np.ones(d) * -0.2 + np.random.randn(d) * 0.05
    A = np.random.randn(m, d) * 0.15 + 0.1
    B = np.eye(d)
    l = -np.ones(d) * 0.8
    u = np.ones(d) * 0.8
    beta = 0.85
    kappa = 0.15
    
    params = CVQPParams(
        P=P, q=q, A=A, B=B, l=l, u=u, beta=beta, kappa=kappa
    )
    
    # Solve with CVQP solver
    cvqp = CVQP(params, CVQPConfig(verbose=False, max_iter=2000))
    results = cvqp.solve()
    
    # Check that result is approximately feasible
    assert results.problem_status == "optimal"
    assert all(results.x >= params.l - 1e-6)
    assert all(results.x <= params.u + 1e-6)
    
    # Check CVaR constraint
    sorted_vals = np.sort(params.A @ results.x)
    k = int(m * (1 - params.beta))
    cvar = np.mean(sorted_vals[-k:])
    assert cvar <= params.kappa + 1e-3
    
    # Compare with CVXPY solution
    x_cvxpy = cp.Variable(d)
    
    # Linear objective
    objective = cp.Minimize(q @ x_cvxpy)
    
    # Constraints using CVXPY's cvar atom
    constraints = [
        cp.cvar(A @ x_cvxpy, beta) <= kappa,  # CVaR constraint
        l <= B @ x_cvxpy,                     # Lower bounds
        B @ x_cvxpy <= u                      # Upper bounds
    ]
    
    # Solve with CVXPY
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=False)
    
    # Check if CVXPY found a solution
    if prob.status == cp.OPTIMAL:
        # Calculate objective value for CVXPY solution
        cvxpy_obj = q @ x_cvxpy.value
        cvqp_obj = results.objval[-1]
        
        # Verify that solutions match
        verify_solutions(x_cvxpy.value, results.x, cvxpy_obj, cvqp_obj, "Linear:")


if __name__ == "__main__":
    test_projection()
    test_cvqp_dense()
    test_cvqp_sparse()
    test_cvqp_linear()
    print("All tests passed!")