"""Optimal observer synthesis."""

from typing import Any, Dict, Optional, Tuple

import control
import cvxpy
import numpy as np
import scipy.linalg


def mixed_H2_Hinf(
    P: control.StateSpace,
    n_z2: int,
    n_w2: int,
    n_y: int,
    n_u: int,
    initial_guess: Optional[float] = None,
    bisection_eps: float = 1e-4,
    max_iterations: int = 100,
    sdp_eps: float = 1e-6,
    sdp_strictness: float = 1e-5,
    cvxpy_verbose: bool = False,
    cost_scale: float = 1,
) -> Tuple[control.StateSpace, Dict[str, Any]]:
    """Mixed H2-Hinf synthesis."""
    info = {
        "status": None,
        "nu": None,
        "bisection_iterations": None,
        "cond(M_left)": None,
        "cond(M_right)": None,
    }
    # Get dimensions
    n_x = P.nstates
    n_w = P.ninputs - n_u
    n_z = P.noutputs - n_y
    n_w1 = n_w - n_w2
    n_z1 = n_z - n_z2
    # Divide state-space matrices
    A = P.A
    B_1 = P.B[:, :n_w]
    B_2 = P.B[:, n_w:]
    C_1 = P.C[:n_z, :]
    C_2 = P.C[n_z:, :]
    D_11 = P.D[:n_z, :n_w]
    D_12 = P.D[:n_z, n_w:]
    D_21 = P.D[n_z:, :n_w]
    D_22 = P.D[n_z:, n_w:]
    # Subdivide state-space matrices
    B_1_1 = B_1[:, :n_w1]
    B_1_2 = B_1[:, n_w1:]
    C_1_1 = C_1[:n_z1, :]
    C_1_2 = C_1[n_z1:, :]
    D_11_11 = D_11[:n_z1, :n_w1]
    D_11_12 = D_11[:n_z1, n_w1:]
    D_11_21 = D_11[n_z1:, :n_w1]
    D_11_22 = D_11[n_z1:, n_w1:]
    D_12_1 = D_12[:n_z1, :]
    D_12_2 = D_12[n_z1:, :]
    D_21_1 = D_21[:, :n_w1]
    D_21_2 = D_21[:, n_w1:]
    # Define variables
    A_n = cvxpy.Variable((n_x, n_x), name="A_n")
    B_n = cvxpy.Variable((n_x, n_y), name="B_n")
    C_n = cvxpy.Variable((n_u, n_x), name="C_n")
    D_n = cvxpy.Variable((n_u, n_y), name="D_n")
    X_1 = cvxpy.Variable((n_x, n_x), name="X_1", symmetric=True)
    Y_1 = cvxpy.Variable((n_x, n_x), name="Y_1", symmetric=True)
    Z = cvxpy.Variable((n_z1, n_z1), name="Z", symmetric=True)
    gamma = 1 - sdp_strictness
    # Define block matrices
    block_1 = cvxpy.bmat(
        [
            [
                X_1,
                np.eye(X_1.shape[0]),
                X_1 @ A + B_n @ C_2,
                A_n,
                X_1 @ B_1_1 + B_n @ D_21_1,
            ],
            [
                (np.eye(X_1.shape[0])).T,
                Y_1,
                A + B_2 @ D_n @ C_2,
                A @ Y_1 + B_2 @ C_n,
                B_1_1 + B_2 @ D_n @ D_21_1,
            ],
            [
                (X_1 @ A + B_n @ C_2).T,
                (A + B_2 @ D_n @ C_2).T,
                X_1,
                np.eye(X_1.shape[0]),
                np.zeros((X_1.shape[0], B_1_1.shape[1])),
            ],
            [
                (A_n).T,
                (A @ Y_1 + B_2 @ C_n).T,
                (np.eye(X_1.shape[0])).T,
                Y_1,
                np.zeros((Y_1.shape[0], B_1_1.shape[1])),
            ],
            [
                (X_1 @ B_1_1 + B_n @ D_21_1).T,
                (B_1_1 + B_2 @ D_n @ D_21_1).T,
                (np.zeros((X_1.shape[0], B_1_1.shape[1]))).T,
                (np.zeros((Y_1.shape[0], B_1_1.shape[1]))).T,
                np.eye(B_1_1.shape[1]),
            ],
        ]
    )
    block_2 = cvxpy.bmat(
        [
            [
                X_1,
                np.eye(X_1.shape[0]),
                X_1 @ A + B_n @ C_2,
                A_n,
                X_1 @ B_1_2 + B_n @ D_21_2,
                np.zeros((X_1.shape[0], C_1_2.shape[0])),
            ],
            [
                (np.eye(X_1.shape[0])).T,
                Y_1,
                A + B_2 @ D_n @ C_2,
                A @ Y_1 + B_2 @ C_n,
                B_1_2 + B_2 @ D_n @ D_21_2,
                np.zeros((Y_1.shape[0], C_1_2.shape[0])),
            ],
            [
                (X_1 @ A + B_n @ C_2).T,
                (A + B_2 @ D_n @ C_2).T,
                X_1,
                np.eye(X_1.shape[0]),
                np.zeros((X_1.shape[0], B_1_2.shape[1])),
                C_1_2.T + C_2.T @ D_n.T @ D_12_2.T,
            ],
            [
                (A_n).T,
                (A @ Y_1 + B_2 @ C_n).T,
                (np.eye(X_1.shape[0])).T,
                Y_1,
                np.zeros((Y_1.shape[0], B_1_2.shape[1])),
                Y_1 @ C_1_2.T + C_n.T @ D_12_2.T,
            ],
            [
                (X_1 @ B_1_2 + B_n @ D_21_2).T,
                (B_1_2 + B_2 @ D_n @ D_21_2).T,
                (np.zeros((X_1.shape[0], B_1_2.shape[1]))).T,
                (np.zeros((Y_1.shape[0], B_1_2.shape[1]))).T,
                cvxpy.multiply(gamma, np.eye(D_11_22.shape[1])),
                D_11_22.T + D_21_2.T @ D_n.T @ D_12_2.T,
            ],
            [
                (np.zeros((X_1.shape[0], C_1_2.shape[0]))).T,
                (np.zeros((Y_1.shape[0], C_1_2.shape[0]))).T,
                (C_1_2.T + C_2.T @ D_n.T @ D_12_2.T).T,
                (Y_1 @ C_1_2.T + C_n.T @ D_12_2.T).T,
                (D_11_22.T + D_21_2.T @ D_n.T @ D_12_2.T).T,
                cvxpy.multiply(gamma, np.eye(D_11_22.shape[0])),
            ],
        ]
    )
    block_3 = cvxpy.bmat(
        [
            [
                Z,
                C_1_1 + D_12_1 @ D_n @ C_2,
                C_1_1 @ Y_1 + D_12_1 @ C_n,
            ],
            [
                (C_1_1 + D_12_1 @ D_n @ C_2).T,
                X_1,
                np.eye(X_1.shape[0]),
            ],
            [
                (C_1_1 @ Y_1 + D_12_1 @ C_n).T,
                (np.eye(X_1.shape[0])).T,
                Y_1,
            ],
        ]
    )
    block_4 = cvxpy.bmat(
        [
            [X_1, np.eye(X_1.shape[0])],
            [np.eye(X_1.shape[0]), Y_1],
        ]
    )
    # Set nu (variable or parameter)
    if initial_guess is None:
        nu = cvxpy.Variable(1, name="nu", nonneg=True)
    else:
        nu = cvxpy.Parameter(1, name="nu")
    # Define constraints
    constraints = [
        X_1 >> sdp_strictness,
        Y_1 >> sdp_strictness,
        Z >> sdp_strictness,
        block_1 >> sdp_strictness,
        block_2 >> sdp_strictness,
        block_3 >> sdp_strictness,
        block_4 >> sdp_strictness,
        D_11_11 + D_12_1 @ D_n @ D_21_1 == 0,
        cvxpy.trace(Z) <= nu,
    ]
    # Solve problem
    if initial_guess is None:
        objective = cvxpy.Minimize(nu * cost_scale)
        problem = cvxpy.Problem(objective, constraints)
        solver_params = dict(
            solver="MOSEK",
            eps=sdp_eps,
            verbose=cvxpy_verbose,
        )
        try:
            result = problem.solve(**solver_params)
            if problem.status != "optimal":
                info["status"] = "failure, solution not optimal"
                return None, info
            info["status"] = "success"
            info["nu"] = [result]
        except cvxpy.error.SolverError:
            info["status"] = "failure, solver error"
            return None, info
    else:
        objective = cvxpy.Minimize(1)
        problem = cvxpy.Problem(objective, constraints)
        solver_params = dict(
            solver="MOSEK",
            eps=sdp_eps,
            verbose=cvxpy_verbose,
            warm_start=True,
        )
        # Make sure initial guess is high enough
        nu_high = initial_guess
        for i in range(max_iterations):
            try:
                # Update nu and solve optimization problem
                problem.param_dict["nu"].value = np.array([nu_high])
                result = problem.solve(**solver_params)
            except cvxpy.SolverError:
                nu_high *= 2
                continue
            if problem.status == "optimal":
                break
            else:
                nu_high *= 2
        else:
            # Could not find a high enough initial `nu` in `max_iterations`
            info["status"] = "failure, can't find upper bound on `nu`"
            return None, info
        # Start iteration
        nu_low = 0
        nus = []
        for i in range(max_iterations):
            nus.append((nu_high + nu_low) / 2)
            try:
                # Update nu and solve optimization problem
                problem.param_dict["nu"].value = np.array([nus[-1]])
                result = problem.solve(**solver_params)
            except cvxpy.SolverError:
                nu_low = nus[-1]
            if problem.status == "optimal":
                nu_high = nus[-1]
                # Only terminate if last iteration succeeded to make sure ``X``
                # has a value.
                if np.abs(nu_high - nu_low) < bisection_eps:
                    break
            else:
                nu_low = nus[-1]
        else:
            info["status"] = "failure, reached max bisection iterations"
            return None, info
        info["status"] = "success"
        info["bisection_iterations"] = i
        info["nu"] = nus
    # Extract controller
    Q, s, Vt = scipy.linalg.svd(
        np.eye(X_1.shape[0]) - X_1.value @ Y_1.value,
        full_matrices=True,
    )
    X_2 = Q @ np.diag(np.sqrt(s))
    Y_2 = Vt.T @ np.diag(np.sqrt(s))
    M_left = np.block(
        [
            [
                X_2,
                X_1.value @ B_2,
            ],
            [
                np.zeros((B_2.shape[1], X_2.shape[1])),
                np.eye(B_2.shape[1]),
            ],
        ]
    )
    M_middle = np.block(
        [
            [A_n.value, B_n.value],
            [C_n.value, D_n.value],
        ]
    ) - np.block(
        [
            [X_1.value @ A @ Y_1.value, np.zeros_like(B_n.value)],
            [np.zeros_like(C_n.value), np.zeros_like(D_n.value)],
        ]
    )
    M_right = np.block(
        [
            [
                Y_2.T,
                np.zeros((Y_2.T.shape[0], C_2.shape[0])),
            ],
            [
                C_2 @ Y_1.value,
                np.eye(C_2.shape[0]),
            ],
        ]
    )
    info["cond(M_left)"] = np.linalg.cond(M_left)
    info["cond(M_right)"] = np.linalg.cond(M_right)
    K_block = np.linalg.solve(M_right.T, np.linalg.solve(M_left, M_middle).T).T
    n_x_c = A_n.shape[0]
    A_K = K_block[:n_x_c, :n_x_c]
    B_K = K_block[:n_x_c, n_x_c:]
    C_K = K_block[n_x_c:, :n_x_c]
    D_K = K_block[n_x_c:, n_x_c:]
    K = control.StateSpace(
        A_K,
        B_K,
        C_K,
        D_K,
        dt=P.dt,
    )
    return K, info
