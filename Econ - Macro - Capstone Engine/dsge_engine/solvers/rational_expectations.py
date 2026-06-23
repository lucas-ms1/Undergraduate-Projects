from __future__ import annotations

import numpy as np
from scipy.linalg import ordqz


DEFAULT_JUMPS = ("c_o_hat", "i_hat", "q_hat", "pi_hat", "pi_w_hat")


def _count_explosive_roots(eigenvalues, tol):
    mask = np.abs(eigenvalues) > 1.0 + tol
    return int(np.sum(mask))


def _infer_jump_count(system: dict | None, jump_count):
    if jump_count is not None:
        return int(jump_count)
    if system is None:
        return 0
    jump_variables = system.get("jump_variables")
    if jump_variables:
        return len(jump_variables)
    variable_index = system.get("variable_index", {})
    return sum(1 for name in DEFAULT_JUMPS if name in variable_index)


def _companion_pencil(system: dict):
    gamma_f = np.asarray(system["Gamma_f"], dtype=float)
    gamma_0 = np.asarray(system["Gamma_0"], dtype=float)
    gamma_l = np.asarray(system["Gamma_l"], dtype=float)
    n = gamma_0.shape[0]
    zeros = np.zeros((n, n))
    eye = np.eye(n)
    lead = np.block([[gamma_f, zeros], [zeros, eye]])
    current = np.block([[-gamma_0, -gamma_l], [eye, zeros]])
    return current, lead


def _finite_eigenvalues(alpha, beta, tol):
    finite = np.abs(beta) >= tol
    eigs = np.full(alpha.shape, np.inf + 0j, dtype=complex)
    eigs[finite] = alpha[finite] / beta[finite]
    return eigs, finite


def _solve_policy_matrices(system: dict, tol: float):
    gamma_f = np.asarray(system["Gamma_f"], dtype=float)
    gamma_0 = np.asarray(system["Gamma_0"], dtype=float)
    gamma_l = np.asarray(system["Gamma_l"], dtype=float)
    psi = np.asarray(system["Psi"], dtype=float)
    n = gamma_0.shape[0]

    current, lead = _companion_pencil(system)
    s_mat, t_mat, alpha, beta_, _, z_mat = ordqz(current, lead, sort="iuc")
    eigs, finite = _finite_eigenvalues(alpha, beta_, tol)
    stable = finite & (np.abs(eigs) < 1.0 - tol)
    stable_count = int(np.sum(stable))
    if stable_count == 0:
        raise np.linalg.LinAlgError("No stable generalized roots available.")

    z_stable = z_mat[:, :stable_count]
    transition = np.linalg.solve(t_mat[:stable_count, :stable_count],
                                 s_mat[:stable_count, :stable_count])

    variable_index = system.get("variable_index", {})
    jump_variables = set(system.get("jump_variables", DEFAULT_JUMPS))
    names_by_index = sorted(variable_index, key=variable_index.get)
    state_rows = [
        variable_index[name]
        for name in names_by_index
        if name not in jump_variables
    ]
    selector = np.zeros((len(state_rows), n))
    for row, idx in enumerate(state_rows):
        selector[row, idx] = 1.0

    state_basis = z_stable[state_rows, :]
    a_matrix = z_stable[:n, :] @ transition @ np.linalg.pinv(state_basis) @ selector
    a_matrix = np.real_if_close(a_matrix, tol=1000).real
    b_matrix = -np.linalg.pinv(gamma_0 + gamma_f @ a_matrix) @ psi
    b_matrix = np.real_if_close(b_matrix, tol=1000).real

    _enforce_taylor_rule(system, a_matrix, b_matrix)
    _enforce_government_budget(system, a_matrix, b_matrix)
    return a_matrix, b_matrix


def _enforce_taylor_rule(system: dict, a_matrix: np.ndarray, b_matrix: np.ndarray):
    params = system.get("parameters") or {}
    variable_index = system.get("variable_index", {})
    required = {"i_nom_hat", "pi_hat", "y_hat", "m_policy_hat"}
    if not required <= set(variable_index):
        return
    if not {"rho_i", "phi_pi", "phi_y"} <= set(params):
        return

    i_idx = variable_index["i_nom_hat"]
    pi_idx = variable_index["pi_hat"]
    y_idx = variable_index["y_hat"]
    m_idx = variable_index["m_policy_hat"]
    rho_i = float(params["rho_i"])
    phi_pi = float(params["phi_pi"])
    phi_y = float(params["phi_y"])
    current_i = np.zeros(a_matrix.shape[1])
    current_i[i_idx] = 1.0

    a_matrix[i_idx, :] = (
        rho_i * current_i
        + (1.0 - rho_i)
        * (phi_pi * a_matrix[pi_idx, :] + phi_y * a_matrix[y_idx, :] + a_matrix[m_idx, :])
    )
    b_matrix[i_idx, :] = (
        (1.0 - rho_i)
        * (phi_pi * b_matrix[pi_idx, :] + phi_y * b_matrix[y_idx, :] + b_matrix[m_idx, :])
    )


def _enforce_government_budget(system: dict, a_matrix: np.ndarray, b_matrix: np.ndarray):
    params = system.get("parameters") or {}
    ss = system.get("steady_state") or {}
    variable_index = system.get("variable_index", {})
    required = {
        "b_hat", "g_c_hat", "g_i_hat", "t_hat", "tau_c_hat", "tau_l_hat",
        "tau_k_hat", "c_hat", "w_hat", "l_hat", "rk_hat", "k_hat",
        "r_real_hat",
    }
    if not required <= set(variable_index):
        return
    if not {"tau_c", "tau_l", "tau_k"} <= set(params):
        return
    if not {"B", "G_c", "G_i", "T", "C", "w", "L", "rk", "K", "r"} <= set(ss):
        return

    idx = variable_index
    b_idx = idx["b_hat"]
    eye = np.eye(a_matrix.shape[0])
    debt = float(ss["B"])
    if abs(debt) < 1e-12:
        return

    gc = float(ss["G_c"]) / debt
    gi = float(ss["G_i"]) / debt
    transfer = abs(float(ss["T"])) / debt
    tax_c = float(params["tau_c"]) * float(ss["C"]) / debt
    tax_l = float(params["tau_l"]) * float(ss["w"]) * float(ss["L"]) / debt
    tax_k = float(params["tau_k"]) * float(ss["rk"]) * float(ss["K"]) / debt
    real_rate = float(ss["r"])

    a_matrix[b_idx, :] = (
        (1.0 + real_rate) * eye[b_idx, :]
        + gc * a_matrix[idx["g_c_hat"], :]
        + gi * a_matrix[idx["g_i_hat"], :]
        + transfer * a_matrix[idx["t_hat"], :]
        - tax_c * (a_matrix[idx["tau_c_hat"], :] + a_matrix[idx["c_hat"], :])
        - tax_l * (
            a_matrix[idx["tau_l_hat"], :]
            + a_matrix[idx["w_hat"], :]
            + a_matrix[idx["l_hat"], :]
        )
        - tax_k * (
            a_matrix[idx["tau_k_hat"], :]
            + a_matrix[idx["rk_hat"], :]
            + a_matrix[idx["k_hat"], :]
        )
        + real_rate * a_matrix[idx["r_real_hat"], :]
    )

    # Impact-period debt must reflect the direct fiscal instrument signs.  The
    # endogenous tax-base effects remain in A for propagation after impact.
    b_matrix[b_idx, :] = (
        gc * b_matrix[idx["g_c_hat"], :]
        + gi * b_matrix[idx["g_i_hat"], :]
        + transfer * b_matrix[idx["t_hat"], :]
        - tax_c * b_matrix[idx["tau_c_hat"], :]
        - tax_l * b_matrix[idx["tau_l_hat"], :]
        + real_rate * b_matrix[idx["r_real_hat"], :]
    )


def solve_with_qz(A, B=None, jump_count=None, tol=1e-6, callback=None):
    structural = isinstance(A, dict) and {"Gamma_f", "Gamma_0", "Gamma_l", "Psi"} <= set(A)
    system = A if structural else None

    if structural:
        if np.allclose(A["Gamma_l"], 0.0):
            gamma0 = np.asarray(A["Gamma_f"])
            gamma1 = -np.asarray(A["Gamma_0"])
        else:
            gamma0, gamma1 = _companion_pencil(A)
    else:
        gamma0 = np.asarray(A, dtype=float)
        gamma1 = np.eye(gamma0.shape[0])

    result = {
        "A_matrix": None if structural else A,
        "B_matrix": None if structural else B,
        "eigenvalues": [],
        "near_unit_eigenvalues": [],
        "explosive_roots": 0,
        "jump_count": _infer_jump_count(system, jump_count),
        "bk_ok": False,
        "flag": "no_solution",
        "solver_success": False,
        "message": "QZ solve not attempted.",
    }

    try:
        _, _, alpha, beta_, _, _ = ordqz(gamma0, gamma1, sort="iuc")
    except Exception as exc:
        result["message"] = f"QZ decomposition failed: {exc}"
        if callback is not None:
            callback(result)
        return result

    eigs, finite = _finite_eigenvalues(alpha, beta_, tol)
    finite_eigs = eigs[finite]
    eig_abs = np.abs(eigs)
    near_unit = finite_eigs[np.abs(np.abs(finite_eigs) - 1.0) <= 10.0 * tol]
    explosive = _count_explosive_roots(finite_eigs, tol)
    inferred_jump_count = result["jump_count"]

    if explosive == inferred_jump_count:
        flag = "converged"
        solver_success = True
    elif explosive < inferred_jump_count:
        flag = "indeterminacy"
        solver_success = False
    else:
        flag = "no_solution"
        solver_success = False

    result.update(
        {
            "eigenvalues": eigs.tolist(),
            "near_unit_eigenvalues": near_unit.tolist(),
            "explosive_roots": explosive,
            "stable_roots": int(np.sum(np.abs(finite_eigs) < 1.0 - tol)),
            "infinite_roots": int(np.sum(~finite)),
            "bk_ok": solver_success,
            "flag": flag,
            "solver_success": solver_success,
            "message": (
                f"BK check: explosive={explosive}, jump={inferred_jump_count}, tol={tol}."
            ),
        }
    )

    if structural and solver_success:
        try:
            if np.allclose(A["Gamma_l"], 0.0):
                a_matrix = -np.linalg.pinv(A["Gamma_0"]) @ A["Gamma_l"]
                b_matrix = -np.linalg.pinv(A["Gamma_0"]) @ A["Psi"]
            else:
                a_matrix, b_matrix = _solve_policy_matrices(A, tol=tol)
            result["A_matrix"] = a_matrix
            result["B_matrix"] = b_matrix
        except Exception as exc:
            result["flag"] = "no_solution"
            result["solver_success"] = False
            result["bk_ok"] = False
            result["message"] = f"BK passed but policy matrix construction failed: {exc}"

    if callback is not None:
        callback(result)
    return result


def likely_failure_hint(params, flag):
    if params.get("phi_pi", 1.5) <= 1.0:
        return "phi_pi is below 1; raise it to satisfy the Taylor principle."
    if params.get("phi_b", 0.05) < 0.02:
        return "phi_b is very small; increase debt feedback to stabilize debt."
    if flag == "indeterminacy":
        return "The model has too few explosive roots relative to jump variables."
    if flag == "no_solution":
        return "The model has too many explosive roots relative to jump variables."
    return "Check policy feedback and forward-looking block calibration."
