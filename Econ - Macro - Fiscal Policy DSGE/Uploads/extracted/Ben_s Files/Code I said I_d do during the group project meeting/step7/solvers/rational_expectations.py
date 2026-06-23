"""Rational-expectations diagnostics and reduced-form handling."""

import numpy as np
from scipy.linalg import ordqz


def _count_explosive_pairs(eigenvalues, tol):
    """Count explosive roots, treating conjugate pairs as one."""
    explosive = 0
    used = np.zeros(len(eigenvalues), dtype=bool)
    for i, ev in enumerate(eigenvalues):
        if used[i]:
            continue
        if np.abs(ev) <= 1.0 + tol:
            used[i] = True
            continue
        if np.isclose(np.imag(ev), 0.0, atol=tol):
            explosive += 1
            used[i] = True
            continue

        conj_idx = None
        for j in range(i + 1, len(eigenvalues)):
            if used[j]:
                continue
            if np.isclose(ev, np.conj(eigenvalues[j]), atol=tol, rtol=0.0):
                conj_idx = j
                break
        explosive += 1
        used[i] = True
        if conj_idx is not None:
            used[conj_idx] = True
    return explosive


def solve_with_qz(A, B, jump_count=0, tol=1e-6, callback=None):
    """Solve/diagnose the RE system via ordered QZ decomposition."""
    n = A.shape[0]
    gamma0 = A
    gamma1 = np.eye(n)

    result = {
        "A_matrix": A,
        "B_matrix": B,
        "eigenvalues": [],
        "near_unit_eigenvalues": [],
        "explosive_roots": 0,
        "bk_ok": False,
        "flag": "no_solution",
        "solver_success": False,
        "message": "QZ solve not attempted.",
    }

    try:
        AA, BB, alpha, beta_, Q, Z = ordqz(
            gamma0,
            gamma1,
            sort=lambda a, b: np.abs(a) < np.abs(b),
        )
    except Exception as exc:
        result["message"] = f"QZ decomposition failed: {exc}"
        if callback is not None:
            callback(result)
        return result

    beta_safe = np.where(np.abs(beta_) < tol, np.nan + 0j, beta_)
    eigs = alpha / beta_safe
    eig_abs = np.abs(eigs)
    near_unit_mask = np.abs(eig_abs - 1.0) <= 10.0 * tol
    near_unit = eigs[near_unit_mask]

    stable_count = int(np.sum(eig_abs < 1.0 - tol))
    if stable_count > 0:
        z11 = Z[:stable_count, :stable_count]
        if not np.all(np.isfinite(z11)) or np.linalg.cond(z11) > 1.0 / tol:
            result.update(
                {
                    "eigenvalues": eigs.tolist(),
                    "near_unit_eigenvalues": near_unit.tolist(),
                    "message": "QZ reordering pivot ill-conditioned.",
                }
            )
            if callback is not None:
                callback(result)
            return result

    explosive = _count_explosive_pairs(eigs, tol)
    bk_ok = explosive == jump_count
    flag = "converged" if bk_ok else "indeterminacy"

    result.update(
        {
            "eigenvalues": eigs.tolist(),
            "near_unit_eigenvalues": near_unit.tolist(),
            "explosive_roots": explosive,
            "bk_ok": bk_ok,
            "flag": flag,
            "solver_success": flag == "converged",
            "message": f"BK check: explosive={explosive}, jump={jump_count}, tol={tol}.",
        }
    )

    if callback is not None:
        callback(result)
    return result


def likely_failure_hint(params, flag):
    """Return a concise suggestion for likely determinacy failures."""
    if params.get("phi_pi", 2.0) <= 1.0:
        return "phi_pi is below 1; raise it to satisfy the Taylor principle."
    if params.get("phi_b", 1.0) < 0.02:
        return "phi_b is very small; increase debt feedback to stabilize dynamics."
    theta_p = params.get("theta_p", 0.75)
    if theta_p <= 0.12 or theta_p >= 0.93:
        return "theta_p is near a boundary; move away from extremes to improve stability."
    if flag == "no_solution":
        return "Try backing away from extreme price rigidity or weak policy feedback."
    return "Try increasing phi_b and keeping phi_pi comfortably above 1."
