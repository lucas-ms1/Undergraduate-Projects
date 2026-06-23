"""CRRA (constant relative risk aversion) preference elicitation helpers.

This is intentionally lightweight: no SciPy, no MCMC. We maintain a posterior
over a 1D grid of CRRA coefficients (sigma) and update it from binary choices
using a soft (logit) likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def crra_utility(x: np.ndarray, sigma: np.ndarray | float) -> np.ndarray:
    """CRRA utility with a log case at sigma==1.

    Uses the common shifted form: u(x)= (x^(1-sigma)-1)/(1-sigma) for sigma!=1,
    u(x)=log(x) for sigma==1. Requires x>0.
    """
    x = np.asarray(x, dtype=float)
    if np.any(x <= 0):
        out = np.full(np.broadcast(x, sigma).shape, -np.inf, dtype=float)
        pos = x > 0
        if not np.any(pos):
            return out
        # Only compute on positive entries; keep -inf elsewhere.
        x_pos = x[pos]
        out_pos = crra_utility(x_pos, sigma)
        out[pos] = out_pos
        return out

    sigma_arr = np.asarray(sigma, dtype=float)
    if sigma_arr.ndim == 0:
        if abs(float(sigma_arr) - 1.0) < 1e-10:
            return np.log(x)
        return (x ** (1.0 - float(sigma_arr)) - 1.0) / (1.0 - float(sigma_arr))

    # Vector sigma: broadcast with x
    sigma_b = np.broadcast_to(sigma_arr, np.broadcast(x, sigma_arr).shape)
    x_b = np.broadcast_to(x, sigma_b.shape)
    out = np.empty_like(x_b, dtype=float)
    is_log = np.isclose(sigma_b, 1.0, atol=1e-10)
    out[is_log] = np.log(x_b[is_log])
    not_log = ~is_log
    s = sigma_b[not_log]
    out[not_log] = (x_b[not_log] ** (1.0 - s) - 1.0) / (1.0 - s)
    return out


def crra_inverse_utility(u: np.ndarray, sigma: np.ndarray | float) -> np.ndarray:
    """Inverse of crra_utility for x>0."""
    u = np.asarray(u, dtype=float)
    sigma_arr = np.asarray(sigma, dtype=float)
    if sigma_arr.ndim == 0:
        if abs(float(sigma_arr) - 1.0) < 1e-10:
            return np.exp(u)
        s = float(sigma_arr)
        return np.maximum(0.0, ((1.0 - s) * u + 1.0)) ** (1.0 / (1.0 - s))

    sigma_b = np.broadcast_to(sigma_arr, np.broadcast(u, sigma_arr).shape)
    u_b = np.broadcast_to(u, sigma_b.shape)
    out = np.empty_like(u_b, dtype=float)
    is_log = np.isclose(sigma_b, 1.0, atol=1e-10)
    out[is_log] = np.exp(u_b[is_log])
    not_log = ~is_log
    s = sigma_b[not_log]
    out[not_log] = np.maximum(0.0, ((1.0 - s) * u_b[not_log] + 1.0)) ** (1.0 / (1.0 - s))
    return out


def logsumexp(a: np.ndarray) -> float:
    """Stable log(sum(exp(a)))."""
    a = np.asarray(a, dtype=float)
    m = float(np.max(a))
    return m + float(np.log(np.sum(np.exp(a - m))))


def normalized_log_probs(logp: np.ndarray) -> np.ndarray:
    """Normalize log probabilities to sum to 1 in probability space."""
    logp = np.asarray(logp, dtype=float)
    return logp - logsumexp(logp)


def probs_from_logp(logp: np.ndarray) -> np.ndarray:
    logp = normalized_log_probs(logp)
    p = np.exp(logp)
    return p / np.sum(p)


def entropy_from_logp(logp: np.ndarray) -> float:
    """Shannon entropy of the implied distribution."""
    p = probs_from_logp(logp)
    p = np.clip(p, 1e-300, 1.0)
    return float(-np.sum(p * np.log(p)))


@dataclass(frozen=True)
class ChoiceQuestion:
    low: float
    high: float
    p_high: float
    sure: float


def eu_gamble(question: ChoiceQuestion, sigma_grid: np.ndarray) -> np.ndarray:
    low = float(question.low)
    high = float(question.high)
    p = float(question.p_high)
    u_low = crra_utility(low, sigma_grid)
    u_high = crra_utility(high, sigma_grid)
    return (1.0 - p) * u_low + p * u_high


def eu_sure(question: ChoiceQuestion, sigma_grid: np.ndarray) -> np.ndarray:
    return crra_utility(float(question.sure), sigma_grid)


def choice_prob_choose_gamble(
    question: ChoiceQuestion,
    sigma_grid: np.ndarray,
    sensitivity: float = 8.0,
) -> np.ndarray:
    """Logit choice probability of choosing the gamble over the sure option."""
    eu_b = eu_gamble(question, sigma_grid)
    eu_a = eu_sure(question, sigma_grid)
    delta = eu_b - eu_a
    # Normalize to reduce scale sensitivity across outcome ranges.
    scale = np.abs(crra_utility(question.high, sigma_grid) - crra_utility(question.low, sigma_grid))
    scale = np.maximum(scale, 1e-8)
    z = sensitivity * (delta / scale)
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def update_posterior_logp(
    prior_logp: np.ndarray,
    sigma_grid: np.ndarray,
    question: ChoiceQuestion,
    chose_gamble: bool,
    sensitivity: float = 8.0,
) -> np.ndarray:
    """Bayes update on a sigma grid; returns normalized log posterior."""
    p_g = choice_prob_choose_gamble(question, sigma_grid, sensitivity=sensitivity)
    like = p_g if chose_gamble else (1.0 - p_g)
    like = np.clip(like, 1e-12, 1.0)
    post = np.asarray(prior_logp, dtype=float) + np.log(like)
    return normalized_log_probs(post)


def posterior_quantile(
    sigma_grid: np.ndarray,
    logp: np.ndarray,
    q: float,
) -> float:
    p = probs_from_logp(logp)
    cdf = np.cumsum(p)
    idx = int(np.searchsorted(cdf, q, side="left"))
    idx = int(np.clip(idx, 0, len(sigma_grid) - 1))
    return float(sigma_grid[idx])


def posterior_summary(sigma_grid: np.ndarray, logp: np.ndarray) -> dict[str, float]:
    p = probs_from_logp(logp)
    mean = float(np.sum(p * sigma_grid))
    map_sigma = float(sigma_grid[int(np.argmax(logp))])
    median = posterior_quantile(sigma_grid, logp, 0.5)
    lo = posterior_quantile(sigma_grid, logp, 0.025)
    hi = posterior_quantile(sigma_grid, logp, 0.975)
    return {"mean": mean, "map": map_sigma, "median": median, "ci_lo": lo, "ci_hi": hi}


def certainty_equivalent(question: ChoiceQuestion, sigma_grid: np.ndarray) -> np.ndarray:
    """Certainty equivalent of the gamble for each sigma in the grid."""
    u = eu_gamble(question, sigma_grid)
    return crra_inverse_utility(u, sigma_grid)


def pick_next_question(
    sigma_grid: np.ndarray,
    logp: np.ndarray,
    low: float,
    high: float,
    p_candidates: np.ndarray | None = None,
    sure_candidates: np.ndarray | None = None,
    sensitivity: float = 8.0,
) -> ChoiceQuestion:
    """Choose a (p, sure) pair minimizing expected posterior entropy."""
    if p_candidates is None:
        p_candidates = np.linspace(0.1, 0.9, 9)
    if sure_candidates is None:
        sure_candidates = np.linspace(low, high, 25)
    p_candidates = np.asarray(p_candidates, dtype=float)
    sure_candidates = np.asarray(sure_candidates, dtype=float)

    prior_p = probs_from_logp(logp)
    best = None
    best_score = np.inf

    # Precompute utilities of endpoints on the sigma grid for speed.
    u_low = crra_utility(low, sigma_grid)
    u_high = crra_utility(high, sigma_grid)
    denom = np.maximum(np.abs(u_high - u_low), 1e-8)

    for p in p_candidates:
        eu_b = (1.0 - p) * u_low + p * u_high  # (n_sigma,)
        for sure in sure_candidates:
            q = ChoiceQuestion(low=float(low), high=float(high), p_high=float(p), sure=float(sure))
            eu_a = crra_utility(float(sure), sigma_grid)
            delta = eu_b - eu_a
            z = sensitivity * (delta / denom)
            z = np.clip(z, -60.0, 60.0)
            p_g = 1.0 / (1.0 + np.exp(-z))

            # Predictive probability of choosing gamble under current posterior.
            p_choose_g = float(np.sum(prior_p * p_g))
            p_choose_g = float(np.clip(p_choose_g, 1e-9, 1.0 - 1e-9))

            # Expected entropy after observing the choice.
            post_g = normalized_log_probs(logp + np.log(np.clip(p_g, 1e-12, 1.0)))
            post_s = normalized_log_probs(logp + np.log(np.clip(1.0 - p_g, 1e-12, 1.0)))
            score = p_choose_g * entropy_from_logp(post_g) + (1.0 - p_choose_g) * entropy_from_logp(post_s)

            if score < best_score:
                best_score = score
                best = q

    assert best is not None
    return best

