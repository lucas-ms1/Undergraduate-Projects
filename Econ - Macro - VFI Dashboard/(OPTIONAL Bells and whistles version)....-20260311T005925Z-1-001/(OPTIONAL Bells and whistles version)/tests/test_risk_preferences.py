import numpy as np

from analysis.risk_preferences import (
    ChoiceQuestion,
    crra_inverse_utility,
    crra_utility,
    pick_next_question,
    probs_from_logp,
    update_posterior_logp,
)


def test_crra_log_case_matches_log():
    x = np.array([0.5, 1.0, 2.0, 10.0])
    u = crra_utility(x, 1.0)
    assert np.allclose(u, np.log(x))


def test_crra_inverse_roundtrip():
    x = np.array([0.5, 1.0, 2.0, 10.0])
    for sigma in [0.1, 0.5, 1.0, 2.0, 5.0]:
        u = crra_utility(x, sigma)
        x2 = crra_inverse_utility(u, sigma)
        assert np.allclose(x2, x, rtol=1e-10, atol=1e-12)


def test_posterior_update_normalizes():
    sigma_grid = np.linspace(0.0, 10.0, 1001)
    prior_logp = np.full_like(sigma_grid, -np.log(len(sigma_grid)))
    q = ChoiceQuestion(low=10.0, high=30.0, p_high=0.5, sure=20.0)
    post_logp = update_posterior_logp(prior_logp, sigma_grid, q, chose_gamble=True, sensitivity=8.0)
    p = probs_from_logp(post_logp)
    assert np.isfinite(post_logp).all()
    assert np.isclose(float(np.sum(p)), 1.0)
    assert np.all(p >= 0.0)


def test_pick_next_question_in_bounds():
    sigma_grid = np.linspace(0.0, 10.0, 1001)
    logp = np.full_like(sigma_grid, -np.log(len(sigma_grid)))
    q = pick_next_question(sigma_grid, logp, low=10.0, high=30.0, sensitivity=8.0)
    assert 10.0 <= q.sure <= 30.0
    assert 0.0 < q.p_high < 1.0

