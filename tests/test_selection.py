import numpy as np
import pytest
from scipy import stats

from roughmax import selection, theory


def _block_corr(sizes, within, between):
    n = sum(sizes)
    C = np.full((n, n), between)
    start = 0
    for s in sizes:
        C[start : start + s, start : start + s] = within
        start += s
    np.fill_diagonal(C, 1.0)
    return C


def _factor_corr(n, k, seed):
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((n, k)) * rng.uniform(0.2, 1.5, size=k)
    cov = B @ B.T + np.diag(rng.uniform(0.3, 1.0, size=n))
    d = np.sqrt(np.diag(cov))
    return cov / np.outer(d, d)


def test_identity_gives_iid_expected_max():
    b = selection.sandwich_bounds(np.eye(50))
    assert b["lower"] == pytest.approx(theory.mu_n(50), abs=2e-3)
    assert b["upper"] == pytest.approx(theory.mu_n(50), abs=2e-3)


def test_ultrametric_correlation_is_exact():
    C = _block_corr([40, 60, 30], within=0.6, between=0.1)
    b = selection.sandwich_bounds(C)
    assert b["upper"] - b["lower"] < 2e-3
    mc, se = selection.mc_expected_max(C, 40000, np.random.default_rng(1))
    assert abs(mc - b["upper"]) < 3.5 * se


@pytest.mark.parametrize("seed", [2, 3])
def test_sandwich_brackets_monte_carlo(seed):
    C = _factor_corr(200, 4, seed)
    b = selection.sandwich_bounds(C)
    mc, se = selection.mc_expected_max(C, 30000, np.random.default_rng(seed))
    assert b["lower"] <= mc + 3 * se
    assert mc <= b["upper"] + 3 * se
    assert b["lower"] < b["upper"]


def test_floor_removes_within_cluster_variance():
    C = _block_corr([20, 20], within=0.9, between=0.0)
    Z = selection.linkage_tree(C, "complete")
    full = selection.tree_expected_max(Z)
    between_only = selection.tree_expected_max(Z, floor=float(np.sqrt(2 * (1 - 0.9))))
    assert between_only < full
    assert selection.tree_expected_max(Z, floor=10.0) == pytest.approx(0.0, abs=1e-9)


def test_effective_trials_inverts_mu():
    assert selection.effective_trials(theory.mu_n(50)) == pytest.approx(50, rel=1e-3)


def test_bl_formula_close_to_exact_iid():
    assert selection.bl_expected_max(1000) == pytest.approx(theory.mu_n(1000), rel=0.03)


def test_deflated_sharpe_hand_value():
    got = selection.deflated_sharpe(0.1, 252, skew=0.0, kurt=3.0, sr0=0.0)
    expected = stats.norm.cdf(0.1 * np.sqrt(251) / np.sqrt(1 + 0.5 * 0.01))
    assert got == pytest.approx(expected)


def test_mc_handles_non_psd_input():
    C = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
    mc, se = selection.mc_expected_max(C, 2000, np.random.default_rng(0))
    assert np.isfinite(mc) and se > 0
