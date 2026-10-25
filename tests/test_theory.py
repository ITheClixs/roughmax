import math

import pytest

from roughmax import theory


def test_mu_n_small_cases():
    assert theory.mu_n(1) == 0.0
    assert theory.mu_n(2) == pytest.approx(1 / math.sqrt(math.pi), rel=1e-8)


def test_bounds_bracket_brownian_value():
    exact = math.sqrt(2 / math.pi)
    assert theory.lower_bound(0.5) <= exact <= theory.upper_bound(0.5)


def test_bounds_reproduce_paper_table():
    assert theory.upper_bound(0.1) == pytest.approx(4.935, abs=2e-3)
    assert theory.lower_bound(0.01) == pytest.approx(11.118, abs=2e-3)
    assert theory.upper_bound(1e-6) == pytest.approx(1414.223, abs=2e-3)


@pytest.mark.parametrize("H", [1e-6, 1e-3, 0.05, 0.2, 0.5])
def test_explicit_upper_dominates_optimised(H):
    assert theory.explicit_upper(H) >= theory.upper_bound(H) - 1e-9


@pytest.mark.parametrize("H", [1e-4, 0.01, 0.1, 0.3])
def test_bounds_bracket_the_asymptotic_value(H):
    lo, up = theory.lower_bound(H), theory.upper_bound(H)
    assert 0 < lo < theory.asymptotic_sup(H) < up


def test_grid_limit_values():
    assert theory.grid_limit(0.0) == 0.0
    assert theory.grid_limit(0.25) == pytest.approx(0.5)
    assert theory.grid_limit(0.5) == pytest.approx(math.sqrt(0.5))
    assert theory.grid_limit(math.inf) == pytest.approx(math.sqrt(2))
    assert theory.grid_limit(3.0) < math.sqrt(2)


def test_grid_limit_is_continuous_at_one_half():
    assert theory.grid_limit(0.5 - 1e-9) == pytest.approx(theory.grid_limit(0.5 + 1e-9), abs=1e-6)


def test_captured_fraction_and_bias():
    H, n = 0.1, 2**20
    frac = theory.captured_fraction(H, n)
    assert 0 < frac < 1
    expected = math.sqrt(math.e / 2) * math.exp(-H * math.log(n))
    assert theory.discretisation_bias(H, n) * math.sqrt(H) == pytest.approx(expected)


def test_log_points_needed_inverts_captured_fraction():
    H = 0.05
    for theta in (0.3, 0.5, 0.8, 0.95):
        log_n = theory.log_points_needed(H, theta)
        frac = theory.grid_limit(H * log_n) / math.sqrt(2)
        assert frac == pytest.approx(theta, rel=1e-9)


def test_peak_quantile_bound_monotone_and_scaled():
    H = 0.1
    q90 = theory.peak_quantile_bound(H, 0.10)
    q99 = theory.peak_quantile_bound(H, 0.01)
    assert q99 > q90 > theory.upper_bound(H)
    scaled = theory.peak_quantile_bound(H, 0.01, nu=2.0, T=4.0)
    assert scaled == pytest.approx(2.0 * 4.0**H * q99)
