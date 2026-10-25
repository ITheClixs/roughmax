import numpy as np
import pytest

from roughmax import peakrisk, simulate, theory


def test_mc_and_asymptotic_are_close_for_small_h():
    rng = np.random.default_rng(1)
    mc, se = peakrisk.expected_grid_peak(0.02, 1.0, 1024, method="mc", size=4000, rng=rng)
    asym, _ = peakrisk.expected_grid_peak(0.02, 1.0, 1024, method="asymptotic")
    assert 0.75 < mc / asym < 1.0
    assert se < 0.02 * mc


def test_scaling_in_nu():
    a, _ = peakrisk.expected_grid_peak(0.1, 1.0, 50, method="asymptotic")
    b, _ = peakrisk.expected_grid_peak(0.1, 0.4, 50, method="asymptotic")
    assert b == pytest.approx(0.4 * a)


def test_noise_raises_the_expected_peak():
    rng = np.random.default_rng(2)
    clean, _ = peakrisk.expected_grid_peak(0.1, 0.3, 20, method="mc", size=20000, rng=rng)
    noisy, _ = peakrisk.expected_grid_peak(
        0.1, 0.3, 20, noise_sd=0.2, method="mc", size=20000, rng=rng
    )
    assert noisy > clean + 0.05


def test_empirical_matches_mc_on_simulated_path():
    rng = np.random.default_rng(3)
    H, nu, n = 0.1, 0.3, 20
    x = nu * simulate.fbm(40000, H, 1, rng, T=40000.0)[0]
    emp = peakrisk.empirical_grid_peak(x, n)
    mc, mc_se = peakrisk.expected_grid_peak(H, nu, n, method="mc", size=40000, rng=rng)
    assert abs(emp["mean"] - mc) < 3 * np.hypot(emp["se"], mc_se)
    assert emp["count"] == 40001 - n


def test_empirical_skips_windows_with_gaps():
    x = np.arange(10.0)
    x[4] = np.nan
    emp = peakrisk.empirical_grid_peak(x, 2)
    assert emp["count"] == 5


def test_quantile_bound_dominates_simulation():
    rng = np.random.default_rng(4)
    H, n = 0.1, 64
    samples = peakrisk.grid_peak_samples(H, 1.0, n, 20000, rng)
    assert peakrisk.grid_peak_bound(H, 1.0, n, 0.05) >= np.quantile(samples, 0.95)
    assert peakrisk.grid_peak_bound(H, 1.0, n, 0.05) == pytest.approx(
        theory.peak_quantile_bound(H, 0.05, 1.0, float(n))
    )


def test_continuous_gap_orders_grid_below_sup():
    gap = peakrisk.continuous_peak_gap(0.1, 0.3, 30)
    assert gap["grid_asymptotic"] < gap["sup_upper"]
    assert gap["sup_lower"] < gap["sup_upper"]


def test_quantile_bound_with_noise_dominates_noisy_simulation():
    rng = np.random.default_rng(5)
    H, nu, n, s = 0.1, 0.3, 30, 0.15
    samples = peakrisk.grid_peak_samples(H, nu, n, 20000, rng, noise_sd=s)
    bound = peakrisk.grid_peak_bound(H, nu, n, 0.05, noise_sd=s)
    assert bound >= np.quantile(samples, 0.95)
    assert bound > peakrisk.grid_peak_bound(H, nu, n, 0.05)


def test_empirical_peaks_values():
    x = np.array([0.0, 1.0, -1.0, 2.0, 5.0, np.nan])
    peaks = peakrisk.empirical_peaks(x, 2)
    np.testing.assert_allclose(peaks, [1.0, 1.0, 6.0])


def test_ou_peaks_match_brownian_for_tiny_mean_reversion():
    rng = np.random.default_rng(6)
    n, theta = 30, 1e-6
    sigma = 1.0 / np.sqrt(2 * theta)  # one-step increment variance about 1
    ou = peakrisk.ou_grid_peak_samples(theta, sigma, n, 20000, rng).mean()
    bm, se = peakrisk.expected_grid_peak(0.5, 1.0, n, method="mc", size=20000, rng=rng)
    assert ou == pytest.approx(bm, abs=4 * se + 0.02)


def test_ou_peaks_saturate():
    rng = np.random.default_rng(7)
    short = peakrisk.ou_grid_peak_samples(0.2, 0.5, 20, 5000, rng).mean()
    long = peakrisk.ou_grid_peak_samples(0.2, 0.5, 400, 5000, rng).mean()
    assert long < 2.0 < 4 * short
