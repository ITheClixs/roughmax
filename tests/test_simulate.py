import numpy as np
import pytest

from roughmax import simulate


def test_fbm_has_unit_variance_at_one():
    rng = np.random.default_rng(1)
    paths = simulate.fbm(256, 0.3, 6000, rng)
    assert paths.shape == (6000, 257)
    assert np.all(paths[:, 0] == 0.0)
    assert paths[:, -1].var() == pytest.approx(1.0, abs=0.06)


def test_fbm_scales_with_horizon():
    rng = np.random.default_rng(2)
    paths = simulate.fbm(128, 0.2, 6000, rng, T=16.0)
    assert paths[:, -1].var() == pytest.approx(16.0**0.4, rel=0.06)


@pytest.mark.parametrize("H", [0.05, 0.25, 0.5])
def test_fgn_lag_one_autocovariance(H):
    n = 512
    rng = np.random.default_rng(3)
    x = simulate.fgn(n, H, 4000, rng)
    lag1 = np.mean(x[:, 1:] * x[:, :-1])
    expected = 0.5 * n ** (-2 * H) * (2 ** (2 * H) - 2)
    var = np.mean(x**2)
    assert var == pytest.approx(n ** (-2 * H), rel=0.03)
    assert lag1 == pytest.approx(expected, abs=0.03 * n ** (-2 * H))


def test_odd_size_and_reproducibility():
    a = simulate.fgn(64, 0.1, 3, np.random.default_rng(7))
    b = simulate.fgn(64, 0.1, 3, np.random.default_rng(7))
    assert a.shape == (3, 64)
    np.testing.assert_array_equal(a, b)


def test_invalid_hurst_rejected():
    with pytest.raises(ValueError):
        simulate.fgn(16, 1.2, 1, np.random.default_rng(0))
