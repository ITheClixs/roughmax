import math

import numpy as np
import pytest

from roughmax import extremes


def test_brownian_grid_max_matches_known_expansion():
    n = 1024
    mean, se = extremes.grid_max_mc(0.5, n, 20000, np.random.default_rng(11))
    expected = math.sqrt(2 / math.pi) - 0.5826 / math.sqrt(n)
    assert abs(mean - expected) < 3.5 * se
    assert 0 < se < 0.01


def test_grid_max_samples_shape_and_positivity():
    samples = extremes.grid_max_samples(0.2, 64, 1000, np.random.default_rng(1))
    assert samples.shape == (1000,)
    assert (samples > -5).all()


def test_richardson_exact_on_power_law():
    A, c, r = 3.0, 2.0, 0.15

    def v(n):
        return A - c * n ** (-r)

    assert extremes.richardson(1000, v(1000), 64000, v(64000), r) == pytest.approx(A)


def test_fit_power_bias_recovers_parameters():
    ns = np.array([2.0**k for k in range(6, 21, 2)])
    A, c, r = 4.2, 3.1, 0.12
    means = A - c * ns ** (-r)
    fit = extremes.fit_power_bias(ns, means, np.full(ns.size, 1e-4))
    assert fit["A"] == pytest.approx(A, rel=1e-4)
    assert fit["r"] == pytest.approx(r, rel=1e-3)
    assert fit["c"] == pytest.approx(c, rel=1e-3)
