"""Monte Carlo grid maxima of fBm and extrapolation of the discretisation bias.

The paper shows that, for small H, the maximum over n grid points is far below the
supremum, with first-order bias proportional to n^{-H} (Corollary 2.14). Richardson-type
extrapolation must therefore use the rate H, not the Brownian rate 1/2.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize

from roughmax.simulate import fgn

_MAX_BATCH_ENTRIES = 2**23  # complex entries per FFT batch, about 128 MB


def grid_max_samples(
    H: float, n: int, size: int, rng: np.random.Generator, batch: int | None = None
) -> np.ndarray:
    """Samples of max_{1<=i<=n} B^H_{i/n} (the grid excludes the origin)."""
    if batch is None:
        batch = max(1, min(size, _MAX_BATCH_ENTRIES // (2 * n)))
    out = np.empty(size)
    done = 0
    while done < size:
        k = min(batch, size - done)
        paths = np.cumsum(fgn(n, H, k, rng), axis=1)
        out[done : done + k] = paths.max(axis=1)
        done += k
    return out


def grid_max_mc(
    H: float, n: int, size: int, rng: np.random.Generator, batch: int | None = None
) -> tuple[float, float]:
    """Monte Carlo mean and standard error of max_{1<=i<=n} B^H_{i/n}."""
    samples = grid_max_samples(H, n, size, rng, batch)
    return float(samples.mean()), float(samples.std(ddof=1) / np.sqrt(size))


def richardson(n1: float, v1: float, n2: float, v2: float, rate: float) -> float:
    """Two-point extrapolation assuming v(n) = A - c n^{-rate}; returns A."""
    w1, w2 = n1**rate, n2**rate
    return (v2 * w2 - v1 * w1) / (w2 - w1)


def fit_power_bias(ns: np.ndarray, means: np.ndarray, ses: np.ndarray) -> dict[str, float]:
    """Weighted least-squares fit of v(n) = A - c n^{-r}; returns A, c, r and their s.e."""
    ns, means, ses = (np.asarray(a, dtype=float) for a in (ns, means, ses))
    if np.any(ses <= 0) or ns.size < 3:
        raise ValueError("need at least three points with positive standard errors")
    log_n = np.log(ns)

    def model(x: np.ndarray, A: float, c: float, r: float) -> np.ndarray:
        return A - c * np.exp(-r * x)

    span = means[-1] - means[0]
    p0 = (means[-1] + abs(span), abs(span) + 1e-3, 0.3)
    params, cov = optimize.curve_fit(
        model,
        log_n,
        means,
        p0=p0,
        sigma=ses,
        absolute_sigma=True,
        bounds=([-np.inf, 0.0, 1e-4], [np.inf, np.inf, 2.0]),
        maxfev=20000,
    )
    errs = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    return {
        "A": float(params[0]),
        "c": float(params[1]),
        "r": float(params[2]),
        "A_se": float(errs[0]),
        "c_se": float(errs[1]),
        "r_se": float(errs[2]),
    }
