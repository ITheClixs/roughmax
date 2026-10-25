"""Exact simulation of fractional Gaussian noise and fractional Brownian motion.

Uses the Davies-Harte circulant embedding: the autocovariance of fractional Gaussian
noise is embedded in a circulant matrix of size 2n, whose eigenvalues are non-negative
for every H in (0, 1), and one complex FFT yields two independent exact samples.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


def _check(n: int, H: float) -> None:
    if n < 1:
        raise ValueError("n must be positive")
    if not 0.0 < H < 1.0:
        raise ValueError(f"Hurst index must lie in (0, 1), got {H}")


def fgn_autocovariance(n: int, H: float) -> np.ndarray:
    """Autocovariance gamma(0..n) of the increments of B^H on the grid of step 1/n."""
    k = np.arange(n + 1, dtype=float)
    two_h = 2.0 * H
    gamma = 0.5 * (np.abs(k + 1) ** two_h - 2.0 * k**two_h + np.abs(k - 1) ** two_h)
    return gamma * float(n) ** (-two_h)


@lru_cache(maxsize=64)
def _sqrt_eigenvalues(n: int, H: float) -> np.ndarray:
    gamma = fgn_autocovariance(n, H)
    row = np.concatenate([gamma, gamma[-2:0:-1]])  # first row of the 2n circulant
    eig = np.fft.fft(row).real
    if eig.min() < -1e-10 * eig.max():
        raise RuntimeError("circulant embedding is not positive semi-definite")
    return np.sqrt(np.clip(eig, 0.0, None) / row.size)


def fgn(n: int, H: float, size: int, rng: np.random.Generator) -> np.ndarray:
    """Exact fractional Gaussian noise: increments of B^H over the grid {k/n}.

    Returns an array of shape (size, n); row i holds B^H_{(k+1)/n} - B^H_{k/n}, k < n.
    """
    _check(n, H)
    if size < 1:
        raise ValueError("size must be positive")
    scale = _sqrt_eigenvalues(n, H)
    m = scale.size
    pairs = (size + 1) // 2
    w = rng.standard_normal((pairs, m)) + 1j * rng.standard_normal((pairs, m))
    y = np.fft.fft(w * scale, axis=1)[:, :n]
    out = np.empty((2 * pairs, n))
    out[0::2] = y.real
    out[1::2] = y.imag
    return out[:size]


def fbm(n: int, H: float, size: int, rng: np.random.Generator, T: float = 1.0) -> np.ndarray:
    """Exact fractional Brownian motion on the grid {kT/n : 0 <= k <= n}.

    Returns an array of shape (size, n + 1) whose first column is zero.
    """
    increments = fgn(n, H, size, rng)
    paths = np.zeros((size, n + 1))
    np.cumsum(increments, axis=1, out=paths[:, 1:])
    return paths * T**H
