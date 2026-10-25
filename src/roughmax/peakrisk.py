"""Peak-volatility risk under rough (fractional) log-volatility.

Model: log-volatility x_t = x_0 + nu B^H_t observed once per step (t in steps, e.g. days),
possibly with i.i.d. measurement noise. The quantity of interest is the peak excursion
max_{1<=i<=n} (x_i - x_0) over an n-step horizon.

The paper gives (i) its first-order mean nu n^H Lambda(H log n)/sqrt(H) (Theorem 2.13),
(ii) the continuous-time supremum, bracketed by nu n^H [lower_bound(H), upper_bound(H)]
(Theorem 2.1), and (iii) Borell-TIS tails, which turn (ii) into quantile bounds.
"""

from __future__ import annotations

import math

import numpy as np

from roughmax import theory
from roughmax.simulate import fbm


def grid_peak_samples(
    H: float,
    nu: float,
    n: int,
    size: int,
    rng: np.random.Generator,
    noise_sd: float = 0.0,
    batch: int = 2048,
) -> np.ndarray:
    """Samples of max_{1<=i<=n} (x_i - x_0) with x = nu B^H at unit spacing plus noise."""
    out = np.empty(size)
    done = 0
    while done < size:
        k = min(batch, size - done)
        x = nu * fbm(n, H, k, rng, T=float(n))
        if noise_sd > 0.0:
            x = x + noise_sd * rng.standard_normal(x.shape)
        out[done : done + k] = (x[:, 1:] - x[:, :1]).max(axis=1)
        done += k
    return out


def expected_grid_peak(
    H: float,
    nu: float,
    n: int,
    noise_sd: float = 0.0,
    method: str = "asymptotic",
    size: int = 4000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Expected peak excursion over n steps; returns (value, standard error).

    method="asymptotic" uses the first-order formula of the paper (noise-free only);
    method="mc" simulates exactly and returns the Monte Carlo standard error.
    """
    if method == "asymptotic":
        if noise_sd > 0.0:
            raise ValueError("the asymptotic formula has no noise term; use method='mc'")
        return nu * n**H * theory.grid_max_asymptotic(H, n), 0.0
    if method == "mc":
        rng = np.random.default_rng() if rng is None else rng
        samples = grid_peak_samples(H, nu, n, size, rng, noise_sd)
        return float(samples.mean()), float(samples.std(ddof=1) / math.sqrt(size))
    raise ValueError(f"unknown method {method!r}")


def grid_peak_bound(H: float, nu: float, n: int, alpha: float, noise_sd: float = 0.0) -> float:
    """Upper bound on the (1-alpha)-quantile of the peak over n steps.

    Without noise, the grid maximum is at most the supremum over [0, n], whose quantile is
    bounded by the paper's upper bound on the mean plus the Borell-TIS deviation term. With
    i.i.d. N(0, s^2) noise the peak max_i (nu B_i + e_i - e_0) is still Gaussian, its mean is
    at most nu n^H U(H) + s mu_n, and its largest variance is at most nu^2 n^{2H} + 2 s^2.
    """
    if noise_sd == 0.0:
        return theory.peak_quantile_bound(H, alpha, nu=nu, T=float(n))
    mean_bound = nu * n**H * theory.upper_bound(H) + noise_sd * theory.mu_n(n)
    sigma = math.sqrt(nu**2 * n ** (2.0 * H) + 2.0 * noise_sd**2)
    return mean_bound + sigma * math.sqrt(2.0 * math.log(1.0 / alpha))


def empirical_peaks(x: np.ndarray, n: int, step: int = 1) -> np.ndarray:
    """Peak excursions max_{1<=i<=n}(x_{t+i} - x_t) over all complete windows of a series.

    Windows containing a missing value are skipped.
    """
    x = np.asarray(x, dtype=float)
    if n < 1 or x.size <= n:
        raise ValueError("series too short for the requested horizon")
    windows = np.lib.stride_tricks.sliding_window_view(x, n + 1)[::step]
    windows = windows[np.isfinite(windows).all(axis=1)]
    return (windows[:, 1:] - windows[:, :1]).max(axis=1)


def empirical_grid_peak(x: np.ndarray, n: int, step: int = 1) -> dict[str, float]:
    """Summary of the empirical peak excursions over n steps (see empirical_peaks).

    Windows overlap when step < n, so the reported standard error uses the effective number
    of non-overlapping windows.
    """
    peaks = empirical_peaks(x, n, step)
    if peaks.size < 2:
        raise ValueError("fewer than two complete windows")
    effective = max(1.0, peaks.size * min(1.0, step / n))
    q = np.quantile(peaks, [0.5, 0.9, 0.95, 0.99])
    return {
        "mean": float(peaks.mean()),
        "se": float(peaks.std(ddof=1) / math.sqrt(effective)),
        "count": int(peaks.size),
        "q50": float(q[0]),
        "q90": float(q[1]),
        "q95": float(q[2]),
        "q99": float(q[3]),
    }


def continuous_peak_gap(H: float, nu: float, n: int) -> dict[str, float]:
    """Grid peak versus the continuous-time supremum over the same horizon of n steps."""
    scale = nu * n**H
    return {
        "grid_asymptotic": scale * theory.grid_max_asymptotic(H, n),
        "sup_lower": scale * theory.lower_bound(H),
        "sup_upper": scale * theory.upper_bound(H),
        "sup_asymptotic": scale * theory.asymptotic_sup(H),
    }
