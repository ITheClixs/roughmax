"""Closed-form and numerical results for M_H = E sup_{[0,1]} B^H and for grid maxima.

The formulas are those of "The maximum of fractional Brownian motion as H -> 0"
(M. D. Gueven, 2026), referred to as "the paper" below.

Upper bound, Theorem 2.1(i) (b-adic tree majorant and barrier lemma):
    M_H <= (1+d) sqrt(ln b coth(H ln b / 2)) + (1 + ln^+ P) / c,
    P = 1 / (b^(2d + d^2) - 1),  c = (1+d) sqrt(4 ln b / (1 - b^(-2H))).
Lower bound, Theorem 2.1(ii) (Cantor minorant and greedy bound):
    M_H >= mu_b (2b)^(-H) sqrt(coth(H ln(2b) / 2) / 2).
Grid phase diagram, Theorem 2.13: sqrt(H) E max_{i<=n} B^H_{i/n} -> Lambda(H ln n).
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy import integrate, optimize, stats

SQRT2 = math.sqrt(2.0)

# Bases b used to optimise the upper bound: all small integers and a geometric
# grid up to 2^20, as in the paper's numerical section.
_UPPER_BASES = tuple(sorted(set(range(2, 65)) | {round(2 ** (k / 2)) for k in range(13, 41)}))
_LEMMA_MIN_LOG_N = 1000.0


@lru_cache(maxsize=4096)
def mu_n(n: float) -> float:
    """Expected maximum of n i.i.d. standard normal variables (n may be huge)."""
    if n < 1:
        raise ValueError("n must be at least 1")
    if n == 1:
        return 0.0
    log_cdf = stats.norm.logcdf
    peak = math.sqrt(2.0 * math.log(n))
    pos, _ = integrate.quad(
        lambda x: -np.expm1(n * log_cdf(x)), 0.0, 60.0, limit=800, points=[peak]
    )
    neg, _ = integrate.quad(lambda x: np.exp(n * log_cdf(x)), -60.0, 0.0, limit=800)
    return pos - neg


def _mu_lower_large(log_n: float) -> float:
    """Lemma 3.4(b): mu_n >= sqrt(2l) - (log l + 3)/sqrt(2l) for l = log n >= 1000."""
    if log_n < _LEMMA_MIN_LOG_N:
        raise ValueError("Lemma 3.4(b) needs log n >= 1000")
    root = math.sqrt(2.0 * log_n)
    return root - (math.log(log_n) + 3.0) / root


def _check_hurst(H: float) -> None:
    if not 0.0 < H < 1.0:
        raise ValueError(f"Hurst index must lie in (0, 1), got {H}")


def _upper_at(H: float, b: int, delta: float) -> float:
    log_b = math.log(b)
    main = (1.0 + delta) * math.sqrt(log_b / math.tanh(H * log_b / 2.0))
    p_delta = 1.0 / math.expm1((2.0 * delta + delta * delta) * log_b)
    c_delta = (1.0 + delta) * math.sqrt(4.0 * log_b / -math.expm1(-2.0 * H * log_b))
    return main + (1.0 + max(math.log(p_delta), 0.0)) / c_delta


def _upper_fixed_base(H: float, b: int) -> float:
    res = optimize.minimize_scalar(
        lambda s: _upper_at(H, b, math.exp(s)),
        bounds=(math.log(1e-12), math.log(2.0)),
        method="bounded",
        options={"xatol": 1e-10},
    )
    return min(float(res.fun), _upper_at(H, b, H))


@lru_cache(maxsize=1024)
def upper_bound(H: float) -> float:
    """Upper bound (2.1) on M_H, minimised over delta > 0 and the base b."""
    _check_hurst(H)
    return min(_upper_fixed_base(H, b) for b in _UPPER_BASES)


def _lower_at(H: float, mu_b: float, log_2b: float) -> float:
    x = H * log_2b
    return mu_b * math.exp(-x) * math.sqrt(1.0 / math.tanh(x / 2.0) / 2.0)


@lru_cache(maxsize=1024)
def lower_bound(H: float) -> float:
    """Lower bound (2.2) on M_H, maximised over b = 2^k (k <= 400) and b = ceil(e^L), L >= 1000."""
    _check_hurst(H)
    best = 0.0
    for k in range(1, 401):
        b = 2.0**k
        best = max(best, _lower_at(H, mu_n(b), (k + 1) * math.log(2.0)))
    for big_l in np.geomspace(_LEMMA_MIN_LOG_N, 1e9, 400):
        # b = ceil(e^L) satisfies log(2b) <= L + e^{-L} + log 2.
        log_2b = big_l + math.exp(-big_l) + math.log(2.0)
        best = max(best, _lower_at(H, _mu_lower_large(big_l), log_2b))
    return best


def explicit_upper(H: float) -> float:
    """Closed form of Theorem 2.1(iii): M_H <= sqrt(2/H) + sqrt(H)(2 + log(1/H)), H <= 1/2."""
    if not 0.0 < H <= 0.5:
        raise ValueError("the closed-form upper bound holds for H in (0, 1/2]")
    return math.sqrt(2.0 / H) + math.sqrt(H) * (2.0 + math.log(1.0 / H))


def asymptotic_sup(H: float) -> float:
    """First-order value sqrt(2/H) of M_H as H -> 0 (Theorem 1.1)."""
    _check_hurst(H)
    return math.sqrt(2.0 / H)


def grid_limit(kappa: float) -> float:
    """Lambda(kappa) of Theorem 2.13: limit of sqrt(H) E max over the uniform grid."""
    if kappa < 0:
        raise ValueError("kappa must be non-negative")
    if math.isinf(kappa):
        return SQRT2
    if kappa <= 0.5:
        return math.sqrt(kappa)
    return SQRT2 - math.exp(0.5 - kappa) / SQRT2


def grid_max_asymptotic(H: float, n: int) -> float:
    """First-order approximation Lambda(H log n)/sqrt(H) of E max_{1<=i<=n} B^H_{i/n}."""
    _check_hurst(H)
    return grid_limit(H * math.log(n)) / math.sqrt(H)


def captured_fraction(H: float, n: int) -> float:
    """Asymptotic fraction Lambda(H log n)/sqrt(2) of M_H captured by an n-point uniform grid."""
    _check_hurst(H)
    return grid_limit(H * math.log(n)) / SQRT2


def discretisation_bias(H: float, n: int) -> float:
    """First-order bias M_H - E max over the grid, sqrt(e/(2H)) n^{-H} (Corollary 2.14).

    The formula is the joint limit H -> 0, H log n -> kappa >= 1/2.
    """
    _check_hurst(H)
    return math.sqrt(math.e / (2.0 * H)) * math.exp(-H * math.log(n))


def log_points_needed(H: float, theta: float) -> float:
    """log n for a uniform grid to capture the fraction theta of sqrt(2/H) to first order."""
    _check_hurst(H)
    if not 0.0 < theta < 1.0:
        raise ValueError("theta must lie in (0, 1)")
    kappa = 2.0 * theta**2 if theta <= 0.5 else 0.5 + math.log(1.0 / (2.0 * (1.0 - theta)))
    return kappa / H


def peak_quantile_bound(H: float, alpha: float, nu: float = 1.0, T: float = 1.0) -> float:
    """Upper bound on the (1-alpha)-quantile of sup_{[0,T]} nu B^H.

    Combines the upper bound (2.1) on the mean with the one-sided Borell-TIS inequality
    P(sup - E sup > u) <= exp(-u^2 / (2 sigma^2)), sigma^2 = nu^2 T^{2H}.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    return nu * T**H * (upper_bound(H) + math.sqrt(2.0 * math.log(1.0 / alpha)))
