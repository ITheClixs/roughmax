"""Realised variance and roughness estimation for log-volatility series.

The roughness fit follows Gatheral, Jaisson and Rosenbaum ("Volatility is rough", 2018):
the second-order structure function m(2, D) = E |x_{t+D} - x_t|^2 of log-volatility
scales like nu^2 D^{2H}. Daily log realised volatility also carries estimation noise,
which adds a constant 2 s^2; the noisy fit estimates (H, nu, s) jointly.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize

SECONDS_PER_DAY = 86_400


@dataclass(frozen=True)
class RoughnessFit:
    """Fitted structure function m(2, D) = 2 noise_sd^2 + nu^2 D^{2H} (D in sampling steps)."""

    H: float
    nu: float
    noise_sd: float
    lags: tuple[int, ...]
    m: tuple[float, ...]

    def model(self, lags: Iterable[float]) -> np.ndarray:
        d = np.asarray(list(lags), dtype=float)
        return 2.0 * self.noise_sd**2 + self.nu**2 * d ** (2.0 * self.H)


def daily_realized_variance(bars: pd.DataFrame, min_frac: float = 0.9) -> pd.Series:
    """Sum of squared intraday log returns per UTC day.

    Returns are taken between consecutive available bars, so a return across a gap spans
    the missing interval and the realised variance stays unbiased; days with fewer than
    min_frac of the expected number of bars are dropped because their estimate is noisier.
    """
    if bars.empty:
        raise ValueError("no bars given")
    close = bars["close"].astype(float)
    spacing = pd.Series(close.index).diff().median()
    bars_per_day = SECONDS_PER_DAY / spacing.total_seconds()
    returns = np.log(close).diff().dropna()
    day = returns.index.floor("D")
    grouped = (returns**2).groupby(day)
    rv = grouped.sum()
    count = grouped.count()
    keep = count >= min_frac * bars_per_day
    out = rv[keep]
    out.index.name = "date"
    out.name = "rv"
    return out


def remove_weekly_seasonality(x: pd.Series, reference: pd.Series | None = None) -> pd.Series:
    """Subtract the day-of-week mean of a daily series.

    Crypto volatility has a weekly cycle (quiet weekends), which puts dips into the
    structure function at multiples of 7 days. The weekday means are estimated on
    `reference` (default: x itself), so out-of-sample use never peeks at test data.
    """
    ref = x if reference is None else reference
    means = ref.groupby(ref.index.day_of_week).mean()
    return x - means.reindex(x.index.day_of_week).to_numpy()


def structure_function(x: np.ndarray, lags: Iterable[int], q: float = 2.0) -> np.ndarray:
    """m(q, D) = mean |x_{t+D} - x_t|^q over all t with both values present."""
    x = np.asarray(x, dtype=float)
    values = []
    for lag in lags:
        if lag < 1 or lag >= x.size:
            raise ValueError(f"invalid lag {lag}")
        diff = x[lag:] - x[:-lag]
        diff = diff[np.isfinite(diff)]
        if diff.size == 0:
            raise ValueError(f"no complete pairs at lag {lag}")
        values.append(np.mean(np.abs(diff) ** q))
    return np.asarray(values)


def _positive_structure(x: np.ndarray, lags: tuple[int, ...]) -> np.ndarray:
    m = structure_function(x, lags)
    if np.any(m <= 0):
        raise ValueError("series is constant at some lag; roughness is not identified")
    return m


def fit_roughness(
    x: np.ndarray, lags: Iterable[int] = range(1, 31), noise: bool = True
) -> RoughnessFit:
    """Fit H, nu (and the noise level) from the second-order structure function.

    Without noise this is the log-log regression of Gatheral, Jaisson and Rosenbaum; with
    noise the model 2 s^2 + nu^2 D^{2H} is fitted by least squares on the log scale.
    """
    lags = tuple(int(v) for v in lags)
    m = _positive_structure(x, lags)
    log_d, log_m = np.log(lags), np.log(m)
    slope, intercept = np.polyfit(log_d, log_m, 1)
    H0, nu0 = float(np.clip(slope / 2.0, 1e-3, 0.998)), float(np.exp(intercept / 2.0))
    if not noise:
        return RoughnessFit(float(H0), nu0, 0.0, lags, tuple(m))

    def resid(p: np.ndarray) -> np.ndarray:
        h, log_nu, log_s = p
        model = 2.0 * np.exp(2.0 * log_s) + np.exp(2.0 * log_nu) * np.exp(2.0 * h * log_d)
        return np.log(model) - log_m

    start = np.array([H0, np.log(nu0), np.log(0.3 * np.sqrt(m[0] / 2.0) + 1e-12)])
    sol = optimize.least_squares(
        resid, start, bounds=([1e-3, -np.inf, -20.0], [0.999, np.inf, np.inf]), x_scale="jac"
    )
    h, log_nu, log_s = sol.x
    return RoughnessFit(float(h), float(np.exp(log_nu)), float(np.exp(log_s)), lags, tuple(m))


@dataclass(frozen=True)
class OUFit:
    """Stationary Ornstein-Uhlenbeck fit m(2, D) = 2 s^2 + 2 sigma^2 (1 - exp(-theta D))."""

    theta: float
    sigma: float
    noise_sd: float
    lags: tuple[int, ...]
    m: tuple[float, ...]

    def model(self, lags: Iterable[float]) -> np.ndarray:
        d = np.asarray(list(lags), dtype=float)
        return 2.0 * self.noise_sd**2 + 2.0 * self.sigma**2 * -np.expm1(-self.theta * d)


def fit_ou(x: np.ndarray, lags: Iterable[int] = range(1, 31), noise: bool = True) -> OUFit:
    """Fit a mean-reverting (H = 1/2) Ornstein-Uhlenbeck model to the structure function.

    This is the classical non-rough benchmark for log-volatility, fitted on the same lags,
    the same statistic and with the same optional noise term as fit_roughness.
    """
    lags = tuple(int(v) for v in lags)
    m = _positive_structure(x, lags)
    d_lags, log_m = np.asarray(lags, dtype=float), np.log(m)

    def resid(p: np.ndarray) -> np.ndarray:
        theta, sigma = np.exp(p[:2])
        s2 = np.exp(2.0 * p[2]) if noise else 0.0
        return np.log(2.0 * s2 + 2.0 * sigma**2 * -np.expm1(-theta * d_lags)) - log_m

    start = [np.log(0.05), np.log(np.sqrt(m[-1] / 2.0))]
    lower, upper = [-np.inf, -np.inf], [np.inf, np.inf]
    if noise:
        start.append(np.log(0.3 * np.sqrt(m[0] / 2.0) + 1e-12))
        lower.append(-20.0)
        upper.append(np.inf)
    sol = optimize.least_squares(resid, np.array(start), bounds=(lower, upper), x_scale="jac")
    theta, sigma = np.exp(sol.x[:2])
    noise_sd = float(np.exp(sol.x[2])) if noise else 0.0
    return OUFit(float(theta), float(sigma), noise_sd, lags, tuple(m))
