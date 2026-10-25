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

    Days with fewer than min_frac of the expected number of bars are dropped, so that
    missing data never silently lowers the realised variance.
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


def fit_roughness(
    x: np.ndarray, lags: Iterable[int] = range(1, 31), noise: bool = True
) -> RoughnessFit:
    """Fit H, nu (and the noise level) from the second-order structure function.

    Without noise this is the log-log regression of Gatheral, Jaisson and Rosenbaum; with
    noise the model 2 s^2 + nu^2 D^{2H} is fitted by least squares on the log scale.
    """
    lags = tuple(int(v) for v in lags)
    m = structure_function(x, lags)
    log_d, log_m = np.log(lags), np.log(m)
    slope, intercept = np.polyfit(log_d, log_m, 1)
    H0, nu0 = max(slope / 2.0, 1e-3), float(np.exp(intercept / 2.0))
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
