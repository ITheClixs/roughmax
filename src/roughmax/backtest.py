"""Daily trend-following backtests used to build a correlated strategy sweep.

Positions are decided with information up to the close of day t and earn the return of
day t+1; trading costs are charged on the turnover at the close of day t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 365  # crypto trades every day
TSMOM_LOOKBACKS = tuple(int(v) for v in np.unique(np.round(np.geomspace(5, 250, 20))))
MA_PAIRS = tuple((f, s) for f in (5, 10, 20, 40) for s in (50, 100, 150, 200))
VOL_WINDOW = 60
MAX_LEVERAGE = 2.0
WARMUP = max(max(TSMOM_LOOKBACKS), max(slow for _, slow in MA_PAIRS), VOL_WINDOW)


def tsmom_positions(prices: pd.Series, lookback: int) -> pd.Series:
    """Time-series momentum: sign of the log return over the last `lookback` days."""
    signal = np.sign(np.log(prices).diff(lookback))
    return signal.fillna(0.0)


def ma_cross_positions(prices: pd.Series, fast: int, slow: int) -> pd.Series:
    """Moving-average crossover: sign of the fast minus the slow simple moving average."""
    if fast >= slow:
        raise ValueError("fast window must be shorter than slow window")
    gap = prices.rolling(fast).mean() - prices.rolling(slow).mean()
    return np.sign(gap).fillna(0.0)


def strategy_returns(
    prices: pd.Series,
    positions: pd.Series,
    vol_target: float | None = None,
    rebalance: int = 1,
    cost_bps: float = 10.0,
) -> pd.Series:
    """Net daily returns of trading `positions` on `prices`.

    vol_target (annualised) scales positions by vol_target / trailing 60-day volatility,
    capped at 2x leverage. With rebalance = k the position is only updated every k days.
    """
    if rebalance < 1:
        raise ValueError("rebalance must be at least 1")
    asset_ret = prices.pct_change()
    pos = positions.astype(float)
    if vol_target is not None:
        realised = asset_ret.rolling(VOL_WINDOW, min_periods=20).std() * np.sqrt(TRADING_DAYS)
        pos = (pos * (vol_target / realised)).clip(-MAX_LEVERAGE, MAX_LEVERAGE).fillna(0.0)
    if rebalance > 1:
        on_schedule = np.arange(len(pos)) % rebalance == 0
        pos = pos.where(on_schedule).ffill().fillna(0.0)
    turnover = pos.diff().abs()
    net = pos.shift(1) * asset_ret - cost_bps / 1e4 * turnover
    return net.iloc[1:]


def _signals(prices: pd.Series) -> dict[str, pd.Series]:
    out = {f"tsmom{L}": tsmom_positions(prices, L) for L in TSMOM_LOOKBACKS}
    for fast, slow in MA_PAIRS:
        out[f"ma{fast}-{slow}"] = ma_cross_positions(prices, fast, slow)
    return out


def build_sweep(prices: pd.DataFrame, cost_bps: float = 10.0) -> pd.DataFrame:
    """Daily net returns of every configuration in the sweep.

    Universes: each asset alone and an equal-weight average across assets ("EW"). Signals:
    time-series momentum over 20 log-spaced lookbacks and 16 moving-average crossovers.
    Variants: with or without 40% volatility targeting, daily or weekly (7-day) rebalancing.
    Each asset's first WARMUP days after its listing are discarded, the equal-weight basket
    is only defined when every asset trades, and rows with any missing value are dropped.
    """
    variants = [(vt, rb) for vt in (None, 0.4) for rb in (1, 7)]
    columns: dict[str, pd.Series] = {}
    per_asset: dict[str, dict[str, pd.Series]] = {}
    for asset in prices.columns:
        series = prices[asset].dropna()
        for name, pos in _signals(series).items():
            for vt, rb in variants:
                key = f"{name}|vt{0 if vt is None else 1}|rb{rb}"
                ret = strategy_returns(series, pos, vol_target=vt, rebalance=rb, cost_bps=cost_bps)
                ret.iloc[:WARMUP] = np.nan
                columns[f"{asset}|{key}"] = ret
                per_asset.setdefault(key, {})[asset] = ret
    for key, by_asset in per_asset.items():
        columns[f"EW|{key}"] = pd.DataFrame(by_asset).mean(axis=1, skipna=False)
    return pd.DataFrame(columns).dropna()
