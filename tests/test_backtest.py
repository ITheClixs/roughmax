import numpy as np
import pandas as pd
import pytest

from roughmax import backtest


def _prices(n=400, seed=0, assets=("A", "B")):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC")
    data = {a: 100 * np.exp(np.cumsum(0.02 * rng.standard_normal(n))) for a in assets}
    return pd.DataFrame(data, index=idx)


def test_tsmom_sign():
    p = pd.Series([1.0, 2.0, 3.0, 2.0, 1.0])
    pos = backtest.tsmom_positions(p, 2)
    assert pos.tolist() == [0.0, 0.0, 1.0, 0.0, -1.0]


def test_no_look_ahead():
    prices = _prices()["A"]
    pos = backtest.tsmom_positions(prices, 20)
    base = backtest.strategy_returns(prices, pos, vol_target=0.4, rebalance=5)
    cut = prices.index[250]
    shocked = prices.copy()
    shocked[shocked.index > cut] *= np.exp(np.random.default_rng(9).standard_normal(149))
    pos2 = backtest.tsmom_positions(shocked, 20)
    other = backtest.strategy_returns(shocked, pos2, vol_target=0.4, rebalance=5)
    pd.testing.assert_series_equal(base[base.index <= cut], other[other.index <= cut])


def test_costs_are_charged_on_turnover():
    idx = pd.date_range("2022-01-01", periods=6, freq="D", tz="UTC")
    prices = pd.Series(100.0, index=idx)
    pos = pd.Series([0.0, 1.0, -1.0, -1.0, 1.0, 0.0], index=idx)
    ret = backtest.strategy_returns(prices, pos, cost_bps=10.0)
    assert ret.tolist() == pytest.approx([-0.001, -0.002, 0.0, -0.002, -0.001])


def test_build_sweep_shape_and_names():
    sweep = backtest.build_sweep(_prices(assets=("A", "B", "C")))
    assert sweep.shape[1] == 4 * 36 * 4
    assert any(c.startswith("EW|") for c in sweep.columns)
    assert sweep.notna().all().all()
