import numpy as np
import pandas as pd
import pytest

from roughmax import estimation, simulate


def _bars(start, n, freq="5min", seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    close = 100 * np.exp(np.cumsum(0.001 * rng.standard_normal(n)))
    return pd.DataFrame({"close": close}, index=idx)


def test_daily_rv_skips_incomplete_days():
    full = _bars("2024-01-01", 288 * 2)
    holed = full.drop(full.index[288 + 10 : 288 + 200])
    rv = estimation.daily_realized_variance(holed, min_frac=0.9)
    assert list(rv.index) == [pd.Timestamp("2024-01-01", tz="UTC")]
    assert rv.iloc[0] > 0


def test_daily_rv_value():
    bars = _bars("2024-01-01", 288 * 3)
    rv = estimation.daily_realized_variance(bars)
    r = np.log(bars["close"]).diff()
    day2 = r[(r.index >= "2024-01-02") & (r.index < "2024-01-03")]
    assert rv.loc[pd.Timestamp("2024-01-02", tz="UTC")] == pytest.approx((day2**2).sum())


def test_structure_function_on_random_walk():
    x = np.cumsum(np.random.default_rng(2).standard_normal(50000))
    m = estimation.structure_function(x, [1, 4, 16])
    assert m == pytest.approx([1, 4, 16], rel=0.08)


def test_structure_function_ignores_nans():
    x = np.array([0.0, 1.0, np.nan, 3.0, 4.0])
    assert estimation.structure_function(x, [1])[0] == pytest.approx(1.0)


def test_fit_roughness_recovers_parameters_with_noise():
    rng = np.random.default_rng(5)
    n = 20000
    path = 0.3 * simulate.fbm(n, 0.12, 1, rng, T=float(n))[0]
    x = path + 0.1 * rng.standard_normal(n + 1)
    fit = estimation.fit_roughness(x, lags=range(1, 31), noise=True)
    assert fit.H == pytest.approx(0.12, abs=0.03)
    assert fit.nu == pytest.approx(0.3, rel=0.15)
    assert fit.noise_sd == pytest.approx(0.1, abs=0.03)


def test_fit_roughness_without_noise_is_log_linear():
    rng = np.random.default_rng(6)
    n = 20000
    x = 0.5 * simulate.fbm(n, 0.3, 1, rng, T=float(n))[0]
    fit = estimation.fit_roughness(x, lags=range(1, 31), noise=False)
    assert fit.H == pytest.approx(0.3, abs=0.03)
    assert fit.noise_sd == 0.0


def test_remove_weekly_seasonality_uses_reference_means():
    idx = pd.date_range("2024-01-01", periods=28, freq="D", tz="UTC")
    pattern = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0])
    x = pd.Series(np.tile(pattern, 4), index=idx)
    clean = estimation.remove_weekly_seasonality(x)
    assert np.allclose(clean, 0.0)
    ref = x.iloc[:14]
    shifted = x + 2.0
    out = estimation.remove_weekly_seasonality(shifted, reference=ref)
    assert np.allclose(out, 2.0)


def test_remove_weekly_seasonality_keeps_nans():
    idx = pd.date_range("2024-01-01", periods=14, freq="D", tz="UTC")
    x = pd.Series(np.arange(14.0), index=idx)
    x.iloc[3] = np.nan
    out = estimation.remove_weekly_seasonality(x)
    assert np.isnan(out.iloc[3]) and out.notna().sum() == 13
