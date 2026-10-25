"""Experiment 03: is crypto volatility rough, and how rough?

Daily realised variance from 5-minute Binance bars (2019-01 to 2026-09) gives a daily
log-volatility series x_t = log sqrt(RV_t), from which the day-of-week means are removed
(crypto volatility is lower at weekends, which puts dips at 7, 14, 21, 28 days into the
structure function). Following Gatheral, Jaisson and Rosenbaum
(2018), the structure functions m(q, D) = E|x_{t+D} - x_t|^q scale like D^{q H}. The noisy
fit m(2, D) = 2 s^2 + nu^2 D^{2H} separates roughness from the estimation noise of RV.

Outputs: reports/03_structure.png, reports/03_roughness.json
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from common import (
    BLUE,
    GREY,
    INK,
    ORANGE,
    RV_SYMBOLS,
    daily_log_vol,
    log,
    save_fig,
    save_json,
    style,
)

from roughmax import estimation

LAGS = tuple(range(1, 31))
QS = (0.5, 1.0, 1.5, 2.0, 3.0)
BLOCK, BOOT = 120, 300


def zeta_q(x: np.ndarray) -> dict[str, float]:
    """Scaling exponents of m(q, D) on lags 5..30, where estimation noise matters little."""
    lags = np.arange(5, 31)
    out = {}
    for q in QS:
        m = estimation.structure_function(x, lags, q=q)
        out[str(q)] = float(np.polyfit(np.log(lags), np.log(m), 1)[0])
    return out


def block_bootstrap_h(x: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """Moving-block bootstrap 90% interval for the noisy-fit H (blocks of 120 days)."""
    n = x.size
    starts_max = n - BLOCK
    hs = []
    for _ in range(BOOT):
        starts = rng.integers(0, starts_max, size=n // BLOCK + 1)
        # Separate blocks by max(LAGS) NaNs so that no increment spans two blocks.
        gap = np.full(max(LAGS), np.nan)
        pieces = [np.concatenate([x[s : s + BLOCK], gap]) for s in starts]
        hs.append(estimation.fit_roughness(np.concatenate(pieces), LAGS, noise=True).H)
    lo, hi = np.quantile(hs, [0.05, 0.95])
    return float(lo), float(hi)


def analyse(symbol: str, rng: np.random.Generator) -> dict:
    raw = daily_log_vol(symbol)
    x_series = estimation.remove_weekly_seasonality(raw)
    x = x_series.to_numpy()
    raw_fit = estimation.fit_roughness(raw.to_numpy(), LAGS, noise=True)
    noisy = estimation.fit_roughness(x, LAGS, noise=True)
    plain = estimation.fit_roughness(x, LAGS, noise=False)
    halves = {}
    cut = len(x_series) // 2
    for name, part in (("first_half", x_series.iloc[:cut]), ("second_half", x_series.iloc[cut:])):
        fit = estimation.fit_roughness(part.to_numpy(), LAGS, noise=True)
        halves[name] = {
            "H": fit.H,
            "nu": fit.nu,
            "noise_sd": fit.noise_sd,
            "start": str(part.index[0].date()),
            "end": str(part.index[-1].date()),
        }
    ci = block_bootstrap_h(x, rng)
    log.info(
        "%s: H=%.3f nu=%.3f s=%.3f (no-noise H=%.3f) 90%% CI %s",
        symbol,
        noisy.H,
        noisy.nu,
        noisy.noise_sd,
        plain.H,
        ci,
    )
    return {
        "symbol": symbol,
        "days": int(np.isfinite(x).sum()),
        "start": str(x_series.index[0].date()),
        "end": str(x_series.index[-1].date()),
        "mean_log_vol": float(np.nanmean(x)),
        "noisy_fit": {"H": noisy.H, "nu": noisy.nu, "noise_sd": noisy.noise_sd},
        "plain_fit": {"H": plain.H, "nu": plain.nu},
        "fit_without_deseasonalisation": {
            "H": raw_fit.H,
            "nu": raw_fit.nu,
            "noise_sd": raw_fit.noise_sd,
        },
        "m2_raw": list(raw_fit.m),
        "H_bootstrap_90": ci,
        "halves": halves,
        "zeta_q": zeta_q(x),
        "lags": list(LAGS),
        "m2": list(noisy.m),
        "model_m2": noisy.model(LAGS).tolist(),
    }


def figure(results: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))
    ax = axes[0]
    colours = (INK, BLUE)
    for res, colour in zip(results, colours, strict=True):
        lags = np.array(res["lags"])
        ax.plot(
            np.log(lags),
            np.log(res["m2"]),
            "o",
            ms=3,
            color=colour,
            label=f"{res['symbol'][:3]}  H = {res['noisy_fit']['H']:.3f}",
        )
        ax.plot(np.log(lags), np.log(res["model_m2"]), color=colour, lw=1.0)
    ax.set_xlabel(r"$\log \Delta$ (days)")
    ax.set_ylabel(r"$\log m(2,\Delta)$")
    ax.plot(
        np.log(results[0]["lags"]),
        np.log(results[0]["m2_raw"]),
        "x",
        ms=3,
        color=GREY,
        label="BTC before removing the weekly cycle",
    )
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("structure function of daily log-volatility", fontsize=9)
    ax = axes[1]
    for res, colour in zip(results, colours, strict=True):
        qs = np.array([float(q) for q in res["zeta_q"]])
        ax.plot(
            qs,
            list(res["zeta_q"].values()),
            "o-",
            ms=3,
            color=colour,
            lw=1.0,
            label=res["symbol"][:3],
        )
    qline = np.linspace(0, 3.2, 10)
    ax.plot(qline, qline * results[0]["plain_fit"]["H"], color=GREY, ls=":", lw=0.9)
    ax.set_xlabel("$q$")
    ax.set_ylabel(r"scaling exponent $\zeta_q$ of $m(q,\Delta)$")
    ax.set_title(r"monofractal check: $\zeta_q \approx qH$", fontsize=9)
    ax.legend(fontsize=8)
    ax.text(0.1, 0.9 * ax.get_ylim()[1], "lags 5 to 30 days", color=ORANGE, fontsize=7)
    save_fig(fig, "03_structure.png")


if __name__ == "__main__":
    style()
    rng = np.random.default_rng(3)
    results = [analyse(symbol, rng) for symbol in RV_SYMBOLS]
    figure(results)
    save_json("03_roughness.json", {"results": results})
