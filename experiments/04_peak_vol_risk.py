"""Experiment 04: peak-volatility risk of BTC and ETH from a three-parameter rough model.

For each horizon of n days the expected peak excursion of daily log-volatility,
E max_{1<=i<=n} (x_{t+i} - x_t), is measured on the data and predicted by the rough model
x = nu B^H + noise, with (H, nu, s) fitted only on the structure function at lags of at
most 30 days. Two non-rough benchmarks use the same data and the same lags: Brownian
log-volatility (H = 1/2, same one-day variance) and a mean-reverting Ornstein-Uhlenbeck
log-volatility fitted to the same structure function.

Out of sample: parameters are fitted on the first half of the sample and the predicted
peaks and the noise-aware quantile bounds of roughmax.peakrisk are checked on the second
half.

Outputs: reports/04_peak_vol.png, reports/04_out_of_sample.png, reports/04_peak_vol.json
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
from common import (
    BLUE,
    GREEN,
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

from roughmax import estimation, peakrisk, theory

HORIZONS = (1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90, 120, 180, 250)
LAGS = tuple(range(1, 31))
MC_PATHS = 20000
ALPHAS = (0.10, 0.05, 0.01)
COVERAGE_HORIZONS = (7, 30, 90)


def predictions(
    H: float, nu: float, s: float, ou: estimation.OUFit, rng: np.random.Generator
) -> dict:
    out = {
        "rough": [],
        "rough_se": [],
        "rough_latent": [],
        "brownian": [],
        "ou": [],
        "asymptotic": [],
    }
    for n in HORIZONS:
        out["ou"].append(
            float(peakrisk.ou_grid_peak_samples(ou.theta, ou.sigma, n, MC_PATHS, rng).mean())
        )
        mean, se = peakrisk.expected_grid_peak(H, nu, n, s, "mc", MC_PATHS, rng)
        out["rough"].append(mean)
        out["rough_se"].append(se)
        out["rough_latent"].append(
            peakrisk.expected_grid_peak(H, nu, n, 0.0, "mc", MC_PATHS, rng)[0]
        )
        out["brownian"].append(peakrisk.expected_grid_peak(0.5, nu, n, s, "mc", MC_PATHS, rng)[0])
        out["asymptotic"].append(peakrisk.expected_grid_peak(H, nu, n, method="asymptotic")[0])
    return out


def empirical(x: np.ndarray) -> dict:
    stats = [peakrisk.empirical_grid_peak(x, n) for n in HORIZONS]
    return {
        "mean": [s["mean"] for s in stats],
        "se": [s["se"] for s in stats],
        "count": [s["count"] for s in stats],
    }


def errors(pred: list[float], emp: list[float]) -> dict[str, float]:
    """Mean absolute error (log-vol units) and mean relative error for horizons >= 5 days."""
    p, e = np.asarray(pred), np.asarray(emp)
    long = np.asarray(HORIZONS) >= 5
    return {
        "mae": float(np.mean(np.abs(p - e))),
        "mean_rel_error_h5plus": float(np.mean(np.abs(p[long] - e[long]) / e[long])),
    }


def coverage(
    x_test: np.ndarray,
    H: float,
    nu: float,
    s: float,
    ou: estimation.OUFit,
    rng: np.random.Generator,
) -> list[dict]:
    """Exceedance rates of three (1-alpha) peak quantiles on the test windows.

    'bound' is the rigorous bound of peakrisk.grid_peak_bound; 'rough' and 'ou' are Monte
    Carlo quantiles of the fitted models. Windows overlap, so exceedances are autocorrelated
    and the rates are descriptive, not independent Bernoulli trials.
    """
    rows = []
    for n in COVERAGE_HORIZONS:
        peaks = peakrisk.empirical_peaks(x_test, n)
        rough_sim = peakrisk.grid_peak_samples(H, nu, n, MC_PATHS, rng, s)
        ou_sim = peakrisk.ou_grid_peak_samples(ou.theta, ou.sigma, n, MC_PATHS, rng)
        for alpha in ALPHAS:
            levels = {
                "bound": peakrisk.grid_peak_bound(H, nu, n, alpha, noise_sd=s),
                "rough": float(np.quantile(rough_sim, 1 - alpha)),
                "ou": float(np.quantile(ou_sim, 1 - alpha)),
            }
            rows.append(
                {
                    "horizon": n,
                    "alpha": alpha,
                    "windows": int(peaks.size),
                    **{f"{k}_level": v for k, v in levels.items()},
                    **{f"{k}_exceedance": float(np.mean(peaks > v)) for k, v in levels.items()},
                }
            )
    return rows


def analyse(symbol: str, rng: np.random.Generator) -> dict:
    raw = daily_log_vol(symbol)
    series = estimation.remove_weekly_seasonality(raw)
    x = series.to_numpy()
    fit = estimation.fit_roughness(x, LAGS, noise=True)
    ou_full = estimation.fit_ou(x, LAGS)
    full = {
        "empirical": empirical(x),
        "model": predictions(fit.H, fit.nu, fit.noise_sd, ou_full, rng),
    }
    cut = len(series) // 2
    train = estimation.remove_weekly_seasonality(raw.iloc[:cut]).to_numpy()
    test = estimation.remove_weekly_seasonality(raw.iloc[cut:], reference=raw.iloc[:cut]).to_numpy()
    fit_tr = estimation.fit_roughness(train, LAGS, noise=True)
    ou_tr = estimation.fit_ou(train, LAGS)
    oos_pred = predictions(fit_tr.H, fit_tr.nu, fit_tr.noise_sd, ou_tr, rng)
    oos_emp = empirical(test)
    gap = peakrisk.continuous_peak_gap(fit.H, fit.nu, 30)
    result = {
        "symbol": symbol,
        "fit": {"H": fit.H, "nu": fit.nu, "noise_sd": fit.noise_sd},
        "ou_fit": {
            "theta": ou_full.theta,
            "sigma": ou_full.sigma,
            "half_life_days": float(np.log(2) / ou_full.theta),
        },
        "full_sample": full,
        "out_of_sample": {
            "train": {
                "start": str(series.index[0].date()),
                "end": str(series.index[cut - 1].date()),
                "H": fit_tr.H,
                "nu": fit_tr.nu,
                "noise_sd": fit_tr.noise_sd,
                "ou_theta": ou_tr.theta,
                "ou_sigma": ou_tr.sigma,
            },
            "test": {"start": str(series.index[cut].date()), "end": str(series.index[-1].date())},
            "empirical": oos_emp,
            "model": oos_pred,
            "errors": {
                name: errors(oos_pred[name], oos_emp["mean"])
                for name in ("rough", "ou", "brownian", "asymptotic")
            },
            "coverage": coverage(test, fit_tr.H, fit_tr.nu, fit_tr.noise_sd, ou_tr, rng),
        },
        "intraday_gap_30d": {
            **gap,
            "vol_ratio_sup_asymptotic_over_daily_grid": math.exp(
                gap["sup_asymptotic"] - gap["grid_asymptotic"]
            ),
            "captured_fraction_daily_grid": theory.captured_fraction(fit.H, 30),
        },
    }
    log.info("%s out of sample: %s", symbol, result["out_of_sample"]["errors"])
    return result


def panel(ax, title: str, emp: dict, model: dict) -> None:
    h = np.array(HORIZONS, dtype=float)
    ax.errorbar(
        h,
        emp["mean"],
        yerr=2 * np.array(emp["se"]),
        fmt="o",
        ms=3,
        color=INK,
        capsize=2,
        lw=0.8,
        label="data (daily log-vol)",
    )
    ax.plot(h, model["rough"], color=BLUE, label="rough model (fitted H)")
    ax.plot(h, model["ou"], color=GREEN, ls="-.", label="mean-reverting OU log-vol")
    ax.plot(h, model["brownian"], color=ORANGE, ls="--", label="Brownian log-vol (H = 1/2)")
    ax.plot(h, model["asymptotic"], color=GREY, ls=":", label="paper's first-order formula")
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 2.5)
    ax.set_xlabel("horizon (days)")
    ax.set_title(title, fontsize=9)


def figures(results: list[dict]) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(9.6, 3.3), sharey=True)
    for ax, res in zip(axes, results, strict=True):
        panel(
            ax,
            f"{res['symbol'][:3]}, full sample, H = {res['fit']['H']:.3f}",
            res["full_sample"]["empirical"],
            res["full_sample"]["model"],
        )
    axes[0].set_ylabel("expected peak of log-volatility")
    axes[0].legend(fontsize=7, loc="upper left")
    save_fig(fig, "04_peak_vol.png")
    fig, axes = plt.subplots(1, len(results), figsize=(9.6, 3.3), sharey=True)
    for ax, res in zip(axes, results, strict=True):
        oos = res["out_of_sample"]
        panel(
            ax,
            f"{res['symbol'][:3]}: fit {oos['train']['start'][:4]}-{oos['train']['end'][:4]}, "
            f"test {oos['test']['start'][:4]}-{oos['test']['end'][:4]}",
            oos["empirical"],
            oos["model"],
        )
    axes[0].set_ylabel("expected peak of log-volatility")
    axes[0].legend(fontsize=7, loc="upper left")
    save_fig(fig, "04_out_of_sample.png")


if __name__ == "__main__":
    style()
    rng = np.random.default_rng(4)
    results = [analyse(symbol, rng) for symbol in RV_SYMBOLS]
    figures(results)
    save_json("04_peak_vol.json", {"horizons": list(HORIZONS), "results": results})
