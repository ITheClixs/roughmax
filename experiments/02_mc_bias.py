"""Experiment 02: Monte Carlo grid maxima of rough paths against the proved bounds.

For H in {0.05, 0.1, 0.2} the expected maximum over n = 2^6 ... 2^20 grid points is
estimated by exact simulation and compared with the rigorous bracket
lower_bound(H) <= E sup B^H <= upper_bound(H). The bias is then extrapolated with a fitted
power law and with Richardson steps at the paper's rate n^{-H} and the Brownian rate n^{-1/2}.

Outputs: reports/02_mc_vs_bounds.png, reports/02_mc_bias.json
"""

from __future__ import annotations

import math
import time

import matplotlib.pyplot as plt
import numpy as np
from common import BLUE, GREY, INK, ORANGE, log, save_fig, save_json, style

from roughmax import extremes, theory

HS = (0.05, 0.1, 0.2)
LOG2_NS = tuple(range(6, 21, 2))
FIT_FROM = 10  # fit the power law on n >= 2^10


def paths_for(log2n: int) -> int:
    return {6: 20000, 8: 20000, 10: 12000, 12: 8000, 14: 4000, 16: 2400, 18: 1200, 20: 700}[log2n]


def run_one(H: float, rng: np.random.Generator) -> dict:
    ns, means, ses = [], [], []
    for k in LOG2_NS:
        t0 = time.time()
        mean, se = extremes.grid_max_mc(H, 2**k, paths_for(k), rng)
        log.info("H=%.2f n=2^%d mean=%.4f se=%.4f (%.1fs)", H, k, mean, se, time.time() - t0)
        ns.append(2**k)
        means.append(mean)
        ses.append(se)
    ns_a, means_a, ses_a = (np.array(v, dtype=float) for v in (ns, means, ses))
    use = ns_a >= 2**FIT_FROM
    fit = extremes.fit_power_bias(ns_a[use], means_a[use], ses_a[use])
    lo, up = theory.lower_bound(H), theory.upper_bound(H)
    rich_h = extremes.richardson(ns[-3], means[-3], ns[-1], means[-1], H)
    rich_half = extremes.richardson(ns[-3], means[-3], ns[-1], means[-1], 0.5)
    return {
        "H": H,
        "n": ns,
        "mean": means,
        "se": ses,
        "paths": [paths_for(k) for k in LOG2_NS],
        "lower_bound": lo,
        "upper_bound": up,
        "asymptotic_sup": theory.asymptotic_sup(H),
        "grid_asymptotic": [theory.grid_max_asymptotic(H, n) for n in ns],
        "fit": fit,
        "richardson_rate_H": rich_h,
        "richardson_rate_half": rich_half,
        "finest_grid_below_lower_bound": bool(means[-1] + 3 * ses[-1] < lo),
        "fitted_limit_inside_bracket": bool(
            lo - 3 * fit["A_se"] <= fit["A"] <= up + 3 * fit["A_se"]
        ),
        "fraction_of_lower_bound_at_finest_grid": means[-1] / lo,
    }


def figure(results: list[dict]) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(10.5, 3.2))
    for ax, res in zip(axes, results, strict=True):
        n = np.array(res["n"], dtype=float)
        ax.axhspan(res["lower_bound"], res["upper_bound"], color=GREY, alpha=0.18, lw=0)
        ax.text(
            2**6.2,
            res["upper_bound"],
            r"proved range of $\mathbb{E}\sup$",
            va="top",
            fontsize=7,
            color=GREY,
        )
        ax.errorbar(
            np.log2(n),
            res["mean"],
            yerr=2 * np.array(res["se"]),
            fmt="o",
            ms=3,
            color=INK,
            capsize=2,
            lw=0.8,
            label="Monte Carlo grid maximum",
        )
        fit = res["fit"]
        xs = np.linspace(6, 40, 200)
        ax.plot(
            xs,
            fit["A"] - fit["c"] * 2.0 ** (-fit["r"] * xs),
            color=BLUE,
            lw=1.0,
            label=rf"fit $A-cn^{{-r}}$, $r={fit['r']:.3f}$",
        )
        ax.plot(
            np.log2(n),
            res["grid_asymptotic"],
            color=ORANGE,
            ls="--",
            lw=1.0,
            label=r"first-order $\Lambda(H\log n)/\sqrt{H}$",
        )
        ax.set_title(f"H = {res['H']}", fontsize=9)
        ax.set_xlabel(r"$\log_2 n$")
        ax.set_xlim(5, 40)
    axes[0].set_ylabel(r"$\mathbb{E}\max_{i\leq n} B^H_{i/n}$")
    axes[0].legend(fontsize=7, loc="lower right")
    save_fig(fig, "02_mc_vs_bounds.png")


if __name__ == "__main__":
    style()
    rng = np.random.default_rng(20261025)
    results = [run_one(H, rng) for H in HS]
    figure(results)
    save_json("02_mc_bias.json", {"results": results, "note": "se = Monte Carlo standard error"})
    for res in results:
        log.info(
            "H=%.2f finest=%.3f lower=%.3f upper=%.3f fitted A=%.3f r=%.3f rich(H)=%.3f "
            "rich(1/2)=%.3f",
            res["H"],
            res["mean"][-1],
            res["lower_bound"],
            res["upper_bound"],
            res["fit"]["A"],
            res["fit"]["r"],
            res["richardson_rate_H"],
            res["richardson_rate_half"],
        )
    log.info("sqrt(2/H): %s", {h: round(math.sqrt(2 / h), 3) for h in HS})
