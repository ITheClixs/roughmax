"""Experiment 01: what the paper's formulas say, in numbers and pictures.

Outputs: reports/01_bounds.png, reports/01_grid_fraction.png, reports/01_theory.json
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
from common import BLUE, GREY, INK, ORANGE, save_fig, save_json, style
from scipy.special import gamma

from roughmax import theory


def bisewski_lower(H: float) -> float:
    """Explicit lower bound of Bisewski (2023), Cor. 2(i), with T = 1."""
    c_h = math.sqrt(gamma(0.5 + H) * gamma(2 - 2 * H) / (2 * H * gamma(1.5 - H)))
    return 1.0 / (math.sqrt(2 * math.pi) * c_h * H * (H + 0.5) * math.sin(math.pi * (H + 0.5) / 2))


def bmnz_upper(H: float) -> float:
    """Upper bound 1.695 / sqrt(H) of Borovkov, Mishura, Novikov and Zhitlukhin (2018)."""
    return 1.695 / math.sqrt(H)


def figure_bounds() -> dict:
    hs = np.geomspace(1e-6, 0.4, 70)
    up = np.array([theory.upper_bound(h) for h in hs]) * np.sqrt(hs)
    lo = np.array([theory.lower_bound(h) for h in hs]) * np.sqrt(hs)
    bis = np.array([bisewski_lower(h) for h in hs]) * np.sqrt(hs)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.axhline(math.sqrt(2), color=GREY, lw=0.8, ls=":")
    ax.plot(hs, np.full_like(hs, 1.695), color=ORANGE, ls="--", lw=1.1)
    ax.plot(hs, bis, color=BLUE, ls="--", lw=1.1)
    ax.plot(hs, up, color=INK)
    ax.plot(hs, lo, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("Hurst index $H$")
    ax.set_ylabel(r"$\sqrt{H}\times$ bound on $\mathbb{E}\sup_{[0,1]}B^H$")
    ax.set_ylim(0.6, 1.85)
    ax.text(2e-6, 1.43, r"$\sqrt{2}$ (the limit)", color=GREY, fontsize=8)
    ax.text(2e-6, 1.71, "previous upper bound", color=ORANGE, fontsize=8)
    ax.text(2e-6, 1.06, "previous lower bound", color=BLUE, fontsize=8)
    ax.text(2e-6, 1.33, "new lower bound", color=INK, fontsize=8)
    ax.text(1.5e-4, 1.48, "new upper bound", color=INK, fontsize=8)
    save_fig(fig, "01_bounds.png")
    return {
        "H": hs.tolist(),
        "sqrtH_upper": up.tolist(),
        "sqrtH_lower": lo.tolist(),
    }


def figure_grid_fraction() -> dict:
    kappa = np.linspace(0, 6, 400)
    frac = np.array([theory.grid_limit(k) for k in kappa]) / math.sqrt(2)
    hs = np.linspace(0.01, 0.5, 120)
    log10n = np.linspace(1, 12, 150)
    grid = np.array([[theory.captured_fraction(h, 10**ln) for ln in log10n] for h in hs])
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.1))
    ax = axes[0]
    ax.plot(kappa, frac, color=INK)
    ax.axvline(0.5, color=GREY, lw=0.8, ls=":")
    ax.set_xlabel(r"$\kappa = H\log n$")
    ax.set_ylabel("fraction of the supremum captured")
    ax.text(0.55, 0.15, r"$\kappa=\frac{1}{2}$", color=GREY, fontsize=8)
    ax.set_ylim(0, 1.02)
    ax = axes[1]
    ax.grid(False)
    mesh = ax.pcolormesh(log10n, hs, grid, cmap="Greys", vmin=0, vmax=1, shading="auto")
    cs = ax.contour(log10n, hs, grid, levels=[0.5, 0.8, 0.9, 0.95], colors=[BLUE], linewidths=0.9)
    ax.clabel(cs, fmt="%.2f", fontsize=7)
    ax.set_xlabel(r"$\log_{10} n$ (grid points)")
    ax.set_ylabel("Hurst index $H$")
    fig.colorbar(mesh, ax=ax, fraction=0.05, pad=0.02)
    save_fig(fig, "01_grid_fraction.png")
    return {}


def headline_numbers() -> dict:
    out = {}
    for H in (0.05, 0.1, 0.15):
        out[f"H={H}"] = {
            "upper_bound": theory.upper_bound(H),
            "lower_bound": theory.lower_bound(H),
            "asymptotic": theory.asymptotic_sup(H),
            "captured_fraction_daily_1y": theory.captured_fraction(H, 365),
            "captured_fraction_5min_1y": theory.captured_fraction(H, 365 * 288),
            "points_for_90pct": math.exp(theory.log_points_needed(H, 0.9)),
            "points_for_95pct": math.exp(theory.log_points_needed(H, 0.95)),
        }
    return out


if __name__ == "__main__":
    style()
    payload = {"bounds_curve": figure_bounds(), "headline": headline_numbers()}
    figure_grid_fraction()
    save_json("01_theory.json", payload)
