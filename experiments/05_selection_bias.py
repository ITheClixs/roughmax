"""Experiment 05: selection bias in a correlated strategy sweep.

A sweep of 1584 trend-following configurations (time-series momentum and moving-average
crossovers, 10 crypto assets and their equal-weight basket, with and without volatility
targeting, daily or weekly rebalancing, 10 bp costs) is backtested on daily Binance data.
Under the null of no skill the standardised Sharpe ratios z_i = SR_i sqrt(T) are
approximately N(0, C), with C the correlation of the strategy returns.

The expected best null z-score E max z sets the bar for the Deflated Sharpe Ratio. It is
computed four ways: independent trials (exact and the usual Bailey-Lopez de Prado
approximation), the certified tree bracket of roughmax.selection, and Monte Carlo. The
bracket is evaluated on the whole sweep and inside each of the 11 strategy families
(one universe each), where the correlation structure is hierarchical.

Outputs: reports/05_selection.png, reports/05_split_sample.png, reports/05_selection.json
"""

from __future__ import annotations

import math
import time

import matplotlib.pyplot as plt
import numpy as np
from common import BLUE, GREY, INK, ORANGE, daily_closes, log, save_fig, save_json, style
from scipy import stats
from scipy.cluster.hierarchy import fcluster, leaves_list

from roughmax import backtest, selection, theory

MC_DRAWS = 20000


def sharpe(returns: np.ndarray) -> np.ndarray:
    return returns.mean(axis=0) / returns.std(axis=0, ddof=1)


def expected_max_table(C: np.ndarray, rng: np.random.Generator) -> dict:
    N = C.shape[0]
    t0 = time.time()
    sandwich = selection.sandwich_bounds(C)
    t_tree = time.time() - t0
    t0 = time.time()
    mc, mc_se = selection.mc_expected_max(C, MC_DRAWS, rng)
    t_mc = time.time() - t0
    return {
        "N": N,
        "iid_exact": theory.mu_n(N),
        "iid_bailey_lopez_de_prado": selection.bl_expected_max(N),
        "sandwich_lower": sandwich["lower"],
        "sandwich_upper": sandwich["upper"],
        "sandwich_lower_method": sandwich["lower_method"],
        "monte_carlo": mc,
        "monte_carlo_se": mc_se,
        "effective_trials_mc": selection.effective_trials(mc),
        "effective_trials_upper": selection.effective_trials(sandwich["upper"]),
        "seconds_sandwich": t_tree,
        "seconds_monte_carlo": t_mc,
    }


def family_brackets(C: np.ndarray, names: list[str], rng: np.random.Generator) -> list[dict]:
    """Bracket, Monte Carlo and iid value of E max z inside each universe's family."""
    universes = sorted({name.split("|")[0] for name in names})
    rows = []
    for universe in universes:
        idx = np.array([i for i, name in enumerate(names) if name.startswith(universe + "|")])
        sub = C[np.ix_(idx, idx)]
        bracket = selection.sandwich_bounds(sub)
        mc, se = selection.mc_expected_max(sub, MC_DRAWS, rng)
        rows.append(
            {
                "family": universe,
                "N": int(idx.size),
                "iid": theory.mu_n(idx.size),
                "lower": bracket["lower"],
                "upper": bracket["upper"],
                "monte_carlo": mc,
                "monte_carlo_se": se,
                "effective_trials_mc": selection.effective_trials(mc),
                "effective_trials_upper": selection.effective_trials(bracket["upper"]),
            }
        )
    return rows


def deflated(best: np.ndarray, T: int, emax: float) -> float:
    sr = float(best.mean() / best.std(ddof=1))
    return selection.deflated_sharpe(
        sr,
        T,
        float(stats.skew(best)),
        float(stats.kurtosis(best, fisher=False)),
        emax / math.sqrt(T),
    )


def split_sample(sweep, names: list[str]) -> dict:
    half = len(sweep) // 2
    first, second = sweep.iloc[:half].to_numpy(), sweep.iloc[half:].to_numpy()
    sr1, sr2 = sharpe(first) * math.sqrt(365), sharpe(second) * math.sqrt(365)
    best = int(np.argmax(sr1))
    top = np.argsort(sr1)[::-1][:20]
    return {
        "first_half": [str(sweep.index[0].date()), str(sweep.index[half - 1].date())],
        "second_half": [str(sweep.index[half].date()), str(sweep.index[-1].date())],
        "best_in_sample": names[best],
        "best_sr_in_sample": float(sr1[best]),
        "best_sr_out_of_sample": float(sr2[best]),
        "top20_mean_sr_in_sample": float(sr1[top].mean()),
        "top20_mean_sr_out_of_sample": float(sr2[top].mean()),
        "all_mean_sr_in_sample": float(sr1.mean()),
        "all_mean_sr_out_of_sample": float(sr2.mean()),
        "spearman_is_oos": float(stats.spearmanr(sr1, sr2)[0]),
        "sr1": sr1.tolist(),
        "sr2": sr2.tolist(),
    }


def figures(C: np.ndarray, table: dict, families: list[dict], split: dict) -> None:
    order = leaves_list(selection.linkage_tree(C, "complete"))
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.6), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    fig.subplots_adjust(wspace=0.5)
    ax = axes[0]
    ax.grid(False)
    im = ax.imshow(C[np.ix_(order, order)], cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title(f"correlation of {C.shape[0]} strategies\n(ordered by the dendrogram)", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, ticks=[-1, 0, 1])
    ax = axes[1]
    labels = ["independent\ntrials", "Monte\nCarlo", "certified\nbracket"]
    ax.bar([0], [table["iid_exact"]], color=GREY, width=0.6)
    ax.errorbar(
        [1], [table["monte_carlo"]], yerr=2 * table["monte_carlo_se"], fmt="o", color=INK, capsize=3
    )
    ax.vlines(2, table["sandwich_lower"], table["sandwich_upper"], color=BLUE, lw=6)
    ax.set_xticks(range(3), labels, fontsize=8)
    ax.set_ylabel("expected best null z-score")
    ax.set_title(f"whole sweep ({C.shape[0]} strategies)", fontsize=9)
    ax.set_ylim(0, 1.15 * table["iid_exact"])
    ax = axes[2]
    xs = np.arange(len(families))
    ax.vlines(
        xs,
        [f["lower"] for f in families],
        [f["upper"] for f in families],
        color=BLUE,
        lw=5,
        label="certified bracket",
    )
    ax.plot(xs, [f["monte_carlo"] for f in families], "o", ms=3, color=INK, label="Monte Carlo")
    ax.plot(
        xs, [f["iid"] for f in families], "_", ms=10, color=GREY, mew=2, label="independent trials"
    )
    ax.set_xticks(xs, [f["family"] for f in families], fontsize=7, rotation=60)
    ax.set_ylabel("expected best null z-score")
    ax.set_title("inside each strategy family (144 each)", fontsize=9)
    ax.set_ylim(0.8, 3.0)
    ax.legend(fontsize=7, loc="upper left", ncol=3, handlelength=1.2, columnspacing=0.8)
    save_fig(fig, "05_selection.png")

    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    ax.scatter(split["sr1"], split["sr2"], s=4, color=GREY, alpha=0.5, lw=0)
    lim = [min(min(split["sr1"]), min(split["sr2"])), max(max(split["sr1"]), max(split["sr2"]))]
    ax.plot(lim, lim, color=GREY, ls=":", lw=0.8)
    ax.scatter(
        [split["best_sr_in_sample"]],
        [split["best_sr_out_of_sample"]],
        color=ORANGE,
        s=25,
        zorder=3,
        label="best in-sample configuration",
    )
    ax.set_xlabel(f"annualised Sharpe, {split['first_half'][0][:4]}-{split['first_half'][1][:4]}")
    ax.set_ylabel(f"annualised Sharpe, {split['second_half'][0][:4]}-{split['second_half'][1][:4]}")
    ax.legend(fontsize=7, loc="lower right")
    save_fig(fig, "05_split_sample.png")


if __name__ == "__main__":
    style()
    rng = np.random.default_rng(5)
    closes = daily_closes()
    sweep = backtest.build_sweep(closes)
    names = list(sweep.columns)
    returns = sweep.to_numpy()
    T, N = returns.shape
    log.info("sweep: %d days x %d configurations", T, N)
    C = np.corrcoef(returns, rowvar=False)
    table = expected_max_table(C, rng)
    log.info("E max: %s", {k: v for k, v in table.items() if isinstance(v, float)})
    sr = sharpe(returns)
    best = int(np.argmax(sr))
    dsr = {
        name: deflated(returns[:, best], T, table[key])
        for name, key in (
            ("iid", "iid_exact"),
            ("certified_upper", "sandwich_upper"),
            ("monte_carlo", "monte_carlo"),
        )
    }
    families = family_brackets(C, names, rng)
    split = split_sample(sweep, names)
    figures(C, table, families, split)
    clusters = fcluster(selection.linkage_tree(C, "complete"), t=11, criterion="maxclust")
    save_json(
        "05_selection.json",
        {
            "days": T,
            "start": str(sweep.index[0].date()),
            "end": str(sweep.index[-1].date()),
            "configurations": N,
            "mean_abs_correlation": float(np.abs(C[np.triu_indices(N, 1)]).mean()),
            "expected_max": table,
            "best": {
                "name": names[best],
                "annualised_sharpe": float(sr[best] * math.sqrt(365)),
                "z": float(sr[best] * math.sqrt(T)),
                "deflated_sharpe": dsr,
            },
            "families": families,
            "cluster_sizes_at_11": np.bincount(clusters)[1:].tolist(),
            "split_sample": {k: v for k, v in split.items() if k not in ("sr1", "sr2")},
        },
    )
