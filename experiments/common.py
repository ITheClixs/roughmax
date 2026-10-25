"""Shared paths, data loading and plot style for the experiment scripts."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from roughmax import data, estimation  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
START, END = "2019-01", "2026-09"
RV_SYMBOLS = ("BTCUSDT", "ETHUSDT")
SWEEP_SYMBOLS = (
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "LTCUSDT",
    "LINKUSDT",
    "XLMUSDT",
    "TRXUSDT",
    "ETCUSDT",
)
SWEEP_START = "2019-02"

# Okabe-Ito colours (colour-blind safe), black for the main series.
INK, BLUE, ORANGE, GREEN, PURPLE, GREY = (
    "#111111",
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#8a8a8a",
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("experiments")


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e6e6e6",
            "grid.linewidth": 0.6,
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.4,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def save_json(name: str, payload: dict) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    log.info("wrote %s", path.relative_to(ROOT))
    return path


def save_fig(fig, name: str) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / name
    fig.savefig(path)
    plt.close(fig)
    log.info("wrote %s", path.relative_to(ROOT))
    return path


def daily_log_vol(symbol: str) -> pd.Series:
    """Daily log realised volatility from 5-minute bars (cached as a processed CSV)."""
    csv = PROCESSED / f"{symbol.lower()}_daily_rv.csv"
    if csv.exists():
        rv = pd.read_csv(csv, index_col=0, parse_dates=True)["rv"]
    else:
        paths = data.download_klines(symbol, "5m", START, END, RAW)
        rv = estimation.daily_realized_variance(data.read_klines(paths))
        PROCESSED.mkdir(parents=True, exist_ok=True)
        rv.to_csv(csv)
    full = rv.asfreq("D")  # missing days become NaN, never interpolated
    full = full.where(full > 0)
    return 0.5 * np.log(full)


def daily_closes() -> pd.DataFrame:
    """Daily close prices of the sweep universe (cached as a processed CSV)."""
    csv = PROCESSED / "daily_closes.csv"
    if csv.exists():
        return pd.read_csv(csv, index_col=0, parse_dates=True)
    closes = {}
    for symbol in SWEEP_SYMBOLS:
        paths = data.download_klines(symbol, "1d", SWEEP_START, END, RAW)
        closes[symbol.removesuffix("USDT")] = data.read_klines(paths)["close"]
    frame = pd.DataFrame(closes)
    frame.index = frame.index.floor("D")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv)
    return frame
