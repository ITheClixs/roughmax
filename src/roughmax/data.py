"""Download and parse Binance public spot klines (https://data.binance.vision).

No API key is needed. Files are monthly zip archives of headerless CSV rows; from
January 2025 the spot timestamps are in microseconds instead of milliseconds, and a few
archives carry a header row. Both cases are handled here.
"""

from __future__ import annotations

import io
import logging
import urllib.error
import urllib.request
import zipfile
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]
_MICROSECOND_THRESHOLD = 10**14  # epoch values above this are microseconds


def kline_url(symbol: str, interval: str, year: int, month: int) -> str:
    """URL of the monthly kline archive for one symbol and bar interval."""
    name = f"{symbol}-{interval}-{year:04d}-{month:02d}.zip"
    return f"{BASE_URL}/{symbol}/{interval}/{name}"


def month_range(start: str, end: str) -> list[tuple[int, int]]:
    """Inclusive list of (year, month) from 'YYYY-MM' to 'YYYY-MM'."""
    y0, m0 = (int(v) for v in start.split("-"))
    y1, m1 = (int(v) for v in end.split("-"))
    months = []
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        months.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return months


def download_klines(
    symbol: str, interval: str, start: str, end: str, cache_dir: Path, timeout: float = 60.0
) -> list[Path]:
    """Download monthly archives into cache_dir (skipping files already present).

    Months that do not exist on the server (symbol not yet listed) are skipped with a
    warning. Returns the paths of all available archives in chronological order.
    """
    target = Path(cache_dir) / symbol / interval
    target.mkdir(parents=True, exist_ok=True)
    paths = []
    for year, month in month_range(start, end):
        url = kline_url(symbol, interval, year, month)
        path = target / url.rsplit("/", 1)[1]
        if not path.exists():
            try:
                with urllib.request.urlopen(url, timeout=timeout) as resp:
                    payload = resp.read()
            except urllib.error.HTTPError as err:
                if err.code == 404:
                    logger.warning("no archive for %s %s %04d-%02d", symbol, interval, year, month)
                    continue
                raise
            tmp = path.with_suffix(".part")
            tmp.write_bytes(payload)
            tmp.rename(path)
        paths.append(path)
    return paths


def _read_one(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"expected one csv in {path}, found {names}")
        raw = zf.read(names[0]).decode()
    frame = pd.read_csv(io.StringIO(raw), header=None, names=COLUMNS)
    if not str(frame.iloc[0, 0]).strip().isdigit():
        frame = frame.iloc[1:]
    return frame


def read_klines(paths: Iterable[Path]) -> pd.DataFrame:
    """Parse kline archives into one frame indexed by the UTC bar open time."""
    frames = [_read_one(Path(p)) for p in paths]
    if not frames:
        raise ValueError("no kline archives given")
    df = pd.concat(frames, ignore_index=True)
    stamps = df["open_time"].astype(np.int64).to_numpy()
    millis = np.where(stamps > _MICROSECOND_THRESHOLD, stamps // 1000, stamps)
    index = pd.to_datetime(millis, unit="ms", utc=True)
    numeric = ["open", "high", "low", "close", "volume", "quote_volume"]
    out = df[numeric].astype(float)
    out["trades"] = df["trades"].astype(np.int64).to_numpy()
    out.index = index
    out.index.name = "open_time"
    out = out[~out.index.duplicated(keep="first")].sort_index()
    return out
