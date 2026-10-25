import zipfile

import pandas as pd

from roughmax import data

ROW_MS = (
    "1704067200000,42283.58,42298.62,42261.02,42298.61,35.9,"
    "1704067259999,1519031.6,1327,23.1,980394.7,0"
)
ROW_US = (
    "1704067260000000,42298.62,42320.00,42298.61,42320.00,21.1,"
    "1704067319999999,895580.8,1348,13.4,570080.7,0"
)


def _zip(path, name, lines):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(name, "\n".join(lines) + "\n")
    return path


def test_kline_url():
    url = data.kline_url("BTCUSDT", "5m", 2024, 3)
    assert url == (
        "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/5m/BTCUSDT-5m-2024-03.zip"
    )


def test_month_range_inclusive():
    assert data.month_range("2023-11", "2024-02") == [(2023, 11), (2023, 12), (2024, 1), (2024, 2)]


def test_read_klines_handles_ms_and_us_and_header(tmp_path):
    a = _zip(tmp_path / "a.zip", "a.csv", [ROW_MS])
    header = "open_time,open,high,low,close,volume,close_time,quote_volume,count,tbb,tbq,ignore"
    b = _zip(tmp_path / "b.zip", "b.csv", [header, ROW_US])
    df = data.read_klines([b, a])
    assert list(df.index) == [
        pd.Timestamp("2024-01-01 00:00:00", tz="UTC"),
        pd.Timestamp("2024-01-01 00:01:00", tz="UTC"),
    ]
    assert df["close"].tolist() == [42298.61, 42320.00]
    assert {"open", "high", "low", "close", "volume"} <= set(df.columns)


def test_read_klines_drops_duplicates(tmp_path):
    a = _zip(tmp_path / "a.zip", "a.csv", [ROW_MS, ROW_MS])
    assert len(data.read_klines([a])) == 1
