"""Download the public Binance data used by the experiments and cache processed CSVs."""

from common import RV_SYMBOLS, daily_closes, daily_log_vol, log

if __name__ == "__main__":
    for symbol in RV_SYMBOLS:
        x = daily_log_vol(symbol)
        log.info("%s: %d days of log realised volatility", symbol, x.notna().sum())
    closes = daily_closes()
    log.info("daily closes: %s rows x %s assets", *closes.shape)
