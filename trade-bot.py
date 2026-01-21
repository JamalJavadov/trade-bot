import os
import pandas as pd
from dotenv import load_dotenv
from binance.client import Client

load_dotenv()

client = Client(
    os.getenv("Gd2QpDUgmGzo8ZLNxuC7j1Zw4kOAUIK3UC4FMEj9qNRtAuHRDhnZoQLZSVWe9TtA"),
    os.getenv("nYUaAcrVpEIIC2cC0wUM8fKTpW0h3CuapqLRFWKOEKiAZaSLxSPJ3HgVV4AIzPuR")
)

def get_ohlcv(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)

    df = pd.DataFrame(klines, columns=[
    "open_time","open","high","low","close","volume",
    "close_time","quote_asset_volume","num_trades",
    "taker_buy_base_asset_volume","taker_buy_quote_asset_volume",
    "ignore"
    ])


    df = df[["open_time","open","high","low","close","volume"]]
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)
    return df

symbol = "BTCUSDT"

frames = {
    "4h": Client.KLINE_INTERVAL_4HOUR,
    "1h": Client.KLINE_INTERVAL_1HOUR,
    "15m": Client.KLINE_INTERVAL_15MINUTE,
    "5m": Client.KLINE_INTERVAL_5MINUTE,
}

for tf, interval in frames.items():
    df = get_ohlcv(symbol, interval, limit=500)
    df.to_csv(f"{symbol}_{tf}.csv", index=False)
    print(tf, "saved:", len(df))
