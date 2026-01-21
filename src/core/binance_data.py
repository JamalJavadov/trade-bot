from __future__ import annotations

import os
import time
from typing import Dict, Any, List, Optional

import pandas as pd
from dotenv import load_dotenv
from binance.client import Client

load_dotenv()

API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")

client = Client(API_KEY, API_SECRET)


def ping() -> dict:
    return client.ping()


def server_time() -> dict:
    return client.get_server_time()


def futures_exchange_info() -> Dict[str, Any]:
    return client.futures_exchange_info()


def _symbol_meta(symbol: str) -> Optional[Dict[str, Any]]:
    info = futures_exchange_info()
    for s in info.get("symbols", []):
        if s.get("symbol") == symbol:
            return s
    return None


def is_valid_usdtm_perp(symbol: str) -> bool:
    m = _symbol_meta(symbol)
    if not m:
        return False
    return (
        m.get("status") == "TRADING"
        and m.get("quoteAsset") == "USDT"
        and m.get("contractType") == "PERPETUAL"
    )


def list_usdtm_perp_symbols() -> List[str]:
    info = futures_exchange_info()
    out = []
    for s in info.get("symbols", []):
        if (
            s.get("status") == "TRADING"
            and s.get("quoteAsset") == "USDT"
            and s.get("contractType") == "PERPETUAL"
        ):
            out.append(s["symbol"])
    out.sort()
    return out


def list_usdtm_perp_symbols_by_volume(limit: int = 200) -> List[str]:
    info = futures_exchange_info()
    valid_symbols = {
        s.get("symbol")
        for s in info.get("symbols", [])
        if s.get("status") == "TRADING"
        and s.get("quoteAsset") == "USDT"
        and s.get("contractType") == "PERPETUAL"
    }

    tickers = client.futures_ticker()
    volumes = {}
    for t in tickers:
        symbol = t.get("symbol")
        if not symbol or symbol not in valid_symbols:
            continue
        try:
            volumes[symbol] = float(t.get("quoteVolume", 0.0))
        except (TypeError, ValueError):
            volumes[symbol] = 0.0

    ranked = sorted(volumes.items(), key=lambda item: item[1], reverse=True)
    return [symbol for symbol, _ in ranked[:limit]]


def get_symbol_filters(symbol: str) -> Dict[str, float]:
    m = _symbol_meta(symbol) or {}
    tick = 0.0
    step = 0.0
    min_qty = 0.0

    for f in (m.get("filters") or []):
        t = f.get("filterType")
        if t == "PRICE_FILTER":
            tick = float(f.get("tickSize", 0.0))
        elif t == "LOT_SIZE":
            step = float(f.get("stepSize", 0.0))
            min_qty = float(f.get("minQty", 0.0))

    return {"tickSize": tick, "stepSize": step, "minQty": min_qty}


def get_ohlcv(symbol: str, interval: str, limit: int = 500, sleep_ms: int = 0) -> pd.DataFrame:
    """
    USDT-M Futures klines.
    Klines fields are typically 12:
    [openTime, open, high, low, close, volume, closeTime, quoteAssetVolume, numTrades,
     takerBuyBaseVol, takerBuyQuoteVol, ignore]
    """
    if sleep_ms > 0:
        time.sleep(sleep_ms / 1000.0)

    kl = client.futures_klines(symbol=symbol, interval=interval, limit=limit)

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "num_trades",
        "taker_base_vol", "taker_quote_vol", "ignore"
    ]

    df = pd.DataFrame(kl, columns=cols)

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    return df
