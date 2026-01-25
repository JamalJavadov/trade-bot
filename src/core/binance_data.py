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

print("DEBUG: Loading binance_data.py V2 with FIX")

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


def get_open_orders(symbol: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Return open futures orders.

    Args:
        symbol: Optional symbol filter.
    """
    try:
        if symbol:
            orders = client.futures_open_orders(symbol=symbol)
        else:
            orders = client.futures_open_orders()
    except Exception:
        return []

    normalized = []
    for order in orders or []:
        try:
            normalized.append(
                {
                    "symbol": order.get("symbol"),
                    "side": order.get("side"),
                    "type": order.get("type"),
                    "price": float(order.get("price", 0.0)),
                    "stopPrice": float(order.get("stopPrice", 0.0)),
                    "origQty": float(order.get("origQty", 0.0)),
                    "status": order.get("status"),
                    "reduceOnly": bool(order.get("reduceOnly", False)),
                    "timeInForce": order.get("timeInForce"),
                }
            )
        except (TypeError, ValueError):
            continue
    return normalized


def get_open_positions(symbol: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Return open futures positions (positionAmt != 0).

    Args:
        symbol: Optional symbol filter.
    """
    try:
        if symbol:
            positions = client.futures_position_information(symbol=symbol)
        else:
            positions = client.futures_position_information()
    except Exception:
        return []

    normalized = []
    for pos in positions or []:
        try:
            amt = float(pos.get("positionAmt", 0.0))
        except (TypeError, ValueError):
            continue
        if amt == 0:
            continue
        try:
            entry_price = float(pos.get("entryPrice", 0.0))
            mark_price = float(pos.get("markPrice", 0.0))
            leverage = int(float(pos.get("leverage", 0)))
            liquidation_price = float(pos.get("liquidationPrice", 0.0))
            unrealized_profit = float(pos.get("unRealizedProfit", 0.0))
            notional = float(pos.get("notional", 0.0))
            initial_margin = float(pos.get("initialMargin", 0.0))
        except (TypeError, ValueError):
            continue
        normalized.append(
            {
                "symbol": pos.get("symbol"),
                "side": "LONG" if amt > 0 else "SHORT",
                "position_amt": amt,
                "entry_price": entry_price,
                "mark_price": mark_price,
                "leverage": leverage,
                "liquidation_price": liquidation_price,
                "unrealized_profit": unrealized_profit,
                "notional": notional,
                "initial_margin": initial_margin,
            }
        )
    return normalized


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

    # 1. Convert numeric columns (coercing errors)
    # We use a dictionary comprehension to process all numeric cols at once
    numeric_cols = ["open", "high", "low", "close", "volume"]
    numeric_data = {c: pd.to_numeric(df[c], errors="coerce") for c in numeric_cols}
    
    # 2. Convert time columns (explicitly handling units and timezone)
    # We maintain the same column names but create new Series
    time_data = {
        "open_time": pd.to_datetime(df["open_time"], unit="ms", utc=True),
        "close_time": pd.to_datetime(df["close_time"], unit="ms", utc=True)
    }
    
    # 3. Assign all processed columns back to the DataFrame
    # Using assign() returns a new object, avoiding ChainedAssignment and schema warnings
    df = df.assign(**numeric_data, **time_data)

    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    return df


# =============================================================================
# Extended Data Functions for Deep Analysis
# =============================================================================

def get_funding_rate(symbol: str) -> Dict[str, Any]:
    """
    Get current funding rate for a futures symbol.
    
    Funding rate indicates market sentiment:
    - Positive = Longs pay shorts (bullish sentiment, potential for correction)
    - Negative = Shorts pay longs (bearish sentiment, potential for bounce)
    
    Returns:
        Dict with fundingRate, fundingTime, and calculated annualized rate
    """
    try:
        data = client.futures_funding_rate(symbol=symbol, limit=1)
        if data:
            latest = data[-1]
            funding_rate = float(latest.get("fundingRate", 0))
            funding_time = int(latest.get("fundingTime", 0))
            
            # Annualized rate (3 fundings per day * 365 days)
            annualized = funding_rate * 3 * 365 * 100
            
            return {
                "symbol": symbol,
                "fundingRate": funding_rate,
                "fundingRatePct": funding_rate * 100,
                "fundingTime": funding_time,
                "annualizedPct": annualized,
                "sentiment": "BULLISH_CROWDED" if funding_rate > 0.0001 else 
                            "BEARISH_CROWDED" if funding_rate < -0.0001 else "NEUTRAL",
            }
    except Exception as e:
        pass
    
    return {
        "symbol": symbol,
        "fundingRate": 0.0,
        "fundingRatePct": 0.0,
        "fundingTime": 0,
        "annualizedPct": 0.0,
        "sentiment": "UNKNOWN",
    }


def get_open_interest(symbol: str) -> Dict[str, Any]:
    """
    Get current open interest for a futures symbol.
    
    Open interest shows total outstanding contracts:
    - Rising OI + Rising Price = New longs entering (bullish)
    - Rising OI + Falling Price = New shorts entering (bearish)
    - Falling OI + Rising Price = Shorts covering (weak rally)
    - Falling OI + Falling Price = Longs closing (weak decline)
    
    Returns:
        Dict with openInterest value and quote value
    """
    try:
        data = client.futures_open_interest(symbol=symbol)
        if data:
            oi = float(data.get("openInterest", 0))
            
            # Get current price for quote calculation
            ticker = client.futures_symbol_ticker(symbol=symbol)
            price = float(ticker.get("price", 0)) if ticker else 0
            
            return {
                "symbol": symbol,
                "openInterest": oi,
                "openInterestQuote": oi * price,
                "time": int(data.get("time", 0)),
            }
    except Exception as e:
        pass
    
    return {
        "symbol": symbol,
        "openInterest": 0.0,
        "openInterestQuote": 0.0,
        "time": 0,
    }


def get_open_interest_history(symbol: str, period: str = "5m", limit: int = 100) -> pd.DataFrame:
    """
    Get historical open interest data.
    
    Args:
        symbol: Trading symbol
        period: Kline interval (5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d)
        limit: Number of data points
    
    Returns:
        DataFrame with timestamp, sumOpenInterest, sumOpenInterestValue
    """
    try:
        data = client.futures_open_interest_hist(symbol=symbol, period=period, limit=limit)
        if data:
            df = pd.DataFrame(data)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df["sumOpenInterest"] = pd.to_numeric(df["sumOpenInterest"], errors="coerce")
            df["sumOpenInterestValue"] = pd.to_numeric(df["sumOpenInterestValue"], errors="coerce")
            return df
    except Exception as e:
        pass
    
    return pd.DataFrame(columns=["timestamp", "sumOpenInterest", "sumOpenInterestValue"])


def get_mark_price(symbol: str) -> Dict[str, Any]:
    """
    Get mark price and index price for a futures symbol.
    
    Mark price is used for liquidation calculations and is less
    susceptible to manipulation than last traded price.
    
    Returns:
        Dict with markPrice, indexPrice, and estimatedSettlePrice
    """
    try:
        data = client.futures_mark_price(symbol=symbol)
        if data:
            return {
                "symbol": symbol,
                "markPrice": float(data.get("markPrice", 0)),
                "indexPrice": float(data.get("indexPrice", 0)),
                "estimatedSettlePrice": float(data.get("estimatedSettlePrice", 0)),
                "lastFundingRate": float(data.get("lastFundingRate", 0)),
                "nextFundingTime": int(data.get("nextFundingTime", 0)),
                "time": int(data.get("time", 0)),
            }
    except Exception as e:
        pass
    
    return {
        "symbol": symbol,
        "markPrice": 0.0,
        "indexPrice": 0.0,
        "estimatedSettlePrice": 0.0,
        "lastFundingRate": 0.0,
        "nextFundingTime": 0,
        "time": 0,
    }


def get_order_book_depth(symbol: str, limit: int = 100) -> Dict[str, Any]:
    """
    Get order book depth for analysis.
    
    Order book analysis reveals:
    - Bid/Ask imbalance (buying vs selling pressure)
    - Large walls (potential support/resistance)
    - Depth at key levels
    
    Args:
        symbol: Trading symbol
        limit: Depth limit (5, 10, 20, 50, 100, 500, 1000)
    
    Returns:
        Dict with bids, asks, imbalance, and summary statistics
    """
    try:
        data = client.futures_order_book(symbol=symbol, limit=limit)
        if data:
            bids = [(float(price), float(qty)) for price, qty in data.get("bids", [])]
            asks = [(float(price), float(qty)) for price, qty in data.get("asks", [])]
            
            total_bid_qty = sum(qty for _, qty in bids)
            total_ask_qty = sum(qty for _, qty in asks)
            
            # Bid/Ask imbalance ratio
            total_qty = total_bid_qty + total_ask_qty
            if total_qty > 0:
                bid_ratio = total_bid_qty / total_qty
                ask_ratio = total_ask_qty / total_qty
                imbalance = (total_bid_qty - total_ask_qty) / total_qty
            else:
                bid_ratio = 0.5
                ask_ratio = 0.5
                imbalance = 0.0
            
            # Find largest walls
            largest_bid = max(bids, key=lambda x: x[1]) if bids else (0, 0)
            largest_ask = min(asks, key=lambda x: x[1]) if asks else (0, 0)
            
            # Calculate weighted average prices
            if total_bid_qty > 0:
                wavg_bid = sum(p * q for p, q in bids) / total_bid_qty
            else:
                wavg_bid = 0
            
            if total_ask_qty > 0:
                wavg_ask = sum(p * q for p, q in asks) / total_ask_qty
            else:
                wavg_ask = 0
            
            return {
                "symbol": symbol,
                "bidCount": len(bids),
                "askCount": len(asks),
                "totalBidQty": total_bid_qty,
                "totalAskQty": total_ask_qty,
                "bidRatio": bid_ratio,
                "askRatio": ask_ratio,
                "imbalance": imbalance,
                "pressure": "BUYING" if imbalance > 0.1 else "SELLING" if imbalance < -0.1 else "NEUTRAL",
                "largestBidPrice": largest_bid[0],
                "largestBidQty": largest_bid[1],
                "largestAskPrice": largest_ask[0],
                "largestAskQty": largest_ask[1],
                "wavgBid": wavg_bid,
                "wavgAsk": wavg_ask,
                "spread": asks[0][0] - bids[0][0] if bids and asks else 0,
                "spreadPct": (asks[0][0] - bids[0][0]) / bids[0][0] * 100 if bids and asks and bids[0][0] > 0 else 0,
            }
    except Exception as e:
        pass
    
    return {
        "symbol": symbol,
        "bidCount": 0,
        "askCount": 0,
        "totalBidQty": 0.0,
        "totalAskQty": 0.0,
        "bidRatio": 0.5,
        "askRatio": 0.5,
        "imbalance": 0.0,
        "pressure": "UNKNOWN",
        "largestBidPrice": 0.0,
        "largestBidQty": 0.0,
        "largestAskPrice": 0.0,
        "largestAskQty": 0.0,
        "wavgBid": 0.0,
        "wavgAsk": 0.0,
        "spread": 0.0,
        "spreadPct": 0.0,
    }


def get_recent_trades(symbol: str, limit: int = 500) -> pd.DataFrame:
    """
    Get recent trades for analysis.
    
    Trade analysis reveals:
    - Trade direction flow
    - Average trade size
    - Large trades (whale activity)
    
    Args:
        symbol: Trading symbol
        limit: Number of trades (max 1000)
    
    Returns:
        DataFrame with trade details
    """
    try:
        data = client.futures_recent_trades(symbol=symbol, limit=limit)
        if data:
            df = pd.DataFrame(data)
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
            df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
            df["quoteQty"] = pd.to_numeric(df["quoteQty"], errors="coerce")
            df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
            df["isBuyerMaker"] = df["isBuyerMaker"].astype(bool)
            return df
    except Exception as e:
        pass
    
    return pd.DataFrame(columns=["id", "price", "qty", "quoteQty", "time", "isBuyerMaker"])


def get_long_short_ratio(symbol: str, period: str = "5m", limit: int = 30) -> Dict[str, Any]:
    """
    Get long/short ratio for top traders.
    
    This shows the positioning of top traders:
    - Ratio > 1: More longs than shorts
    - Ratio < 1: More shorts than longs
    
    Args:
        symbol: Trading symbol
        period: Time period (5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d)
        limit: Number of data points
    
    Returns:
        Dict with current ratio and historical data
    """
    try:
        data = client.futures_top_longshort_position_ratio(symbol=symbol, period=period, limit=limit)
        if data:
            df = pd.DataFrame(data)
            df["longShortRatio"] = pd.to_numeric(df["longShortRatio"], errors="coerce")
            df["longAccount"] = pd.to_numeric(df["longAccount"], errors="coerce")
            df["shortAccount"] = pd.to_numeric(df["shortAccount"], errors="coerce")
            
            current = df.iloc[-1] if not df.empty else None
            
            if current is not None:
                ratio = float(current["longShortRatio"])
                return {
                    "symbol": symbol,
                    "longShortRatio": ratio,
                    "longPct": float(current["longAccount"]) * 100,
                    "shortPct": float(current["shortAccount"]) * 100,
                    "sentiment": "LONG_HEAVY" if ratio > 1.5 else "SHORT_HEAVY" if ratio < 0.67 else "BALANCED",
                    "historicalData": df.to_dict("records"),
                }
    except Exception as e:
        pass
    
    return {
        "symbol": symbol,
        "longShortRatio": 1.0,
        "longPct": 50.0,
        "shortPct": 50.0,
        "sentiment": "UNKNOWN",
        "historicalData": [],
    }


def get_taker_buy_sell_ratio(symbol: str, period: str = "5m", limit: int = 30) -> Dict[str, Any]:
    """
    Get taker buy/sell volume ratio.
    
    This shows whether takers (market orders) are buying or selling:
    - Ratio > 1: More taker buys (bullish pressure)
    - Ratio < 1: More taker sells (bearish pressure)
    
    Args:
        symbol: Trading symbol
        period: Time period (5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d)
        limit: Number of data points
    
    Returns:
        Dict with current ratio and analysis
    """
    try:
        data = client.futures_taker_long_short_ratio(symbol=symbol, period=period, limit=limit)
        if data:
            df = pd.DataFrame(data)
            df["buySellRatio"] = pd.to_numeric(df["buySellRatio"], errors="coerce")
            df["buyVol"] = pd.to_numeric(df["buyVol"], errors="coerce")
            df["sellVol"] = pd.to_numeric(df["sellVol"], errors="coerce")
            
            current = df.iloc[-1] if not df.empty else None
            
            if current is not None:
                ratio = float(current["buySellRatio"])
                return {
                    "symbol": symbol,
                    "buySellRatio": ratio,
                    "buyVol": float(current["buyVol"]),
                    "sellVol": float(current["sellVol"]),
                    "pressure": "BUYING" if ratio > 1.1 else "SELLING" if ratio < 0.9 else "BALANCED",
                    "historicalData": df.to_dict("records"),
                }
    except Exception as e:
        pass
    
    return {
        "symbol": symbol,
        "buySellRatio": 1.0,
        "buyVol": 0.0,
        "sellVol": 0.0,
        "pressure": "UNKNOWN",
        "historicalData": [],
    }


def get_comprehensive_market_data(symbol: str) -> Dict[str, Any]:
    """
    Get comprehensive market data for deep analysis.
    
    This aggregates multiple data sources for professional analysis:
    - Current prices (mark, index)
    - Funding rate
    - Open interest
    - Order book depth
    - Long/short ratio
    - Taker buy/sell ratio
    
    Args:
        symbol: Trading symbol
    
    Returns:
        Comprehensive dict with all market data
    """
    mark_data = get_mark_price(symbol)
    funding_data = get_funding_rate(symbol)
    oi_data = get_open_interest(symbol)
    order_book = get_order_book_depth(symbol, limit=100)
    ls_ratio = get_long_short_ratio(symbol, period="5m", limit=1)
    taker_ratio = get_taker_buy_sell_ratio(symbol, period="5m", limit=1)
    
    # Aggregate sentiment
    bullish_signals = 0
    bearish_signals = 0
    
    # Funding sentiment
    if funding_data["sentiment"] == "BEARISH_CROWDED":
        bullish_signals += 1  # Contrarian
    elif funding_data["sentiment"] == "BULLISH_CROWDED":
        bearish_signals += 1  # Contrarian
    
    # Order book pressure
    if order_book["pressure"] == "BUYING":
        bullish_signals += 1
    elif order_book["pressure"] == "SELLING":
        bearish_signals += 1
    
    # Taker pressure
    if taker_ratio["pressure"] == "BUYING":
        bullish_signals += 1
    elif taker_ratio["pressure"] == "SELLING":
        bearish_signals += 1
    
    if bullish_signals > bearish_signals:
        overall_sentiment = "BULLISH"
    elif bearish_signals > bullish_signals:
        overall_sentiment = "BEARISH"
    else:
        overall_sentiment = "NEUTRAL"
    
    return {
        "symbol": symbol,
        "markPrice": mark_data,
        "funding": funding_data,
        "openInterest": oi_data,
        "orderBook": order_book,
        "longShortRatio": ls_ratio,
        "takerRatio": taker_ratio,
        "overallSentiment": overall_sentiment,
        "bullishSignals": bullish_signals,
        "bearishSignals": bearish_signals,
    }


def place_trade_setup(symbol: str, 
                      side: str, 
                      entry_price: float, 
                      quantity: float, 
                      tp_price: float, 
                      sl_price: float,
                      leverage: int = 1) -> Dict[str, Any]:
    """
    Executes a complete trade setup on Binance Futures.
    
    Sequence:
    1. Set Leverage
    2. Place LIMIT Entry Order
    3. (Ideally) Place OCO or SL/TP orders attached.
       Binance Futures API allows separating orders.
       We will place standalone STOP_MARKET and TAKE_PROFIT_MARKET orders
       linked to the position (reduceOnly=True).
       
    Args:
        symbol: Trading pair (e.g. BTCUSDT)
        side: "LONG" or "SHORT"
        entry_price: Limit price for entry
        quantity: Contract quantity
        tp_price: Take profit price
        sl_price: Stop loss price
        leverage: Leverage to set
        
    Returns:
        Dict with status and order IDs
    """
    results = {"status": "error", "logs": []}
    
    try:
        # A. Set Leverage
        try:
            client.futures_change_leverage(symbol=symbol, leverage=leverage)
            results["logs"].append(f"Leverage set to {leverage}x")
        except Exception as e:
            results["logs"].append(f"Warning: Failed to set leverage: {e}")

        # B. Determine Sides
        if side.upper() == "LONG":
            entry_side = "BUY"
            exit_side = "SELL"
        else:
            entry_side = "SELL"
            exit_side = "BUY"
            
        # C. Place Entry Order (Limit)
        # Note: quantity must be positive
        entry_order = client.futures_create_order(
            symbol=symbol,
            side=entry_side,
            type="LIMIT",
            timeInForce="GTC",
            quantity=abs(quantity),
            price=entry_price
        )
        results["entry_id"] = entry_order.get("orderId")
        results["logs"].append(f"Placed {entry_side} LIMIT at {entry_price}")
        
        # D. Place Protection Orders (Strategy)
        # Since the position isn't open yet (it's a limit order), 
        # placing simple STOP/TP orders works as triggers.
        # Ideally, we use the batch orders endpoint if possible, but sequential is safer for now.
        
        # Stop Loss (STOP_MARKET)
        # Reduce Only = True ensures we don't open new opposite positions
        sl_order = client.futures_create_order(
            symbol=symbol,
            side=exit_side,
            type="STOP_MARKET",
            stopPrice=sl_price,
            closePosition="true" # This implies quantities match the position, safer than specifying qty
        )
        results["sl_id"] = sl_order.get("orderId")
        results["logs"].append(f"Placed SL {exit_side} at {sl_price}")
        
        # Take Profit (TAKE_PROFIT_MARKET)
        tp_order = client.futures_create_order(
            symbol=symbol,
            side=exit_side,
            type="TAKE_PROFIT_MARKET",
            stopPrice=tp_price,
            closePosition="true"
        )
        results["tp_id"] = tp_order.get("orderId")
        results["logs"].append(f"Placed TP {exit_side} at {tp_price}")
        
        results["status"] = "success"
        
    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)
        results["logs"].append(f"CRITICAL ERROR: {e}")
        
    return results
