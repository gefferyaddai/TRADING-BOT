"""
data.py — Market data fetching and morning stock screener.

Public API:
    get_minute_bars(symbol, n_bars)  → pd.DataFrame  (OHLCV)
    get_daily_bars(symbol, n_days)   → pd.DataFrame  (OHLCV)
    get_spy_bars(n_days)             → pd.DataFrame  (OHLCV for SPY regime check)
    run_morning_screen()             → List[str]      (top ticker symbols)
"""

import logging
import time
from datetime import datetime, timedelta
from typing import List

import pandas as pd
import numpy as np

from config import CFG

log = logging.getLogger(__name__)

# ── S&P 500 universe (top 100 liquid names used for screening) ─────────────
SP500_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA", "BRK.B",
    "JPM", "LLY", "V", "UNH", "XOM", "MA", "AVGO", "JNJ", "PG", "HD", "MRK",
    "COST", "CVX", "ABBV", "WMT", "BAC", "PEP", "KO", "ADBE", "CRM", "ACN",
    "MCD", "TMO", "CSCO", "ABT", "AMD", "NFLX", "LIN", "DHR", "TXN", "CMCSA",
    "NKE", "WFC", "PM", "NEE", "RTX", "ORCL", "INTC", "QCOM", "UPS", "HON",
    "IBM", "GS", "MS", "SPGI", "CAT", "AMGN", "ELV", "DE", "AMAT", "INTU",
    "ISRG", "BKNG", "AXP", "MDLZ", "ADI", "VRTX", "BLK", "SYK", "GILD", "MMC",
    "PLD", "REGN", "MU", "LRCX", "ZTS", "NOW", "TJX", "SBUX", "C", "BSX",
    "PGR", "KLAC", "SO", "DUK", "CL", "MO", "SLB", "F", "GM", "GE",
    "UBER", "ABNB", "SNOW", "PLTR", "COIN", "SQ", "SHOP", "ROKU", "ZM", "RBLX",
]

# ── Sector map — used for diversification cap (max 3 per sector) ───────────
SECTOR_MAP = {
    # Technology
    "AAPL": "Tech", "MSFT": "Tech", "NVDA": "Tech", "AVGO": "Tech",
    "ADBE": "Tech", "CRM": "Tech", "AMD": "Tech", "CSCO": "Tech",
    "TXN": "Tech", "INTC": "Tech", "QCOM": "Tech", "ORCL": "Tech",
    "IBM": "Tech", "AMAT": "Tech", "INTU": "Tech", "ADI": "Tech",
    "MU": "Tech", "LRCX": "Tech", "NOW": "Tech", "KLAC": "Tech",
    "SNOW": "Tech", "PLTR": "Tech", "SHOP": "Tech", "ROKU": "Tech", "ZM": "Tech",
    # Communication
    "META": "Comm", "GOOGL": "Comm", "GOOG": "Comm", "NFLX": "Comm",
    "CMCSA": "Comm", "RBLX": "Comm",
    # Consumer Discretionary
    "AMZN": "ConsDisc", "TSLA": "ConsDisc", "HD": "ConsDisc",
    "MCD": "ConsDisc", "NKE": "ConsDisc", "BKNG": "ConsDisc",
    "TJX": "ConsDisc", "SBUX": "ConsDisc", "UBER": "ConsDisc", "ABNB": "ConsDisc",
    # Consumer Staples
    "PG": "ConsStap", "PEP": "ConsStap", "KO": "ConsStap",
    "COST": "ConsStap", "WMT": "ConsStap", "MDLZ": "ConsStap",
    "CL": "ConsStap", "MO": "ConsStap", "PM": "ConsStap",
    # Financials
    "JPM": "Fin", "V": "Fin", "MA": "Fin", "BAC": "Fin",
    "WFC": "Fin", "GS": "Fin", "MS": "Fin", "SPGI": "Fin",
    "BLK": "Fin", "MMC": "Fin", "AXP": "Fin", "PGR": "Fin",
    "C": "Fin", "COIN": "Fin", "SQ": "Fin",
    # Healthcare
    "UNH": "Health", "JNJ": "Health", "LLY": "Health", "MRK": "Health",
    "ABBV": "Health", "TMO": "Health", "ABT": "Health", "DHR": "Health",
    "AMGN": "Health", "ELV": "Health", "ISRG": "Health", "VRTX": "Health",
    "GILD": "Health", "SYK": "Health", "BSX": "Health", "REGN": "Health",
    "ZTS": "Health",
    # Industrials
    "HON": "Indust", "UPS": "Indust", "RTX": "Indust",
    "CAT": "Indust", "DE": "Indust", "GE": "Indust",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "SLB": "Energy",
    # Materials
    "LIN": "Matls",
    # Utilities
    "NEE": "Util", "SO": "Util", "DUK": "Util",
    # Real Estate
    "PLD": "RE",
    # Auto
    "F": "Auto", "GM": "Auto",
}


# ══════════════════════════════════════════════════════════════════════════════
# Alpaca client (lazy-initialised)
# ══════════════════════════════════════════════════════════════════════════════

_api = None


def _get_api():
    global _api
    if _api is None:
        try:
            import alpaca_trade_api as tradeapi
        except ImportError:
            raise ImportError("Run:  pip install alpaca-trade-api")
        base_url = (
            "https://paper-api.alpaca.markets"
            if CFG.PAPER else
            "https://api.alpaca.markets"
        )
        _api = tradeapi.REST(CFG.API_KEY, CFG.API_SECRET, base_url, api_version="v2")
        log.info(f"Alpaca API initialised — mode: {'PAPER' if CFG.PAPER else 'LIVE'}")
    return _api


# ══════════════════════════════════════════════════════════════════════════════
# Bar fetching
# ══════════════════════════════════════════════════════════════════════════════

def get_minute_bars(symbol: str, n_bars: int = CFG.BARS_TO_FETCH) -> pd.DataFrame:
    """Fetch the most recent n_bars minute bars for symbol."""
    api   = _get_api()
    end   = datetime.utcnow()
    start = end - timedelta(minutes=n_bars * 3)
    try:
        bars = api.get_bars(
            symbol, CFG.BAR_TIMEFRAME,
            start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            adjustment="raw",
        ).df
        if bars.empty:
            return pd.DataFrame()
        bars = bars[["open", "high", "low", "close", "volume"]].copy()
        bars.index = pd.to_datetime(bars.index).tz_convert("UTC").tz_localize(None)
        return bars.tail(n_bars)
    except Exception as e:
        log.error(f"get_minute_bars({symbol}): {e}")
        return pd.DataFrame()


def get_daily_bars(symbol: str, n_days: int = 30) -> pd.DataFrame:
    """Fetch daily bars — used by screener and backtester."""
    api   = _get_api()
    end   = datetime.utcnow()
    start = end - timedelta(days=n_days * 2)
    try:
        bars = api.get_bars(
            symbol, "1Day",
            start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            adjustment="raw",
        ).df
        if bars.empty:
            return pd.DataFrame()
        bars = bars[["open", "high", "low", "close", "volume"]].copy()
        bars.index = pd.to_datetime(bars.index).tz_convert("UTC").tz_localize(None)
        return bars.tail(n_days)
    except Exception as e:
        log.error(f"get_daily_bars({symbol}): {e}")
        return pd.DataFrame()


def get_spy_bars(n_days: int = 30) -> pd.DataFrame:
    """
    Fetch daily SPY bars for the market regime filter.
    SPY ADX < REGIME_ADX_THRESHOLD → choppy market → reduce position sizes.
    """
    return get_daily_bars("SPY", n_days=n_days)


# ══════════════════════════════════════════════════════════════════════════════
# Morning screener
# ══════════════════════════════════════════════════════════════════════════════

def _momentum_score(close: pd.Series, lookback: int) -> float:
    if len(close) < lookback + 1:
        return float("-inf")
    return (close.iloc[-1] / close.iloc[-lookback] - 1) * 100


def _avg_volume(volume: pd.Series) -> float:
    return volume.mean()


def _atr_pct(high: pd.Series, low: pd.Series, close: pd.Series) -> float:
    if len(close) < 2:
        return float("inf")
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr_val = tr.ewm(span=CFG.ATR_PERIOD, adjust=False).mean().iloc[-1]
    return atr_val / close.iloc[-1]


def run_morning_screen(universe: List[str] = None) -> List[str]:
    """
    Screen the universe each morning and return top N tickers by momentum.

    Filters  : price, avg volume, ATR%
    Ranks by : 20-day rate of change (descending)
    New      : sector diversification cap — max CFG.MAX_PER_SECTOR per sector
    """
    if universe is None:
        universe = SP500_UNIVERSE

    log.info(f"Running morning screen over {len(universe)} symbols ...")
    scores = []

    for symbol in universe:
        try:
            df = get_daily_bars(symbol, n_days=max(CFG.MOMENTUM_LOOKBACK + 5, 30))
            if df.empty or len(df) < CFG.MOMENTUM_LOOKBACK + 1:
                continue

            price   = df["close"].iloc[-1]
            avg_vol = _avg_volume(df["volume"])
            atr_p   = _atr_pct(df["high"], df["low"], df["close"])
            mom     = _momentum_score(df["close"], CFG.MOMENTUM_LOOKBACK)

            if price   < CFG.MIN_PRICE:      continue
            if avg_vol < CFG.MIN_AVG_VOLUME: continue
            if atr_p   > CFG.MAX_ATR_PCT:    continue
            if mom == float("-inf"):         continue

            sector = SECTOR_MAP.get(symbol, "Other")
            scores.append({
                "symbol": symbol, "momentum": mom,
                "price": price, "avg_vol": avg_vol, "sector": sector,
            })
            time.sleep(0.15)

        except Exception as e:
            log.warning(f"Screen error for {symbol}: {e}")

    if not scores:
        log.warning("Screener returned no results — check credentials / market hours")
        return []

    # Rank by momentum descending
    ranked = sorted(scores, key=lambda x: x["momentum"], reverse=True)

    # Apply sector diversification cap
    selected    = []
    sector_counts: dict = {}

    for r in ranked:
        if len(selected) >= CFG.WATCHLIST_SIZE:
            break
        sector = r["sector"]
        count  = sector_counts.get(sector, 0)
        if count >= CFG.MAX_PER_SECTOR:
            log.debug(
                f"  {r['symbol']} skipped — sector {sector} "
                f"already has {count} stocks"
            )
            continue
        selected.append(r)
        sector_counts[sector] = count + 1

    top = [r["symbol"] for r in selected]

    log.info(f"Today's watchlist ({len(top)}): {top}")
    for r in selected:
        log.info(
            f"  {r['symbol']:<6}  sector={r['sector']:<10} "
            f"mom={r['momentum']:+.1f}%  "
            f"price=${r['price']:.2f}  vol={r['avg_vol']:,.0f}"
        )

    return top