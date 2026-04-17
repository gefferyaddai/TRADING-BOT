"""
config.py — Single source of truth for all bot settings.
Edit this file to tune the bot. Do NOT hardcode values elsewhere.
"""

import os
from dataclasses import dataclass
from typing import List


@dataclass
class Config:
    # ── Alpaca credentials ─────────────────────────────────────────────────────
    # Set these as environment variables, or replace the defaults for testing.
    API_KEY:    str = os.getenv("ALPACA_API_KEY",    "YOUR_API_KEY")
    API_SECRET: str = os.getenv("ALPACA_API_SECRET", "YOUR_API_SECRET")

    # True  → paper trading (safe, no real money)
    # False → live trading  ⚠️  real money at risk
    PAPER: bool = True

    # ── Capital ────────────────────────────────────────────────────────────────
    STARTING_CAPITAL: float = 10_000.0   # virtual capital for paper trading

    # ── Risk management ────────────────────────────────────────────────────────
    RISK_PER_TRADE:     float = 0.02   # 2% of portfolio risked per trade
    ATR_STOP_MULT:      float = 2.0    # stop-loss = entry - (ATR × this)
    MAX_OPEN_POSITIONS: int   = 10     # max simultaneous holdings
    MAX_POSITION_PCT:   float = 0.15   # single position ≤ 15% of portfolio

    # Drawdown guard — pause trading if intraday loss exceeds this
    DAILY_DRAWDOWN_LIMIT: float = 0.05   # 5%

    # ── Stock screener ─────────────────────────────────────────────────────────
    WATCHLIST_SIZE:    int   = 10        # top N stocks selected each morning
    MOMENTUM_LOOKBACK: int   = 20        # days for rate-of-change momentum score
    MIN_AVG_VOLUME:    int   = 500_000   # minimum avg daily volume filter
    MIN_PRICE:         float = 5.0       # ignore penny stocks
    MAX_ATR_PCT:       float = 0.05      # ignore stocks with ATR > 5% of price

    # Sector diversification cap — max stocks from any single sector
    MAX_PER_SECTOR:    int   = 3

    # ── Strategy indicators ────────────────────────────────────────────────────
    EMA_FAST:      int   = 9
    EMA_SLOW:      int   = 21
    ADX_PERIOD:    int   = 14
    ADX_THRESHOLD: float = 25.0   # minimum ADX to confirm trend strength
    ATR_PERIOD:    int   = 14

    # ── Market regime filter (SPY ADX) ────────────────────────────────────────
    # If SPY's ADX falls below this, the market is choppy — reduce size
    REGIME_ADX_THRESHOLD:  float = 20.0
    REGIME_SIZE_MULTIPLIER: float = 0.25  # 25% of normal size in choppy market

    # ── Sentiment engine ───────────────────────────────────────────────────────
    FINNHUB_API_KEY: str = os.getenv("FINNHUB_API_KEY", "YOUR_FINNHUB_KEY")

    # Hours of news to look back when scoring sentiment
    SENTIMENT_LOOKBACK_HOURS: int = 24

    # Minimum number of headlines that must agree on direction
    SENTIMENT_CONSENSUS_MIN: int = 2

    # Gate: sentiment score below this blocks any new buy entry
    SENTIMENT_GATE_THRESHOLD: float = -0.3

    # Sizing bands: score → conviction multiplier applied to position size
    # Format: (min_score, max_score, multiplier)
    # Scores below gate threshold are blocked entirely (no entry)
    SENTIMENT_SIZE_BANDS: tuple = (
        (-0.3,  0.3,  1.0),   # neutral  → normal size
        ( 0.3,  0.6,  1.0),   # mild positive → normal size
        ( 0.6,  0.8,  1.5),   # strong positive → 1.5× size
        ( 0.8,  1.01, 2.0),   # very strong → 2× size (still capped at MAX_POSITION_PCT)
    )

    # ── Backtest settings ──────────────────────────────────────────────────────
    BACKTEST_YEARS:          int = 3   # total historical window
    BACKTEST_INSAMPLE_YEARS: int = 2   # must be < BACKTEST_YEARS; remainder is OOS

    def __post_init__(self):
        if self.BACKTEST_INSAMPLE_YEARS >= self.BACKTEST_YEARS:
            raise ValueError(
                f"BACKTEST_INSAMPLE_YEARS ({self.BACKTEST_INSAMPLE_YEARS}) "
                f"must be less than BACKTEST_YEARS ({self.BACKTEST_YEARS})"
            )

    # ── Execution ──────────────────────────────────────────────────────────────
    BAR_TIMEFRAME:     str = "1Min"  # minute bars for live trading
    BARS_TO_FETCH:     int = 100     # bars loaded per signal check
    LOOP_SLEEP_SEC:    int = 60      # seconds between each signal scan
    PRE_MARKET_HOUR:   int = 9       # hour (ET) to run morning screener
    PRE_MARKET_MIN:    int = 15      # minute (ET) to run morning screener
    MARKET_CLOSE_HOUR: int = 15      # stop opening new trades after 3 PM ET
    MARKET_CLOSE_MIN:  int = 45

    # ── Logging ────────────────────────────────────────────────────────────────
    LOG_FILE:  str = "trading_bot.log"
    LOG_LEVEL: str = "INFO"


# Singleton — import this everywhere
CFG = Config()