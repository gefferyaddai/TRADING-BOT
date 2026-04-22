"""
strategy.py — Technical indicators, regime detection, and signal generation.

Public API:
    add_indicators(df)                        → pd.DataFrame
    get_regime(spy_df)                        → Regime (TRENDING | CHOPPY)
    get_signal(df, sentiment_score)           → Signal (BUY | SELL | HOLD)
    get_conviction_multiplier(sentiment, regime) → float
    indicator_snapshot(df)                    → dict
"""

import logging
from enum import Enum

import pandas as pd
import numpy as np

from config import CFG

log = logging.getLogger(__name__)


class Signal(Enum):
    BUY  = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class Regime(Enum):
    TRENDING = "TRENDING"   # ADX >= threshold — full size
    CHOPPY   = "CHOPPY"     # ADX <  threshold — reduced size


# ══════════════════════════════════════════════════════════════════════════════
# Indicators
# ══════════════════════════════════════════════════════════════════════════════

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _atr(high: pd.Series, low: pd.Series,
         close: pd.Series, period: int) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _adx(high: pd.Series, low: pd.Series,
         close: pd.Series, period: int) -> pd.Series:
    up   = high.diff()
    down = -low.diff()
    plus_dm  = np.where((up > down) & (up > 0),   up,   0.0)
    minus_dm = np.where((down > up) & (down > 0), down,  0.0)
    tr_smooth   = _atr(high, low, close, period)
    plus_di  = (100 * pd.Series(plus_dm,  index=close.index)
                       .ewm(span=period, adjust=False).mean() / tr_smooth)
    minus_di = (100 * pd.Series(minus_dm, index=close.index)
                       .ewm(span=period, adjust=False).mean() / tr_smooth)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9))
    return dx.ewm(span=period, adjust=False).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all indicators on an OHLCV dataframe.

    Adds: ema_fast, ema_slow, adx, atr, ema_diff, prev_ema_diff
    """
    if len(df) < CFG.EMA_SLOW + CFG.ADX_PERIOD:
        return df

    df = df.copy()
    df["ema_fast"]      = _ema(df["close"], CFG.EMA_FAST)
    df["ema_slow"]      = _ema(df["close"], CFG.EMA_SLOW)
    df["adx"]           = _adx(df["high"], df["low"], df["close"], CFG.ADX_PERIOD)
    df["atr"]           = _atr(df["high"], df["low"], df["close"], CFG.ATR_PERIOD)
    df["ema_diff"]      = df["ema_fast"] - df["ema_slow"]
    df["prev_ema_diff"] = df["ema_diff"].shift(1)
    return df.dropna()


# ══════════════════════════════════════════════════════════════════════════════
# Market regime filter
# ══════════════════════════════════════════════════════════════════════════════

def get_regime(spy_df: pd.DataFrame) -> Regime:
    """
    Assess the broad market regime using SPY's ADX.

    TRENDING : SPY ADX >= REGIME_ADX_THRESHOLD → trade at full size
    CHOPPY   : SPY ADX <  REGIME_ADX_THRESHOLD → reduce to 25% size

    This protects against running a momentum strategy in a ranging,
    directionless market where EMA crossovers generate false signals.
    """
    if spy_df is None or spy_df.empty:
        log.warning("SPY data unavailable — defaulting to TRENDING regime")
        return Regime.TRENDING

    spy_ind = add_indicators(spy_df)
    if spy_ind.empty or "adx" not in spy_ind.columns:
        log.warning("SPY indicators unavailable — defaulting to TRENDING regime")
        return Regime.TRENDING

    spy_adx = float(spy_ind["adx"].iloc[-1])

    if spy_adx >= CFG.REGIME_ADX_THRESHOLD:
        log.info(f"Market regime: TRENDING  (SPY ADX={spy_adx:.1f})")
        return Regime.TRENDING
    else:
        log.info(
            f"Market regime: CHOPPY  (SPY ADX={spy_adx:.1f} "
            f"< threshold {CFG.REGIME_ADX_THRESHOLD}) — positions at 25%"
        )
        return Regime.CHOPPY


# ══════════════════════════════════════════════════════════════════════════════
# Signal generation
# ══════════════════════════════════════════════════════════════════════════════

def get_signal(df: pd.DataFrame, sentiment_score: float = 0.0,
               _enriched: bool = False) -> Signal:
    """
    Evaluate the most recent bar and return a trading signal.

    BUY  — all must be true:
        1. EMA fast crosses above EMA slow
        2. ADX >= CFG.ADX_THRESHOLD  (trend confirmed)
        3. sentiment_score >= CFG.SENTIMENT_GATE_THRESHOLD  (news not negative)

    SELL — any one is enough:
        1. EMA fast crosses below EMA slow
        2. ADX drops below 80% of threshold  (trend fading)

    Sentiment never blocks a SELL — technical exits always execute.

    Pass _enriched=True to skip recomputing indicators when the caller has
    already called add_indicators (e.g. after indicator_snapshot).
    """
    if df.empty or len(df) < 2:
        return Signal.HOLD

    df = df if _enriched else add_indicators(df)
    if df.empty:
        return Signal.HOLD

    last = df.iloc[-1]
    required = ["ema_diff", "prev_ema_diff", "adx", "atr"]
    if any(pd.isna(last[c]) for c in required):
        return Signal.HOLD

    crossed_up   = last["prev_ema_diff"] <= 0 and last["ema_diff"] > 0
    crossed_down = last["prev_ema_diff"] >= 0 and last["ema_diff"] < 0
    trend_strong = last["adx"] >= CFG.ADX_THRESHOLD
    trend_faded  = last["adx"] < CFG.ADX_THRESHOLD * 0.80

    # ── SELL: purely technical, sentiment ignored ──────────────────────────
    if crossed_down or trend_faded:
        log.debug(
            f"SELL signal | crossed_down={crossed_down} "
            f"trend_faded={trend_faded}  ADX={last['adx']:.1f}"
        )
        return Signal.SELL

    # ── BUY: requires technical + sentiment gate ───────────────────────────
    if crossed_up and trend_strong:
        if sentiment_score < CFG.SENTIMENT_GATE_THRESHOLD:
            log.info(
                f"BUY signal blocked by sentiment gate  "
                f"score={sentiment_score:.2f} < {CFG.SENTIMENT_GATE_THRESHOLD}"
            )
            return Signal.HOLD
        log.debug(
            f"BUY signal  | EMA diff={last['ema_diff']:.4f} "
            f"ADX={last['adx']:.1f}  sentiment={sentiment_score:.2f}"
        )
        return Signal.BUY

    return Signal.HOLD


# ══════════════════════════════════════════════════════════════════════════════
# Conviction multiplier  (sentiment × regime combined)
# ══════════════════════════════════════════════════════════════════════════════

def get_conviction_multiplier(sentiment_score: float, regime: Regime) -> float:
    """
    Combine sentiment sizing and regime scaling into a single multiplier
    passed to risk.position_size().

    Sentiment bands (from config):
        score -0.3 to 0.6  → 1.0×
        score  0.6 to 0.8  → 1.5×
        score  0.8 to 1.0  → 2.0×

    Regime overlay:
        TRENDING → multiplier unchanged
        CHOPPY   → multiplier × 0.25  (quarter size)

    Final output is always capped so position stays within MAX_POSITION_PCT.
    """
    # Determine sentiment multiplier from bands
    sentiment_mult = 1.0
    for lo, hi, mult in CFG.SENTIMENT_SIZE_BANDS:
        if lo <= sentiment_score < hi:
            sentiment_mult = mult
            break

    # Apply regime overlay
    regime_mult = 1.0 if regime == Regime.TRENDING else CFG.REGIME_SIZE_MULTIPLIER
    combined    = sentiment_mult * regime_mult

    log.debug(
        f"conviction_multiplier={combined:.2f}  "
        f"(sentiment={sentiment_mult:.1f} × regime={regime_mult:.2f})"
    )
    return combined


# ══════════════════════════════════════════════════════════════════════════════
# Indicator snapshot for logging
# ══════════════════════════════════════════════════════════════════════════════

def indicator_snapshot(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Return (snapshot_dict, enriched_df) so callers can reuse the enriched df."""
    enriched = add_indicators(df)
    if enriched.empty:
        return {}, enriched
    last = enriched.iloc[-1]
    snap = {
        "close":    round(float(last["close"]),    4),
        "ema_fast": round(float(last["ema_fast"]), 4),
        "ema_slow": round(float(last["ema_slow"]), 4),
        "adx":      round(float(last["adx"]),      2),
        "atr":      round(float(last["atr"]),       4),
    }
    return snap, enriched