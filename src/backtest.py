"""
backtest.py — Vectorised backtester with in-sample / out-of-sample split.

Public API:
    run_backtest(symbol, variant, start_date, end_date) → dict
    run_insample_oos(symbol, variant)                   → dict
    print_results(results)

Variants:
    "baseline"   — EMA(9/21) + ADX >= 25, no sentiment
    "sentiment"  — baseline + sentiment gate + conviction sizing
    "low_adx"    — EMA(9/21) + ADX >= 15  + sentiment

NOTE: Backtesting uses daily bars (not minute bars) for historical coverage.
Sentiment is simulated in backtest using a random seed for reproducibility
(real sentiment data is not stored historically). The backtest demonstrates
the structural effect of the sentiment gate and sizing logic.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from config import CFG
from data import get_daily_bars
from strategy import _ema, _atr, _adx

log = logging.getLogger(__name__)

COMMISSION_PER_SHARE = 0.005   # $0.005/share round-trip half
SLIPPAGE_BPS         = 5.0     # 5 bps per side


def _prepare(df: pd.DataFrame, adx_threshold: float) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"]   = _ema(df["close"], CFG.EMA_FAST)
    df["ema_slow"]   = _ema(df["close"], CFG.EMA_SLOW)
    df["adx"]        = _adx(df["high"], df["low"], df["close"], CFG.ADX_PERIOD)
    df["atr"]        = _atr(df["high"], df["low"], df["close"], CFG.ATR_PERIOD)
    df["ema_diff"]   = df["ema_fast"] - df["ema_slow"]
    df["cross_up"]   = (df["ema_diff"].shift(1) <= 0) & (df["ema_diff"] > 0)
    df["cross_dn"]   = (df["ema_diff"].shift(1) >= 0) & (df["ema_diff"] < 0)
    df["trend_ok"]   = df["adx"] >= adx_threshold
    df["trend_fade"] = df["adx"] < adx_threshold * 0.80
    return df.dropna()


# ══════════════════════════════════════════════════════════════════════════════
# Simulated sentiment (for backtesting only)
# ══════════════════════════════════════════════════════════════════════════════

def _simulated_sentiment(df: pd.DataFrame, seed: int = 42) -> pd.Series:
    """
    Simulate sentiment scores for backtesting.

    Real historical sentiment data isn't stored, so we approximate using
    a seeded random process that produces realistic score distributions:
    - Mostly neutral (-0.3 to 0.3): ~60% of days
    - Mildly positive (0.3 to 0.6): ~20% of days
    - Strong positive (0.6+): ~10% of days
    - Negative (<-0.3): ~10% of days

    Using a fixed seed ensures results are reproducible across runs.
    """
    rng    = np.random.default_rng(seed)
    n      = len(df)
    scores = rng.normal(loc=0.05, scale=0.35, size=n)   # slight positive bias
    scores = np.clip(scores, -1.0, 1.0)
    return pd.Series(scores, index=df.index)


# ══════════════════════════════════════════════════════════════════════════════
# Core backtester
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(
    df:            pd.DataFrame,
    variant:       str   = "baseline",
    initial_capital: float = None,
    label:         str   = "",
) -> dict:
    """
    Run a vectorised backtest on a prepared OHLCV dataframe.

    variant options:
        "baseline"  — EMA + ADX >= 25, no sentiment
        "sentiment" — EMA + ADX >= 25, sentiment gate + sizing
        "low_adx"   — EMA + ADX >= 15, sentiment gate + sizing

    Returns a dict of performance metrics + equity curve + trade log.
    """
    if initial_capital is None:
        initial_capital = CFG.STARTING_CAPITAL

    adx_threshold = 15.0 if variant == "low_adx" else CFG.ADX_THRESHOLD
    use_sentiment = variant in ("sentiment", "low_adx")

    df = _prepare(df, adx_threshold)
    if df.empty:
        return {}

    sentiment = _simulated_sentiment(df) if use_sentiment else pd.Series(0.0, index=df.index)

    capital      = initial_capital
    position     = 0
    entry_price  = 0.0
    stop_loss    = 0.0
    equity_curve = []
    trades       = []

    for i, (idx, row) in enumerate(df.iterrows()):
        price   = row["close"]
        atr_val = row["atr"]
        sent    = sentiment.iloc[i]

        slip   = SLIPPAGE_BPS / 10_000
        comm   = COMMISSION_PER_SHARE

        # Stop-loss check
        if position > 0 and price <= stop_loss:
            fill_price = price * (1 - slip)
            pnl        = (fill_price - entry_price) * position - comm * position
            capital   += fill_price * position - comm * position
            trades.append({"date": idx, "side": "STOP", "price": fill_price,
                           "shares": position, "pnl": pnl})
            position = 0
            equity_curve.append(capital)
            continue

        # Entry
        if row["cross_up"] and row["trend_ok"] and position == 0:
            # Sentiment gate
            if use_sentiment and sent < CFG.SENTIMENT_GATE_THRESHOLD:
                equity_curve.append(capital)
                continue

            # Conviction multiplier
            conv_mult = 1.0
            if use_sentiment:
                for lo, hi, mult in CFG.SENTIMENT_SIZE_BANDS:
                    if lo <= sent < hi:
                        conv_mult = mult
                        break

            fill_price   = price * (1 + slip)
            risk_dollars = capital * CFG.RISK_PER_TRADE
            stop_dist    = atr_val * CFG.ATR_STOP_MULT
            if stop_dist > 0:
                shares = int((risk_dollars / stop_dist) * conv_mult)
                shares = min(shares, int((capital * CFG.MAX_POSITION_PCT) / fill_price))
                shares = min(shares, int(capital / fill_price))
                if shares > 0:
                    cost        = shares * fill_price + shares * comm
                    capital    -= cost
                    position    = shares
                    entry_price = fill_price
                    stop_loss   = fill_price - stop_dist
                    trades.append({"date": idx, "side": "BUY", "price": fill_price,
                                   "shares": shares, "pnl": 0})

        # Exit
        elif (row["cross_dn"] or row["trend_fade"]) and position > 0:
            fill_price = price * (1 - slip)
            pnl        = (fill_price - entry_price) * position - comm * position
            capital   += fill_price * position - comm * position
            trades.append({"date": idx, "side": "SELL", "price": fill_price,
                           "shares": position, "pnl": pnl})
            position = 0

        equity_curve.append(capital + position * price)

    # Close open position at last bar
    if position > 0:
        last_price   = df["close"].iloc[-1] * (1 - SLIPPAGE_BPS / 10_000)
        capital     += last_price * position - COMMISSION_PER_SHARE * position
        equity_curve[-1] = capital

    equity = pd.Series(equity_curve, index=df.index)
    return _metrics(equity, trades, initial_capital, label)


# ══════════════════════════════════════════════════════════════════════════════
# In-sample / out-of-sample split
# ══════════════════════════════════════════════════════════════════════════════

def run_insample_oos(
    df:      pd.DataFrame,
    variant: str = "baseline",
) -> dict:
    """
    Split df into in-sample (years 1-2) and out-of-sample (year 3),
    run the backtest on each, and return both sets of metrics.

    This is the key validation step — if the strategy only works in-sample
    but breaks in year 3, it's overfit.
    """
    if df.empty or len(df) < 100:
        log.warning("Not enough data for in-sample / out-of-sample split")
        return {}

    total_days = len(df)
    split_idx  = int(total_days * (CFG.BACKTEST_INSAMPLE_YEARS / CFG.BACKTEST_YEARS))

    df_insample = df.iloc[:split_idx]
    df_oos      = df.iloc[split_idx:]

    log.info(
        f"Backtest split — in-sample: {len(df_insample)} bars "
        f"({df_insample.index[0].date()} to {df_insample.index[-1].date()})  "
        f"| out-of-sample: {len(df_oos)} bars "
        f"({df_oos.index[0].date()} to {df_oos.index[-1].date()})"
    )

    insample_results = run_backtest(df_insample, variant=variant, label="IN-SAMPLE")
    oos_results      = run_backtest(df_oos,      variant=variant,
                                    initial_capital=insample_results.get("final_equity",
                                                    CFG.STARTING_CAPITAL),
                                    label="OUT-OF-SAMPLE")

    return {"in_sample": insample_results, "out_of_sample": oos_results}


# ══════════════════════════════════════════════════════════════════════════════
# Metrics calculation
# ══════════════════════════════════════════════════════════════════════════════

def _metrics(equity: pd.Series, trades: list, initial: float, label: str) -> dict:
    returns   = equity.pct_change().dropna()
    total_ret = (equity.iloc[-1] / initial - 1) * 100
    ann_ret   = returns.mean() * 252 * 100
    ann_vol   = returns.std() * np.sqrt(252) * 100
    sharpe    = (ann_ret / ann_vol) if ann_vol else 0.0
    drawdown  = ((equity / equity.cummax()) - 1).min() * 100

    trade_df   = pd.DataFrame(trades)
    exits      = trade_df[trade_df["side"].isin(["SELL", "STOP"])] if len(trade_df) else pd.DataFrame()
    n_trades   = len(exits)
    win_rate   = (len(exits[exits["pnl"] > 0]) / n_trades * 100) if n_trades else 0
    gross_p    = exits[exits["pnl"] > 0]["pnl"].sum() if n_trades else 0
    gross_l    = abs(exits[exits["pnl"] <= 0]["pnl"].sum()) if n_trades else 1e-9
    pf         = gross_p / gross_l

    return {
        "label":            label,
        "final_equity":     round(float(equity.iloc[-1]), 2),
        "total_return_%":   round(total_ret, 2),
        "annual_return_%":  round(ann_ret, 2),
        "annual_vol_%":     round(ann_vol, 2),
        "sharpe_ratio":     round(sharpe, 2),
        "max_drawdown_%":   round(drawdown, 2),
        "total_trades":     n_trades,
        "win_rate_%":       round(win_rate, 2),
        "profit_factor":    round(pf, 2),
        "equity_curve":     equity,
        "trades":           trade_df,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Pretty printer
# ══════════════════════════════════════════════════════════════════════════════

def print_results(results: dict, title: str = "BACKTEST RESULTS"):
    sep = "=" * 52
    log.info(sep)
    log.info(f"  {title}")
    log.info(sep)
    skip = {"equity_curve", "trades", "label"}
    for k, v in results.items():
        if k in skip:
            continue
        log.info(f"  {k:<24} {v}")
    # Quality flags
    sr = results.get("sharpe_ratio", 0)
    dd = results.get("max_drawdown_%", 0)
    pf = results.get("profit_factor", 0)
    if sr >= 1.0:
        log.info("  Sharpe >= 1.0       GOOD")
    else:
        log.info("  Sharpe < 1.0        WEAK — consider tuning")
    if abs(dd) <= 15:
        log.info("  Max drawdown <= 15% GOOD")
    else:
        log.info("  Max drawdown > 15%  HIGH — review risk settings")
    if pf >= 1.5:
        log.info("  Profit factor >= 1.5 GOOD")
    else:
        log.info("  Profit factor < 1.5  MARGINAL")
    log.info(sep)