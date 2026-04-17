"""
risk.py — Position sizing, stop-loss management, drawdown guard,
          and end-of-session trade evaluation.

Public API:
    position_size(capital, atr, price, conviction_mult) → int
    stop_loss_price(entry, atr, side)                   → float
    check_drawdown(equity_start, equity_now)            → bool
    evaluate_session(trades)                            → dict
"""

import logging
from typing import List, Dict, Any

import pandas as pd
import numpy as np

from config import CFG

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Position sizing
# ══════════════════════════════════════════════════════════════════════════════

def position_size(capital: float, atr: float, price: float,
                  conviction_mult: float = 1.0) -> int:
    """
    Calculate shares to buy so that a 2×ATR adverse move equals
    exactly RISK_PER_TRADE% of capital — then scaled by conviction_mult.

    conviction_mult is the combined output of sentiment sizing × regime scaling:
        - CHOPPY market alone           → 0.25
        - Strong sentiment alone        → up to 2.0
        - Strong sentiment + CHOPPY     → 2.0 × 0.25 = 0.5
        - Strong sentiment + TRENDING   → 2.0

    Hard caps still apply regardless of multiplier:
        - Position value ≤ MAX_POSITION_PCT of capital
        - Cannot spend more than available cash
    """
    if atr <= 0 or price <= 0 or capital <= 0:
        return 0

    risk_dollars   = capital * CFG.RISK_PER_TRADE
    stop_distance  = atr * CFG.ATR_STOP_MULT
    shares_by_risk = int((risk_dollars / stop_distance) * conviction_mult)

    # Cap by max position percentage
    max_by_pct     = int((capital * CFG.MAX_POSITION_PCT) / price)
    shares         = min(shares_by_risk, max_by_pct)

    # Cap by available capital
    max_by_capital = int(capital / price)
    shares         = min(shares, max_by_capital)

    if shares < 1:
        log.debug(
            f"position_size=0  "
            f"capital={capital:.0f}  atr={atr:.4f}  "
            f"price={price:.2f}  mult={conviction_mult:.2f}"
        )
        return 0

    cost = shares * price
    log.debug(
        f"position_size={shares}  @ ${price:.2f} = ${cost:.0f}  "
        f"risk=${shares * stop_distance:.0f}  mult={conviction_mult:.2f}"
    )
    return shares


# ══════════════════════════════════════════════════════════════════════════════
# Stop-loss price
# ══════════════════════════════════════════════════════════════════════════════

def stop_loss_price(entry: float, atr: float, side: str = "long") -> float:
    """
    Hard stop-loss level.
    Long : stop = entry - (ATR × ATR_STOP_MULT)
    Short: stop = entry + (ATR × ATR_STOP_MULT)
    """
    offset = atr * CFG.ATR_STOP_MULT
    return round(entry - offset if side == "long" else entry + offset, 4)


# ══════════════════════════════════════════════════════════════════════════════
# Drawdown guard
# ══════════════════════════════════════════════════════════════════════════════

def check_drawdown(equity_start: float, equity_now: float) -> bool:
    """
    Return True if intraday drawdown has breached DAILY_DRAWDOWN_LIMIT.
    Caller should halt new orders when this returns True.
    """
    if equity_start <= 0:
        return False
    drawdown = (equity_start - equity_now) / equity_start
    if drawdown >= CFG.DAILY_DRAWDOWN_LIMIT:
        log.warning(
            f"DRAWDOWN GUARD TRIGGERED  "
            f"loss={drawdown*100:.1f}%  "
            f"(limit={CFG.DAILY_DRAWDOWN_LIMIT*100:.0f}%)"
        )
        return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
# Session evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_session(trades: List[Dict[str, Any]]) -> dict:
    """
    Analyse completed trades for a session and print a diagnostic report.
    Called when drawdown guard fires or at market close.

    Each trade dict should contain:
        symbol, side ('BUY'/'SELL'/'STOP'), qty, price, entry_price, pnl
    """
    if not trades:
        log.info("No trades to evaluate this session.")
        return {}

    df    = pd.DataFrame(trades)
    exits = df[df["side"].isin(["SELL", "STOP"])].copy()

    if exits.empty:
        log.info("No closed trades to evaluate yet.")
        return {"total_trades": 0}

    total         = len(exits)
    winners       = exits[exits["pnl"] > 0]
    losers        = exits[exits["pnl"] <= 0]
    win_rate      = len(winners) / total * 100
    avg_pnl       = exits["pnl"].mean()
    total_pnl     = exits["pnl"].sum()
    avg_win       = winners["pnl"].mean() if len(winners) else 0
    avg_loss      = losers["pnl"].mean()  if len(losers)  else 0
    gross_profit  = winners["pnl"].sum()     if len(winners) else 0.0
    gross_loss    = abs(losers["pnl"].sum()) if len(losers)  else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
    best          = exits.loc[exits["pnl"].idxmax()]
    worst         = exits.loc[exits["pnl"].idxmin()]

    report = {
        "total_closed_trades": total,
        "win_rate_%":          round(win_rate, 1),
        "total_pnl":           round(total_pnl, 2),
        "avg_pnl_per_trade":   round(avg_pnl, 2),
        "avg_winner":          round(avg_win, 2),
        "avg_loser":           round(avg_loss, 2),
        "profit_factor":       round(profit_factor, 2),
        "best_trade_pnl":      round(float(best["pnl"]),  2),
        "worst_trade_pnl":     round(float(worst["pnl"]), 2),
        "best_symbol":         best.get("symbol", ""),
        "worst_symbol":        worst.get("symbol", ""),
    }

    _print_evaluation(report)
    return report


def _print_evaluation(r: dict):
    sep = "=" * 52
    log.info(sep)
    log.info("  SESSION EVALUATION")
    log.info(sep)
    log.info(f"  Closed trades   : {r['total_closed_trades']}")
    log.info(f"  Win rate        : {r['win_rate_%']}%")
    log.info(f"  Total P&L       : ${r['total_pnl']:+.2f}")
    log.info(f"  Avg P&L / trade : ${r['avg_pnl_per_trade']:+.2f}")
    log.info(f"  Avg winner      : ${r['avg_winner']:+.2f}")
    log.info(f"  Avg loser       : ${r['avg_loser']:+.2f}")
    pf     = r["profit_factor"]
    pf_str = "∞ (no losses)" if pf == float("inf") else f"{pf:.2f}"
    log.info(f"  Profit factor   : {pf_str}")
    log.info(f"  Best trade      : {r['best_symbol']}  ${r['best_trade_pnl']:+.2f}")
    log.info(f"  Worst trade     : {r['worst_symbol']}  ${r['worst_trade_pnl']:+.2f}")
    if pf < 1.0:
        log.info("  WARNING: Profit factor < 1 — strategy losing money overall")
    elif pf < 1.5:
        log.info("  INFO: Profit factor 1-1.5 — marginal edge, watch closely")
    else:
        log.info("  OK: Profit factor > 1.5 — healthy edge detected")
    log.info(sep)