"""
compare.py — Strategy comparison runner.

Runs all 3 variants on 3 years of historical data for a set of symbols,
prints a side-by-side comparison table, and recommends the best strategy.

Usage:
    python compare.py                    # uses default symbols
    python compare.py AAPL MSFT NVDA    # custom symbols

Variants compared:
    baseline   — EMA(9/21) + ADX >= 25
    sentiment  — EMA(9/21) + ADX >= 25 + sentiment gate + conviction sizing
    low_adx    — EMA(9/21) + ADX >= 15 + sentiment gate + conviction sizing
"""

import logging
import sys
from datetime import datetime, timedelta

import pandas as pd

from config import CFG
from data import get_daily_bars
from backtest import run_backtest, run_insample_oos, print_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

VARIANTS = ["baseline", "sentiment", "low_adx"]
VARIANT_LABELS = {
    "baseline":  "Baseline (EMA + ADX>=25)",
    "sentiment": "V2: Baseline + Sentiment",
    "low_adx":   "V3: Low ADX(15) + Sentiment",
}

DEFAULT_SYMBOLS = ["AAPL", "MSFT", "NVDA", "AMZN", "JPM"]


def fetch_3yr_data(symbol: str) -> pd.DataFrame:
    """Fetch 3 years of daily bars for a symbol."""
    n_days = CFG.BACKTEST_YEARS * 365 + 30   # buffer
    df = get_daily_bars(symbol, n_days=n_days)
    if df.empty:
        log.warning(f"No data returned for {symbol}")
    return df


def run_comparison(symbols: list = None) -> dict:
    """
    Run all variants across all symbols, aggregate results,
    and return a summary dict.
    """
    if symbols is None:
        symbols = DEFAULT_SYMBOLS

    log.info("=" * 60)
    log.info("  STRATEGY COMPARISON — 3 YEAR BACKTEST")
    log.info(f"  Symbols : {symbols}")
    log.info(f"  Variants: {VARIANTS}")
    log.info("=" * 60)

    # Aggregate metrics across symbols
    agg = {v: {"sharpe": [], "return": [], "drawdown": [],
               "win_rate": [], "profit_factor": [], "trades": []}
           for v in VARIANTS}

    for symbol in symbols:
        log.info(f"\n--- {symbol} ---")
        df = fetch_3yr_data(symbol)
        if df.empty or len(df) < 100:
            log.warning(f"Skipping {symbol} — insufficient data")
            continue

        for variant in VARIANTS:
            res = run_backtest(df.copy(), variant=variant,
                               label=f"{symbol} {VARIANT_LABELS[variant]}")
            if not res:
                continue

            agg[variant]["sharpe"].append(res["sharpe_ratio"])
            agg[variant]["return"].append(res["total_return_%"])
            agg[variant]["drawdown"].append(res["max_drawdown_%"])
            agg[variant]["win_rate"].append(res["win_rate_%"])
            agg[variant]["profit_factor"].append(res["profit_factor"])
            agg[variant]["trades"].append(res["total_trades"])

    # Build summary table
    summary = {}
    for v in VARIANTS:
        a = agg[v]
        if not a["sharpe"]:
            continue
        summary[v] = {
            "label":            VARIANT_LABELS[v],
            "avg_sharpe":       round(sum(a["sharpe"])   / len(a["sharpe"]),   2),
            "avg_return_%":     round(sum(a["return"])   / len(a["return"]),   1),
            "avg_drawdown_%":   round(sum(a["drawdown"]) / len(a["drawdown"]), 1),
            "avg_win_rate_%":   round(sum(a["win_rate"]) / len(a["win_rate"]), 1),
            "avg_profit_factor":round(sum(a["profit_factor"]) / len(a["profit_factor"]), 2),
            "avg_trades":       round(sum(a["trades"]) / len(a["trades"]), 0),
        }

    _print_comparison_table(summary)
    _recommend(summary)

    # Also run in-sample / out-of-sample on the first symbol for the winner
    if summary:
        winner   = max(summary, key=lambda v: summary[v]["avg_sharpe"])
        symbol   = symbols[0]
        df       = fetch_3yr_data(symbol)
        if not df.empty:
            log.info(f"\n--- In-Sample / Out-of-Sample validation: {symbol} ({VARIANT_LABELS[winner]}) ---")
            oos = run_insample_oos(df, variant=winner)
            if oos:
                print_results(oos["in_sample"],    "IN-SAMPLE  (Years 1-2)")
                print_results(oos["out_of_sample"],"OUT-OF-SAMPLE (Year 3 — unseen data)")
                _oos_verdict(oos)

    return summary


def _print_comparison_table(summary: dict):
    sep = "=" * 72
    log.info(f"\n{sep}")
    log.info("  RESULTS SUMMARY (averaged across symbols)")
    log.info(sep)
    header = (
        f"  {'Strategy':<30} {'Sharpe':>7} {'Return%':>8} "
        f"{'Drawdown%':>10} {'WinRate%':>9} {'ProfFact':>9} {'Trades':>7}"
    )
    log.info(header)
    log.info("-" * 72)
    for v, s in summary.items():
        log.info(
            f"  {s['label']:<30} "
            f"{s['avg_sharpe']:>7.2f} "
            f"{s['avg_return_%']:>8.1f} "
            f"{s['avg_drawdown_%']:>10.1f} "
            f"{s['avg_win_rate_%']:>9.1f} "
            f"{s['avg_profit_factor']:>9.2f} "
            f"{s['avg_trades']:>7.0f}"
        )
    log.info(sep)


def _recommend(summary: dict):
    if not summary:
        return

    # Normalize each metric to [0, 1] across variants before weighting
    # so that differences in scale don't bias the composite score.
    sharpes = [summary[v]["avg_sharpe"]        for v in summary]
    pfs     = [summary[v]["avg_profit_factor"] for v in summary]

    def _norm(val, vals):
        lo, hi = min(vals), max(vals)
        return (val - lo) / (hi - lo) if hi > lo else 0.5

    def score(v):
        s = summary[v]
        return _norm(s["avg_sharpe"], sharpes) * 0.6 + _norm(s["avg_profit_factor"], pfs) * 0.4

    winner = max(summary, key=score)
    s      = summary[winner]

    log.info("\n  RECOMMENDATION")
    log.info("  " + "-" * 50)
    log.info(f"  Best strategy: {s['label']}")
    log.info(f"  Avg Sharpe   : {s['avg_sharpe']}")
    log.info(f"  Avg Return   : {s['avg_return_%']}%")
    log.info(f"  Avg Drawdown : {s['avg_drawdown_%']}%")
    log.info("  " + "-" * 50)
    log.info("  Use this variant's parameters in config.py for live trading.")


def _oos_verdict(oos: dict):
    ins  = oos["in_sample"]
    out  = oos["out_of_sample"]
    log.info("\n  OUT-OF-SAMPLE VERDICT")
    log.info("  " + "-" * 50)

    sharpe_drop = ins["sharpe_ratio"] - out["sharpe_ratio"]
    if sharpe_drop <= 0.3:
        log.info("  Sharpe held up in OOS — strategy likely has real edge")
    elif sharpe_drop <= 0.7:
        log.info("  Sharpe degraded moderately — some overfitting, monitor closely")
    else:
        log.info("  Sharpe dropped significantly — possible overfit, re-tune parameters")

    if out["sharpe_ratio"] >= 0.5:
        log.info("  OOS Sharpe >= 0.5 — acceptable for paper trading")
    else:
        log.info("  OOS Sharpe < 0.5 — continue paper trading before going live")
    log.info("  " + "-" * 50)


if __name__ == "__main__":
    symbols = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_SYMBOLS
    run_comparison(symbols)