"""
bot.py — Main trading loop.

Lifecycle:
    1. Pre-market  : screener + sentiment fetch + regime check
    2. Market open : every 60s → bars → signal → execute with conviction sizing
    3. Drawdown guard fires → halt + evaluate
    4. Market close → close all + final evaluation

Run:
    python bot.py
"""

import json
import logging
import os
import time
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional

_ET         = ZoneInfo("America/New_York")
_STATE_FILE = "bot_state.json"

from config import CFG
from data import get_minute_bars, get_spy_bars, run_morning_screen
from strategy import (
    get_signal, get_regime, get_conviction_multiplier,
    indicator_snapshot, Signal, Regime,
)
from risk import (
    position_size, stop_loss_price,
    check_drawdown, evaluate_session,
)
from sentiment import get_sentiment_scores

logging.basicConfig(
    level=getattr(logging, CFG.LOG_LEVEL),
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(CFG.LOG_FILE),
    ],
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Broker wrapper
# ══════════════════════════════════════════════════════════════════════════════

class Broker:
    def __init__(self):
        try:
            import alpaca_trade_api as tradeapi
        except ImportError:
            raise ImportError("Run:  pip install alpaca-trade-api")
        base_url = (
            "https://paper-api.alpaca.markets"
            if CFG.PAPER else "https://api.alpaca.markets"
        )
        self.api = tradeapi.REST(
            CFG.API_KEY, CFG.API_SECRET, base_url, api_version="v2"
        )
        log.info(f"Broker — {'PAPER' if CFG.PAPER else 'LIVE'}")

    def get_equity(self) -> float:
        return float(self.api.get_account().equity)

    def get_cash(self) -> float:
        return float(self.api.get_account().cash)

    def get_position_qty(self, symbol: str) -> int:
        try:
            return int(self.api.get_position(symbol).qty)
        except Exception:
            return 0

    def get_all_positions(self) -> Dict[str, dict]:
        return {p.symbol: {"qty": int(p.qty), "avg_entry": float(p.avg_entry_price)}
                for p in self.api.list_positions()}

    def buy(self, symbol: str, qty: int) -> Optional[str]:
        if qty <= 0:
            return None
        try:
            o = self.api.submit_order(symbol=symbol, qty=qty, side="buy",
                                      type="market", time_in_force="day")
            log.info(f"  BUY  {qty:>5} {symbol:<6}  order={o.id}")
            return o.id
        except Exception as e:
            log.error(f"  BUY failed {symbol}: {e}")
            return None

    def sell(self, symbol: str, qty: int, reason: str = "SIGNAL") -> Optional[str]:
        if qty <= 0:
            return None
        try:
            o = self.api.submit_order(symbol=symbol, qty=qty, side="sell",
                                      type="market", time_in_force="day")
            log.info(f"  SELL [{reason}] {qty:>5} {symbol:<6}  order={o.id}")
            return o.id
        except Exception as e:
            log.error(f"  SELL failed {symbol}: {e}")
            return None

    def close_all_positions(self):
        try:
            self.api.close_all_positions()
            log.info("All positions closed.")
        except Exception as e:
            log.error(f"close_all_positions: {e}")

    def is_market_open(self) -> bool:
        return self.api.get_clock().is_open

    def get_clock(self) -> dict:
        c = self.api.get_clock()
        return {"is_open": c.is_open, "next_open": str(c.next_open)}


# ══════════════════════════════════════════════════════════════════════════════
# Bot state
# ══════════════════════════════════════════════════════════════════════════════

class BotState:
    def __init__(self, equity: float):
        self.session_equity:     float            = equity
        self.watchlist:          List[str]         = []
        self.sentiment_scores:   Dict[str, float]  = {}
        self.regime:             Regime            = Regime.TRENDING
        self.stop_losses:        Dict[str, float]  = {}
        self.entry_prices:       Dict[str, float]  = {}
        self.trades:             List[dict]        = []
        self.halted:             bool              = False
        self.screener_run_today: bool              = False


# ══════════════════════════════════════════════════════════════════════════════
# Main bot
# ══════════════════════════════════════════════════════════════════════════════

class TradingBot:
    def __init__(self):
        self.broker = Broker()
        equity      = self.broker.get_equity()
        self.state  = BotState(equity=equity)
        self._restore_state()
        log.info(f"Bot ready — equity=${equity:,.2f}")

    def _save_state(self):
        payload = {
            "stop_losses":   self.state.stop_losses,
            "entry_prices":  self.state.entry_prices,
        }
        try:
            with open(_STATE_FILE, "w") as f:
                json.dump(payload, f)
        except Exception as e:
            log.warning(f"Could not save bot state: {e}")

    def _restore_state(self):
        if not os.path.exists(_STATE_FILE):
            return
        try:
            with open(_STATE_FILE) as f:
                payload = json.load(f)
            live_positions = set(self.broker.get_all_positions().keys())
            # Only restore state for positions we actually still hold
            self.state.stop_losses  = {s: v for s, v in payload.get("stop_losses",  {}).items() if s in live_positions}
            self.state.entry_prices = {s: v for s, v in payload.get("entry_prices", {}).items() if s in live_positions}
            if self.state.stop_losses:
                log.info(f"Restored stop-losses for: {list(self.state.stop_losses.keys())}")
        except Exception as e:
            log.warning(f"Could not restore bot state: {e}")

    def _is_near_close(self) -> bool:
        n = datetime.now(_ET)
        return (n.hour > CFG.MARKET_CLOSE_HOUR or
                (n.hour == CFG.MARKET_CLOSE_HOUR and n.minute >= CFG.MARKET_CLOSE_MIN))

    def _should_run_screener(self) -> bool:
        n = datetime.now(_ET)
        return (not self.state.screener_run_today and
                n.hour == CFG.PRE_MARKET_HOUR and n.minute >= CFG.PRE_MARKET_MIN)

    def run_morning_prep(self):
        """Run screener, sentiment fetch, and regime check before market open."""
        log.info("=" * 52)
        log.info("  PRE-MARKET PREP")
        log.info("=" * 52)

        # 1. Stock screener with sector cap
        self.state.watchlist = run_morning_screen()
        self.state.screener_run_today = True

        # 2. Sentiment scoring for today's watchlist
        if self.state.watchlist and CFG.FINNHUB_API_KEY != "YOUR_FINNHUB_KEY":
            self.state.sentiment_scores = get_sentiment_scores(self.state.watchlist)
        else:
            log.info("Sentiment skipped (no Finnhub key) — all scores default to 0.0")
            self.state.sentiment_scores = {s: 0.0 for s in self.state.watchlist}

        # 3. Market regime check
        spy_df = get_spy_bars(n_days=30)
        self.state.regime = get_regime(spy_df)

        # 4. Reset session state
        self.state.session_equity = self.broker.get_equity()
        self.state.trades.clear()
        self.state.halted = False

        log.info(
            f"Ready — watchlist={self.state.watchlist}  "
            f"regime={self.state.regime.value}  "
            f"equity=${self.state.session_equity:,.2f}"
        )

    def _check_symbol(self, symbol: str,
                      open_positions: Dict[str, dict],
                      bought_this_scan: set):
        df = get_minute_bars(symbol, CFG.BARS_TO_FETCH)
        if df is None or df.empty:
            return

        sentiment     = self.state.sentiment_scores.get(symbol, 0.0)
        snap          = indicator_snapshot(df)
        signal        = get_signal(df, sentiment_score=sentiment)
        pos_info      = open_positions.get(symbol)
        qty           = int(pos_info["qty"]) if pos_info else 0
        current_price = float(df["close"].iloc[-1])
        atr_val       = snap.get("atr", 0)

        log.debug(
            f"  {symbol:<6}  signal={signal.value:<4}  "
            f"pos={qty}  sent={sentiment:+.2f}  {snap}"
        )

        # Stop-loss check
        if qty > 0 and symbol in self.state.stop_losses:
            if current_price <= self.state.stop_losses[symbol]:
                log.info(
                    f"  STOP-LOSS {symbol}  "
                    f"price={current_price:.2f}  stop={self.state.stop_losses[symbol]:.2f}"
                )
                if self.broker.sell(symbol, qty, reason="STOP"):
                    self._record_exit(symbol, qty, current_price, "STOP")
                    self._clear_state(symbol)
                return

        # Entry — use snapshot of positions + this-scan purchases to avoid race
        if signal == Signal.BUY and qty == 0 and symbol not in bought_this_scan:
            if len(open_positions) + len(bought_this_scan) >= CFG.MAX_OPEN_POSITIONS:
                return

            conviction = get_conviction_multiplier(sentiment, self.state.regime)
            capital    = self.broker.get_cash()
            shares     = position_size(capital, atr_val, current_price,
                                       conviction_mult=conviction)

            if shares > 0:
                if self.broker.buy(symbol, shares):
                    bought_this_scan.add(symbol)
                    stop = stop_loss_price(current_price, atr_val)
                    self.state.stop_losses[symbol]  = stop
                    self.state.entry_prices[symbol] = current_price
                    self._save_state()
                    self._record_entry(symbol, shares, current_price)
                    log.info(
                        f"  ENTRY {symbol}  shares={shares}  "
                        f"stop={stop:.2f}  conviction={conviction:.2f}  "
                        f"regime={self.state.regime.value}"
                    )

        # Exit
        elif signal == Signal.SELL and qty > 0:
            if self.broker.sell(symbol, qty, reason="SIGNAL"):
                self._record_exit(symbol, qty, current_price, "SIGNAL")
                self._clear_state(symbol)

    def _clear_state(self, symbol: str):
        self.state.stop_losses.pop(symbol, None)
        self.state.entry_prices.pop(symbol, None)
        self._save_state()

    def _record_entry(self, symbol: str, qty: int, price: float):
        self.state.trades.append({
            "time": datetime.now().isoformat(), "symbol": symbol,
            "side": "BUY", "qty": qty, "price": price,
            "entry_price": price, "pnl": 0,
        })

    def _record_exit(self, symbol: str, qty: int, price: float, side: str):
        entry = self.state.entry_prices.get(symbol, price)
        pnl   = round((price - entry) * qty, 2)
        self.state.trades.append({
            "time": datetime.now().isoformat(), "symbol": symbol,
            "side": side, "qty": qty, "price": price,
            "entry_price": entry, "pnl": pnl,
        })
        log.info(f"  PNL {symbol}  entry={entry:.2f}  exit={price:.2f}  pnl=${pnl:+.2f}")

    def run(self):
        log.info("=" * 52)
        log.info("  TRADING BOT STARTING")
        log.info(f"  Mode    : {'PAPER' if CFG.PAPER else 'LIVE'}")
        log.info(f"  Capital : ${CFG.STARTING_CAPITAL:,.2f}")
        log.info(f"  Risk    : {CFG.RISK_PER_TRADE*100:.0f}% per trade")
        log.info("=" * 52)

        while True:
            try:
                now = datetime.now(_ET)

                if now.hour == 0 and now.minute < 2:
                    self.state.screener_run_today = False

                if self._should_run_screener():
                    self.run_morning_prep()

                if not self.broker.is_market_open():
                    clock = self.broker.get_clock()
                    log.info(f"Market closed. Next open: {clock['next_open']}")
                    time.sleep(300)
                    continue

                if self._is_near_close():
                    log.info("Near market close — closing all positions.")
                    self.broker.close_all_positions()
                    evaluate_session(self.state.trades)
                    time.sleep(3600)
                    self.state.screener_run_today = False
                    continue

                current_equity = self.broker.get_equity()
                if check_drawdown(self.state.session_equity, current_equity):
                    if not self.state.halted:
                        self.state.halted = True
                        log.warning("Trading HALTED — drawdown limit hit. Evaluating ...")
                        evaluate_session(self.state.trades)
                    time.sleep(CFG.LOOP_SLEEP_SEC)
                    continue

                if self.state.halted:
                    time.sleep(CFG.LOOP_SLEEP_SEC)
                    continue

                if not self.state.watchlist:
                    self.run_morning_prep()
                    if not self.state.watchlist:
                        time.sleep(60)
                        continue

                open_positions = self.broker.get_all_positions()
                log.info(
                    f"Scanning {len(self.state.watchlist)} symbols | "
                    f"equity=${current_equity:,.2f} | "
                    f"regime={self.state.regime.value} | "
                    f"positions={len(open_positions)}"
                )

                bought_this_scan: set = set()
                for symbol in self.state.watchlist:
                    self._check_symbol(symbol, open_positions, bought_this_scan)
                    time.sleep(0.2)

                time.sleep(CFG.LOOP_SLEEP_SEC)

            except KeyboardInterrupt:
                log.info("Shutting down ...")
                self.broker.close_all_positions()
                evaluate_session(self.state.trades)
                break

            except Exception as e:
                log.error(f"Main loop error: {e}", exc_info=True)
                time.sleep(30)


if __name__ == "__main__":
    bot = TradingBot()
    bot.run()