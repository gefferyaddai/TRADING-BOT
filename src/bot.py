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
from database import (
    init_db, record_trade, record_equity,
    record_session, get_today_summary,
    get_all_time_summary, get_positions_summary,
)
from notifier import TelegramNotifier

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
        self.stop_cooldowns:     Dict[str, float]  = {}  # symbol → timestamp of last stop-loss
        self.trades:             List[dict]        = []
        self.halted:             bool              = False
        self.screener_run_today: bool              = False


# ══════════════════════════════════════════════════════════════════════════════
# Main bot
# ══════════════════════════════════════════════════════════════════════════════

class TradingBot:
    def __init__(self):
        init_db()
        self.broker   = Broker()
        self.notifier = TelegramNotifier()
        equity        = self.broker.get_equity()
        self.state    = BotState(equity=equity)
        self._restore_state()
        self.notifier.start_polling()
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

    def _is_regular_hours(self) -> bool:
        n = datetime.now(_ET)
        after_open  = (n.hour > 9) or (n.hour == 9 and n.minute >= 30)
        before_close = (n.hour < CFG.MARKET_CLOSE_HOUR or
                        (n.hour == CFG.MARKET_CLOSE_HOUR and n.minute < CFG.MARKET_CLOSE_MIN))
        return after_open and before_close

    def _is_near_close(self) -> bool:
        n = datetime.now(_ET)
        return (n.hour > CFG.MARKET_CLOSE_HOUR or
                (n.hour == CFG.MARKET_CLOSE_HOUR and n.minute >= CFG.MARKET_CLOSE_MIN))

    def _should_run_screener(self) -> bool:
        n = datetime.now(_ET)
        return (not self.state.screener_run_today and
                n.hour == CFG.PRE_MARKET_HOUR and n.minute >= CFG.PRE_MARKET_MIN)

    # ── Telegram command handler ──────────────────────────────────────────────

    def _process_commands(self):
        """Drain the Telegram command queue and act on each command."""
        while not self.notifier.commands.empty():
            try:
                cmd = self.notifier.commands.get_nowait()
            except Exception:
                break

            log.info(f"Telegram command received: {cmd}")

            if cmd == "/status":
                today   = get_today_summary()
                pos_txt = get_positions_summary(self.broker.get_all_positions())
                self.notifier.send(f"{today}\n\n{pos_txt}")

            elif cmd == "/positions":
                self.notifier.send(get_positions_summary(self.broker.get_all_positions()))

            elif cmd == "/performance":
                self.notifier.send(get_all_time_summary())

            elif cmd.startswith("/sell all"):
                self._manual_sell_all()

            elif cmd.startswith("/sell "):
                symbol = cmd.split("/sell ", 1)[1].strip().upper()
                self._manual_sell(symbol)

            elif cmd == "/help":
                self.notifier.send(
                    "*Available commands*\n"
                    "/status — today P&L + open positions\n"
                    "/positions — open positions\n"
                    "/performance — all-time stats\n"
                    "/sell SYMBOL — close one position\n"
                    "/sell all — close all positions\n"
                    "/help — this message"
                )
            else:
                self.notifier.send(f"Unknown command: `{cmd}`\nSend /help for options.")

    def _manual_sell(self, symbol: str):
        positions = self.broker.get_all_positions()
        if symbol not in positions:
            self.notifier.send(f"No open position in `{symbol}`")
            return
        qty  = positions[symbol]["qty"]
        bars = get_minute_bars(symbol, 5)
        price = float(bars["close"].iloc[-1]) if not bars.empty else positions[symbol]["avg_entry"]
        if self.broker.sell(symbol, qty, reason="MANUAL"):
            self._record_exit(symbol, qty, price, "SELL")
            self._clear_state(symbol)
            self.notifier.send(f"✅ Manual sell executed: `{symbol}` {qty} shares @ ${price:.2f}")

    def _manual_sell_all(self):
        positions = self.broker.get_all_positions()
        if not positions:
            self.notifier.send("No open positions to close")
            return
        for symbol, pos in positions.items():
            qty  = pos["qty"]
            bars = get_minute_bars(symbol, 5)
            price = float(bars["close"].iloc[-1]) if not bars.empty else pos["avg_entry"]
            if self.broker.sell(symbol, qty, reason="MANUAL"):
                self._record_exit(symbol, qty, price, "SELL")
                self._clear_state(symbol)
        self.notifier.send(f"✅ All positions closed manually ({len(positions)} symbols)")

    # ── Morning prep ─────────────────────────────────────────────────────────

    def run_morning_prep(self):
        log.info("=" * 52)
        log.info("  PRE-MARKET PREP")
        log.info("=" * 52)

        self.state.watchlist = run_morning_screen()
        self.state.screener_run_today = True

        if self.state.watchlist and CFG.FINNHUB_API_KEY != "YOUR_FINNHUB_KEY":
            self.state.sentiment_scores = get_sentiment_scores(self.state.watchlist)
        else:
            log.info("Sentiment skipped (no Finnhub key) — all scores default to 0.0")
            self.state.sentiment_scores = {s: 0.0 for s in self.state.watchlist}

        spy_df = get_spy_bars(n_days=30)
        self.state.regime = get_regime(spy_df)

        self.state.session_equity = self.broker.get_equity()
        self.state.trades.clear()
        self.state.halted = False

        self.notifier.notify_market_open(self.state.watchlist, self.state.regime.value)
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

        sentiment        = self.state.sentiment_scores.get(symbol, 0.0)
        snap, enriched   = indicator_snapshot(df)
        signal           = get_signal(enriched, sentiment_score=sentiment, _enriched=True)
        pos_info         = open_positions.get(symbol)
        qty              = int(pos_info["qty"]) if pos_info else 0
        current_price    = float(df["close"].iloc[-1])
        atr_val          = snap.get("atr", 0)

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
                    self.state.stop_cooldowns[symbol] = time.time()
                    log.info(f"  COOLDOWN {symbol} — blocked for 60 min after stop")
                return

        # Entry — use snapshot of positions + this-scan purchases to avoid race
        cooldown_until = self.state.stop_cooldowns.get(symbol, 0)
        if time.time() < cooldown_until + 3600:
            log.debug(f"  {symbol} skipped — in stop cooldown")
            return

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
                    self.notifier.notify_entry(
                        symbol, shares, current_price, stop,
                        conviction, self.state.regime.value,
                    )
                    log.info(
                        f"  ENTRY {symbol}  shares={shares}  "
                        f"stop={stop:.2f}  conviction={conviction:.2f}  "
                        f"regime={self.state.regime.value}"
                    )

        # Exit
        elif signal == Signal.SELL and qty > 0:
            if self.broker.sell(symbol, qty, reason="SIGNAL"):
                self._record_exit(symbol, qty, current_price, "SELL")
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
        record_trade(symbol, "BUY", qty, price, price, 0.0)

    def _record_exit(self, symbol: str, qty: int, price: float, side: str):
        entry = self.state.entry_prices.get(symbol, price)
        pnl   = round((price - entry) * qty, 2)
        self.state.trades.append({
            "time": datetime.now().isoformat(), "symbol": symbol,
            "side": side, "qty": qty, "price": price,
            "entry_price": entry, "pnl": pnl,
        })
        record_trade(symbol, side, qty, price, entry, pnl)
        self.notifier.notify_exit(symbol, qty, price, pnl, side)
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

                if not self.broker.is_market_open() or not self._is_regular_hours():
                    clock = self.broker.get_clock()
                    log.info(f"Outside regular hours (9:30–3:45 ET). Next open: {clock['next_open']}")
                    time.sleep(60)
                    continue

                if self._is_near_close():
                    log.info("Near market close — closing all positions.")
                    for sym, pos in self.broker.get_all_positions().items():
                        bars      = get_minute_bars(sym, 5)
                        eod_price = float(bars["close"].iloc[-1]) if not bars.empty else pos["avg_entry"]
                        self._record_exit(sym, pos["qty"], eod_price, "SELL")
                        self._clear_state(sym)
                    self.broker.close_all_positions()
                    report     = evaluate_session(self.state.trades)
                    end_equity = self.broker.get_equity()
                    record_session(report, self.state.session_equity, end_equity)
                    self.notifier.notify_session_end(report, end_equity)
                    time.sleep(3600)
                    self.state.screener_run_today = False
                    continue

                current_equity = self.broker.get_equity()
                record_equity(current_equity, len(self.broker.get_all_positions()))

                if check_drawdown(self.state.session_equity, current_equity):
                    if not self.state.halted:
                        self.state.halted = True
                        dd_pct = (self.state.session_equity - current_equity) / self.state.session_equity * 100
                        log.warning("Trading HALTED — drawdown limit hit. Evaluating ...")
                        report = evaluate_session(self.state.trades)
                        self.notifier.notify_drawdown(dd_pct)
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

                self._process_commands()

                bought_this_scan: set = set()
                for symbol in self.state.watchlist:
                    self._check_symbol(symbol, open_positions, bought_this_scan)
                    time.sleep(0.2)

                time.sleep(CFG.LOOP_SLEEP_SEC)

            except KeyboardInterrupt:
                log.info("Shutting down ...")
                self.notifier.stop()
                self.broker.close_all_positions()
                report = evaluate_session(self.state.trades)
                self.notifier.notify_session_end(report, self.broker.get_equity())
                break

            except Exception as e:
                log.error(f"Main loop error: {e}", exc_info=True)
                time.sleep(30)


if __name__ == "__main__":
    bot = TradingBot()
    bot.run()