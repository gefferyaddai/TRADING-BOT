"""
notifier.py — Telegram push notifications and manual trade commands.

Notifications sent automatically:
    - Trade entry / exit / stop-loss
    - Drawdown guard triggered
    - Market open / session summary

Manual commands (send to your bot on Telegram):
    /status      — today's P&L and open positions
    /positions   — open positions only
    /performance — all-time stats from the database
    /sell AAPL   — manually close a specific position
    /sell all    — flatten all open positions
    /help        — list available commands
"""

import logging
import queue
import threading
import time

import requests

from config import CFG

log = logging.getLogger(__name__)

_TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"


class TelegramNotifier:
    def __init__(self):
        self.token   = CFG.TELEGRAM_BOT_TOKEN
        self.chat_id = CFG.TELEGRAM_CHAT_ID
        self.enabled = bool(
            self.token and self.token not in ("", "YOUR_TELEGRAM_TOKEN")
            and self.chat_id and self.chat_id not in ("", "YOUR_CHAT_ID")
        )
        self._offset   = 0
        self._running  = False
        self.commands: queue.Queue = queue.Queue()   # main loop reads from here

        if not self.enabled:
            log.info("Telegram disabled — set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to enable")

    # ── Core send ─────────────────────────────────────────────────────────────

    def send(self, text: str):
        if not self.enabled:
            return
        try:
            requests.post(
                _TELEGRAM_API.format(token=self.token, method="sendMessage"),
                json={"chat_id": self.chat_id, "text": text, "parse_mode": "Markdown"},
                timeout=10,
            )
        except Exception as e:
            log.warning(f"Telegram send failed: {e}")

    # ── Command polling (background thread) ───────────────────────────────────

    def _poll(self):
        while self._running:
            try:
                resp = requests.get(
                    _TELEGRAM_API.format(token=self.token, method="getUpdates"),
                    params={"offset": self._offset, "timeout": 30},
                    timeout=35,
                )
                for update in resp.json().get("result", []):
                    self._offset = update["update_id"] + 1
                    text = update.get("message", {}).get("text", "").strip()
                    if text.startswith("/"):
                        self.commands.put(text.lower())
            except Exception as e:
                log.warning(f"Telegram poll error: {e}")
                time.sleep(5)

    def start_polling(self):
        if not self.enabled:
            return
        self._running = True
        threading.Thread(target=self._poll, daemon=True).start()
        log.info("Telegram command polling started")

    def stop(self):
        self._running = False

    # ── Trade notifications ───────────────────────────────────────────────────

    def notify_entry(self, symbol: str, shares: int, price: float,
                     stop: float, conviction: float, regime: str):
        self.send(
            f"🟢 *ENTRY* `{symbol}`\n"
            f"Shares    : {shares} @ ${price:.2f}\n"
            f"Stop      : ${stop:.2f}\n"
            f"Conviction: {conviction:.2f}x | {regime}"
        )

    def notify_exit(self, symbol: str, shares: int, price: float,
                    pnl: float, reason: str):
        emoji = "✅" if pnl >= 0 else "🔴"
        self.send(
            f"{emoji} *{reason}* `{symbol}`\n"
            f"Shares: {shares} @ ${price:.2f}\n"
            f"P&L   : ${pnl:+.2f}"
        )

    def notify_drawdown(self, drawdown_pct: float):
        self.send(
            f"⚠️ *DRAWDOWN GUARD TRIGGERED*\n"
            f"Intraday loss: {drawdown_pct:.1f}%\n"
            f"Trading halted for the session"
        )

    def notify_market_open(self, watchlist: list, regime: str):
        preview = ", ".join(watchlist[:10])
        more    = f" +{len(watchlist)-10} more" if len(watchlist) > 10 else ""
        self.send(
            f"🔔 *MARKET OPEN*\n"
            f"Regime  : {regime}\n"
            f"Watching: `{preview}`{more}"
        )

    def notify_session_end(self, report: dict, end_equity: float):
        pf  = report.get("profit_factor", 0)
        pfs = "∞" if pf == float("inf") else f"{pf:.2f}"
        self.send(
            f"📊 *SESSION COMPLETE*\n"
            f"Equity        : ${end_equity:,.2f}\n"
            f"Closed trades : {report.get('total_closed_trades', 0)}\n"
            f"Win rate      : {report.get('win_rate_%', 0):.1f}%\n"
            f"Total P&L     : ${report.get('total_pnl', 0):+.2f}\n"
            f"Profit factor : {pfs}"
        )
