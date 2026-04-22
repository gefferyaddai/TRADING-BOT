"""
database.py — SQLite trade journal and performance tracking.

Tables:
    trades           — every entry, exit, and stop with full P&L
    equity_snapshots — portfolio value recorded every scan loop
    sessions         — daily summary stats
"""

import logging
import sqlite3
from datetime import datetime
from typing import Dict

log     = logging.getLogger(__name__)
DB_PATH = "trades.db"


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _conn() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT    NOT NULL,
                session_date TEXT   NOT NULL,
                symbol      TEXT    NOT NULL,
                side        TEXT    NOT NULL,
                shares      INTEGER NOT NULL,
                price       REAL    NOT NULL,
                entry_price REAL    NOT NULL,
                pnl         REAL    NOT NULL
            )
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS equity_snapshots (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT NOT NULL,
                equity      REAL NOT NULL,
                positions   INTEGER NOT NULL
            )
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                date          TEXT PRIMARY KEY,
                total_trades  INTEGER,
                win_rate      REAL,
                total_pnl     REAL,
                profit_factor REAL,
                start_equity  REAL,
                end_equity    REAL
            )
        """)
    log.info("Database initialised — trades.db")


def record_trade(symbol: str, side: str, shares: int,
                 price: float, entry_price: float, pnl: float):
    with _conn() as db:
        db.execute("""
            INSERT INTO trades
                (timestamp, session_date, symbol, side, shares, price, entry_price, pnl)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            datetime.now().date().isoformat(),
            symbol, side, shares, price, entry_price, round(pnl, 2),
        ))


def record_equity(equity: float, positions: int):
    with _conn() as db:
        db.execute("""
            INSERT INTO equity_snapshots (timestamp, equity, positions)
            VALUES (?, ?, ?)
        """, (datetime.now().isoformat(), round(equity, 2), positions))


def record_session(report: dict, start_equity: float, end_equity: float):
    pf = report.get("profit_factor", 0)
    if pf == float("inf"):
        pf = 9999.0
    with _conn() as db:
        db.execute("""
            INSERT OR REPLACE INTO sessions
                (date, total_trades, win_rate, total_pnl, profit_factor, start_equity, end_equity)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().date().isoformat(),
            report.get("total_closed_trades", 0),
            report.get("win_rate_%", 0.0),
            report.get("total_pnl", 0.0),
            pf,
            round(start_equity, 2),
            round(end_equity, 2),
        ))


# ── Query helpers for Telegram commands ──────────────────────────────────────

def get_today_summary() -> str:
    today = datetime.now().date().isoformat()
    with _conn() as db:
        row = db.execute("""
            SELECT COUNT(*) as n,
                   COALESCE(SUM(pnl), 0) as total_pnl,
                   COALESCE(AVG(CASE WHEN pnl > 0 THEN 1.0 ELSE 0.0 END) * 100, 0) as win_rate
            FROM trades
            WHERE side IN ('SELL', 'STOP') AND session_date = ?
        """, (today,)).fetchone()
    if not row["n"]:
        return "No closed trades today"
    return (
        f"*Today ({today})*\n"
        f"Closed trades : {row['n']}\n"
        f"Win rate      : {row['win_rate']:.1f}%\n"
        f"P&L           : ${row['total_pnl']:+.2f}"
    )


def get_all_time_summary() -> str:
    with _conn() as db:
        row = db.execute("""
            SELECT COUNT(*) as n,
                   COALESCE(SUM(pnl), 0) as total_pnl,
                   COALESCE(AVG(CASE WHEN pnl > 0 THEN 1.0 ELSE 0.0 END) * 100, 0) as win_rate,
                   COALESCE(SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END), 0) as gross_profit,
                   COALESCE(ABS(SUM(CASE WHEN pnl <= 0 THEN pnl ELSE 0 END)), 0) as gross_loss
            FROM trades
            WHERE side IN ('SELL', 'STOP')
        """).fetchone()
        sessions = db.execute("SELECT COUNT(*) as n FROM sessions").fetchone()["n"]

    if not row["n"]:
        return "No closed trades recorded yet"

    gl   = row["gross_loss"]
    pf   = (row["gross_profit"] / gl) if gl > 0 else float("inf")
    pf_s = "∞" if pf == float("inf") else f"{pf:.2f}"

    return (
        f"*All-time performance*\n"
        f"Sessions      : {sessions}\n"
        f"Closed trades : {row['n']}\n"
        f"Win rate      : {row['win_rate']:.1f}%\n"
        f"Total P&L     : ${row['total_pnl']:+.2f}\n"
        f"Profit factor : {pf_s}"
    )


def get_positions_summary(positions: Dict[str, dict]) -> str:
    if not positions:
        return "No open positions"
    lines = ["*Open positions*"]
    for sym, p in positions.items():
        lines.append(f"`{sym}` — {p['qty']} shares @ ${p['avg_entry']:.2f}")
    return "\n".join(lines)
