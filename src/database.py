"""
Database — SQLite persistence for trades, signals, snapshots, and regime history.
"""
import json
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from src.utils import setup_logging, utc_now

logger = setup_logging()


class Database:
    def __init__(self, db_path: str = "data/bot.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                quantity REAL NOT NULL,
                remaining_quantity REAL,
                usdt_value REAL,
                stop_loss REAL,
                take_profit REAL,
                highest_price REAL,
                pnl REAL,
                pnl_percent REAL,
                fees_paid REAL DEFAULT 0,
                status TEXT DEFAULT 'OPEN',
                regime_at_entry TEXT,
                sector TEXT,
                confluence_score INTEGER DEFAULT 0,
                exit_reason TEXT,
                tranche_exits TEXT DEFAULT '[]',
                entry_time TIMESTAMP,
                exit_time TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP,
                symbol TEXT,
                confluence_score INTEGER,
                score_breakdown TEXT,
                action_taken TEXT,
                regime TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP,
                total_value_usdt REAL,
                cash_usdt REAL,
                invested_usdt REAL,
                num_open_positions INTEGER,
                drawdown_from_peak REAL,
                market_regime TEXT,
                fear_greed_index INTEGER,
                news_sentiment_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS regime_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP,
                regime TEXT,
                confidence REAL,
                btc_price REAL,
                fear_greed INTEGER,
                news_sentiment REAL,
                btc_above_50sma INTEGER DEFAULT 0,
                btc_above_200sma INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS calibration_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP,
                parameter_name TEXT,
                old_value REAL,
                new_value REAL,
                reason TEXT,
                trades_analyzed INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
            CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
            CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time);
            CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);
            CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp);
            CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp ON portfolio_snapshots(timestamp);
            """)
        logger.debug("Database initialized: %s", self.db_path)

    # ------------------------------------------------------------------
    # Trades
    # ------------------------------------------------------------------

    def save_trade(self, trade: dict) -> int:
        with self._conn() as conn:
            cursor = conn.execute("""
                INSERT INTO trades (
                    symbol, side, entry_price, exit_price, quantity,
                    remaining_quantity, usdt_value, stop_loss, take_profit,
                    highest_price, pnl, pnl_percent, fees_paid, status,
                    regime_at_entry, sector, confluence_score, exit_reason,
                    tranche_exits, entry_time, exit_time
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                trade.get("symbol"), trade.get("side", "BUY"),
                trade.get("entry_price"), trade.get("exit_price"),
                trade.get("quantity"), trade.get("remaining_quantity", trade.get("quantity")),
                trade.get("usdt_value"), trade.get("stop_loss"), trade.get("take_profit"),
                trade.get("highest_price", trade.get("entry_price")),
                trade.get("pnl"), trade.get("pnl_percent"),
                trade.get("fees_paid", 0), trade.get("status", "OPEN"),
                trade.get("regime_at_entry"), trade.get("sector"),
                trade.get("confluence_score", 0), trade.get("exit_reason"),
                trade.get("tranche_exits", "[]"),
                trade.get("entry_time"), trade.get("exit_time"),
            ))
            return cursor.lastrowid

    def update_trade(self, trade_id: int, updates: dict):
        if not updates:
            return
        cols = ", ".join(f"{k}=?" for k in updates)
        vals = list(updates.values()) + [trade_id]
        with self._conn() as conn:
            conn.execute(f"UPDATE trades SET {cols} WHERE id=?", vals)

    def get_open_trades(self) -> list:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM trades WHERE status='OPEN' ORDER BY entry_time ASC"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_trade_by_id(self, trade_id: int) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM trades WHERE id=?", (trade_id,)).fetchone()
            return dict(row) if row else None

    def get_closed_trades(self, limit: int = 100) -> list:
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT * FROM trades WHERE status IN ('CLOSED','STOPPED_OUT','PARTIAL')
                ORDER BY exit_time DESC LIMIT ?
            """, (limit,)).fetchall()
            return [dict(r) for r in rows]

    def get_trades_for_period(self, days: int = 1) -> list:
        since = utc_now() - timedelta(days=days)
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT * FROM trades WHERE exit_time >= ?
                ORDER BY exit_time DESC
            """, (since,)).fetchall()
            return [dict(r) for r in rows]

    def get_trade_stats(self, lookback_trades: int = 50) -> dict:
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT pnl, pnl_percent, stop_loss, exit_reason
                FROM trades WHERE status IN ('CLOSED','STOPPED_OUT')
                ORDER BY exit_time DESC LIMIT ?
            """, (lookback_trades,)).fetchall()

        if not rows:
            return {
                "total_trades": 0, "win_rate": 0.55,
                "avg_win": 0.06, "avg_loss": 0.03,
                "profit_factor": 0, "avg_rr": 2.0,
                "stop_loss_hit_rate": 0,
            }

        trades = [dict(r) for r in rows]
        wins = [t for t in trades if (t.get("pnl") or 0) > 0]
        losses = [t for t in trades if (t.get("pnl") or 0) <= 0]
        total = len(trades)
        win_rate = len(wins) / total if total > 0 else 0.55

        avg_win_pct = (sum(t.get("pnl_percent", 0) for t in wins) / len(wins) / 100
                       if wins else 0.06)
        avg_loss_pct = (abs(sum(t.get("pnl_percent", 0) for t in losses)) / len(losses) / 100
                        if losses else 0.03)

        total_wins_usd = sum(t.get("pnl", 0) for t in wins)
        total_losses_usd = abs(sum(t.get("pnl", 0) for t in losses))
        profit_factor = total_wins_usd / total_losses_usd if total_losses_usd > 0 else 0

        avg_rr = avg_win_pct / avg_loss_pct if avg_loss_pct > 0 else 2.0

        stop_hits = sum(1 for t in trades if (t.get("exit_reason") or "").startswith("STOP"))
        stop_loss_hit_rate = stop_hits / total if total > 0 else 0

        return {
            "total_trades": total,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": win_rate,
            "avg_win": avg_win_pct,
            "avg_loss": avg_loss_pct,
            "profit_factor": profit_factor,
            "avg_rr": avg_rr,
            "stop_loss_hit_rate": stop_loss_hit_rate,
        }

    def get_daily_pnl(self) -> float:
        since = utc_now().replace(hour=0, minute=0, second=0, microsecond=0)
        with self._conn() as conn:
            row = conn.execute("""
                SELECT COALESCE(SUM(pnl), 0) as total
                FROM trades WHERE exit_time >= ? AND status IN ('CLOSED','STOPPED_OUT')
            """, (since,)).fetchone()
            return row["total"] if row else 0.0

    def get_consecutive_losses(self) -> int:
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT pnl FROM trades WHERE status IN ('CLOSED','STOPPED_OUT')
                ORDER BY exit_time DESC LIMIT 10
            """).fetchall()
        count = 0
        for row in rows:
            if (row["pnl"] or 0) < 0:
                count += 1
            else:
                break
        return count

    def get_peak_portfolio_value(self) -> float:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT MAX(total_value_usdt) as peak FROM portfolio_snapshots"
            ).fetchone()
            return row["peak"] if row and row["peak"] else 0.0

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    def save_signal(self, signal: dict):
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO signals (timestamp, symbol, confluence_score,
                    score_breakdown, action_taken, regime)
                VALUES (?,?,?,?,?,?)
            """, (
                signal.get("timestamp", utc_now()),
                signal.get("symbol"), signal.get("confluence_score", 0),
                signal.get("score_breakdown"), signal.get("action_taken"),
                signal.get("regime"),
            ))

    def get_recent_signals(self, limit: int = 20) -> list:
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT * FROM signals
                ORDER BY timestamp DESC LIMIT ?
            """, (limit,)).fetchall()
            return [dict(r) for r in rows]

    def get_signals_for_symbol(self, symbol: str, limit: int = 10) -> list:
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT * FROM signals WHERE symbol=?
                ORDER BY timestamp DESC LIMIT ?
            """, (symbol, limit)).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Portfolio Snapshots
    # ------------------------------------------------------------------

    def save_snapshot(self, snapshot: dict):
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO portfolio_snapshots (
                    timestamp, total_value_usdt, cash_usdt, invested_usdt,
                    num_open_positions, drawdown_from_peak, market_regime,
                    fear_greed_index, news_sentiment_score
                ) VALUES (?,?,?,?,?,?,?,?,?)
            """, (
                snapshot.get("timestamp", utc_now()),
                snapshot.get("total_value_usdt", 0),
                snapshot.get("cash_usdt", 0),
                snapshot.get("invested_usdt", 0),
                snapshot.get("num_open_positions", 0),
                snapshot.get("drawdown_from_peak", 0),
                snapshot.get("market_regime"),
                snapshot.get("fear_greed_index", 50),
                snapshot.get("news_sentiment_score", 0),
            ))

    def get_snapshots_for_period(self, days: int = 1) -> list:
        since = utc_now() - timedelta(days=days)
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT * FROM portfolio_snapshots WHERE timestamp >= ?
                ORDER BY timestamp ASC
            """, (since,)).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Regime
    # ------------------------------------------------------------------

    def save_regime(self, regime: dict):
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO regime_history (
                    timestamp, regime, confidence, btc_price,
                    fear_greed, news_sentiment, btc_above_50sma, btc_above_200sma
                ) VALUES (?,?,?,?,?,?,?,?)
            """, (
                regime.get("timestamp", utc_now()),
                regime.get("regime"), regime.get("confidence", 0),
                regime.get("btc_price"), regime.get("fear_greed"),
                regime.get("news_sentiment"), regime.get("btc_above_50sma", False),
                regime.get("btc_above_200sma", False),
            ))

    def get_recent_regime(self, limit: int = 10) -> list:
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT * FROM regime_history ORDER BY timestamp DESC LIMIT ?
            """, (limit,)).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def save_calibration(self, cal: dict):
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO calibration_log (
                    timestamp, parameter_name, old_value, new_value,
                    reason, trades_analyzed
                ) VALUES (?,?,?,?,?,?)
            """, (
                utc_now(), cal.get("parameter_name"),
                cal.get("old_value"), cal.get("new_value"),
                cal.get("reason"), cal.get("trades_analyzed", 0),
            ))
