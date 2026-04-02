import json
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiosqlite
from sqlalchemy import (
    Column, Integer, Float, String, Boolean, DateTime, Text,
    create_engine, event,
)
from sqlalchemy.orm import declarative_base, Session, sessionmaker

from src.utils import setup_logging, utc_now

logger = setup_logging()
Base = declarative_base()


class Trade(Base):
    __tablename__ = "trades"
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, nullable=False)
    side = Column(String, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    quantity = Column(Float, nullable=False)
    usdt_value = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=False)
    take_profit = Column(Float, nullable=True)
    confluence_score = Column(Integer, nullable=False)
    regime_at_entry = Column(String, nullable=False)
    status = Column(String, nullable=False, default="OPEN")
    pnl = Column(Float, nullable=True)
    pnl_percent = Column(Float, nullable=True)
    fees_paid = Column(Float, default=0.0)
    entry_time = Column(DateTime, nullable=False)
    exit_time = Column(DateTime, nullable=True)
    exit_reason = Column(String, nullable=True)
    sector = Column(String, nullable=True)
    highest_price = Column(Float, nullable=True)
    remaining_quantity = Column(Float, nullable=True)
    tranche_exits = Column(Text, default="[]")


class PortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshots"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False)
    total_value_usdt = Column(Float, nullable=False)
    cash_usdt = Column(Float, nullable=False)
    invested_usdt = Column(Float, nullable=False)
    num_open_positions = Column(Integer, nullable=False)
    drawdown_from_peak = Column(Float, default=0.0)
    market_regime = Column(String, nullable=True)
    fear_greed_index = Column(Integer, nullable=True)
    news_sentiment_score = Column(Float, nullable=True)


class SignalLog(Base):
    __tablename__ = "signals_log"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False)
    symbol = Column(String, nullable=False)
    confluence_score = Column(Integer, nullable=False)
    score_breakdown = Column(Text, nullable=True)
    action_taken = Column(String, nullable=False)
    regime = Column(String, nullable=True)


class MarketRegime(Base):
    __tablename__ = "market_regimes"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False)
    regime = Column(String, nullable=False)
    confidence = Column(Float, nullable=True)
    btc_price = Column(Float, nullable=True)
    fear_greed = Column(Integer, nullable=True)
    news_sentiment = Column(Float, nullable=True)
    btc_above_50sma = Column(Boolean, nullable=True)
    btc_above_200sma = Column(Boolean, nullable=True)


class CalibrationHistory(Base):
    __tablename__ = "calibration_history"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False)
    parameter_name = Column(String, nullable=False)
    old_value = Column(Float, nullable=False)
    new_value = Column(Float, nullable=False)
    reason = Column(String, nullable=True)
    trade_count_at_calibration = Column(Integer, nullable=True)


class WatchdogEvent(Base):
    __tablename__ = "watchdog_events"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False)
    event_type = Column(String, nullable=False)
    details = Column(Text, nullable=True)
    resolved = Column(Boolean, default=False)


class BotState(Base):
    __tablename__ = "bot_state"
    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String, unique=True, nullable=False)
    value = Column(Text, nullable=True)
    updated_at = Column(DateTime, nullable=False)


class Database:
    def __init__(self, db_path: str = "data/bot.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def init_db(self):
        Base.metadata.create_all(self.engine)
        logger.info("Database initialized at %s", self.db_path)

    def get_session(self) -> Session:
        return self.SessionLocal()

    def save_trade(self, trade_data: dict) -> int:
        session = self.get_session()
        try:
            trade = Trade(**trade_data)
            session.add(trade)
            session.commit()
            trade_id = trade.id
            return trade_id
        finally:
            session.close()

    def update_trade(self, trade_id: int, updates: dict):
        session = self.get_session()
        try:
            session.query(Trade).filter(Trade.id == trade_id).update(updates)
            session.commit()
        finally:
            session.close()

    def get_open_trades(self) -> list:
        session = self.get_session()
        try:
            trades = session.query(Trade).filter(Trade.status == "OPEN").all()
            result = []
            for t in trades:
                result.append({
                    "id": t.id, "symbol": t.symbol, "side": t.side,
                    "entry_price": t.entry_price, "quantity": t.quantity,
                    "usdt_value": t.usdt_value, "stop_loss": t.stop_loss,
                    "take_profit": t.take_profit, "confluence_score": t.confluence_score,
                    "regime_at_entry": t.regime_at_entry, "status": t.status,
                    "entry_time": t.entry_time, "sector": t.sector,
                    "highest_price": t.highest_price or t.entry_price,
                    "remaining_quantity": t.remaining_quantity or t.quantity,
                    "tranche_exits": json.loads(t.tranche_exits or "[]"),
                })
            return result
        finally:
            session.close()

    def get_recent_trades(self, limit: int = 50) -> list:
        session = self.get_session()
        try:
            trades = (
                session.query(Trade)
                .filter(Trade.status != "OPEN")
                .order_by(Trade.exit_time.desc())
                .limit(limit)
                .all()
            )
            result = []
            for t in trades:
                result.append({
                    "id": t.id, "symbol": t.symbol, "entry_price": t.entry_price,
                    "exit_price": t.exit_price, "pnl": t.pnl,
                    "pnl_percent": t.pnl_percent, "exit_reason": t.exit_reason,
                    "entry_time": t.entry_time, "exit_time": t.exit_time,
                    "fees_paid": t.fees_paid,
                })
            return result
        finally:
            session.close()

    def get_trade_stats(self, lookback: int = 50) -> dict:
        trades = self.get_recent_trades(lookback)
        if not trades:
            return {
                "win_rate": 0.55, "avg_win": 0.03, "avg_loss": 0.02,
                "total_trades": 0, "profit_factor": 1.5,
                "stop_loss_hit_rate": 0.3, "avg_rr": 1.5,
            }
        wins = [t for t in trades if (t["pnl"] or 0) > 0]
        losses = [t for t in trades if (t["pnl"] or 0) <= 0]
        stopped = [t for t in trades if t["exit_reason"] == "STOP_LOSS"]
        win_rate = len(wins) / len(trades) if trades else 0.55
        avg_win = (
            sum(abs(t["pnl_percent"] or 0) for t in wins) / len(wins)
            if wins else 0.03
        )
        avg_loss = (
            sum(abs(t["pnl_percent"] or 0) for t in losses) / len(losses)
            if losses else 0.02
        )
        gross_profit = sum(t["pnl"] or 0 for t in wins)
        gross_loss = abs(sum(t["pnl"] or 0 for t in losses)) or 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 1.5
        stop_rate = len(stopped) / len(trades) if trades else 0.3
        avg_rr = (avg_win / avg_loss) if avg_loss > 0 else 1.5
        return {
            "win_rate": win_rate, "avg_win": avg_win, "avg_loss": avg_loss,
            "total_trades": len(trades), "profit_factor": profit_factor,
            "stop_loss_hit_rate": stop_rate, "avg_rr": avg_rr,
        }

    def get_total_trade_count(self) -> int:
        session = self.get_session()
        try:
            return session.query(Trade).filter(Trade.status != "OPEN").count()
        finally:
            session.close()

    def save_snapshot(self, snapshot_data: dict):
        session = self.get_session()
        try:
            snap = PortfolioSnapshot(**snapshot_data)
            session.add(snap)
            session.commit()
        finally:
            session.close()

    def get_peak_portfolio_value(self) -> float:
        session = self.get_session()
        try:
            from sqlalchemy import func
            result = session.query(func.max(PortfolioSnapshot.total_value_usdt)).scalar()
            return result or 0.0
        finally:
            session.close()

    def save_signal(self, signal_data: dict):
        session = self.get_session()
        try:
            sig = SignalLog(**signal_data)
            session.add(sig)
            session.commit()
        finally:
            session.close()

    def save_regime(self, regime_data: dict):
        session = self.get_session()
        try:
            r = MarketRegime(**regime_data)
            session.add(r)
            session.commit()
        finally:
            session.close()

    def get_latest_regime(self) -> Optional[str]:
        session = self.get_session()
        try:
            r = session.query(MarketRegime).order_by(MarketRegime.timestamp.desc()).first()
            return r.regime if r else None
        finally:
            session.close()

    def save_calibration(self, cal_data: dict):
        session = self.get_session()
        try:
            c = CalibrationHistory(**cal_data)
            session.add(c)
            session.commit()
        finally:
            session.close()

    def get_recent_calibrations(self, limit: int = 10) -> list:
        session = self.get_session()
        try:
            cals = (
                session.query(CalibrationHistory)
                .order_by(CalibrationHistory.timestamp.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "timestamp": c.timestamp, "parameter_name": c.parameter_name,
                    "old_value": c.old_value, "new_value": c.new_value,
                    "reason": c.reason,
                }
                for c in cals
            ]
        finally:
            session.close()

    def save_watchdog_event(self, event_data: dict):
        session = self.get_session()
        try:
            w = WatchdogEvent(**event_data)
            session.add(w)
            session.commit()
        finally:
            session.close()

    def save_state(self, key: str, value: str):
        session = self.get_session()
        try:
            existing = session.query(BotState).filter(BotState.key == key).first()
            if existing:
                existing.value = value
                existing.updated_at = utc_now()
            else:
                state = BotState(key=key, value=value, updated_at=utc_now())
                session.add(state)
            session.commit()
        finally:
            session.close()

    def load_state(self, key: str) -> Optional[str]:
        session = self.get_session()
        try:
            state = session.query(BotState).filter(BotState.key == key).first()
            return state.value if state else None
        finally:
            session.close()

    def get_daily_pnl(self) -> float:
        session = self.get_session()
        try:
            today = utc_now().replace(hour=0, minute=0, second=0, microsecond=0)
            trades = (
                session.query(Trade)
                .filter(Trade.exit_time >= today, Trade.status != "OPEN")
                .all()
            )
            return sum(t.pnl or 0 for t in trades)
        finally:
            session.close()

    def get_consecutive_losses(self) -> int:
        session = self.get_session()
        try:
            trades = (
                session.query(Trade)
                .filter(Trade.status != "OPEN")
                .order_by(Trade.exit_time.desc())
                .limit(20)
                .all()
            )
            count = 0
            for t in trades:
                if (t.pnl or 0) < 0:
                    count += 1
                else:
                    break
            return count
        finally:
            session.close()

    def get_trades_for_period(self, days: int = 7) -> list:
        session = self.get_session()
        try:
            from datetime import timedelta
            cutoff = utc_now() - timedelta(days=days)
            trades = (
                session.query(Trade)
                .filter(Trade.exit_time >= cutoff, Trade.status != "OPEN")
                .order_by(Trade.exit_time.desc())
                .all()
            )
            return [
                {
                    "id": t.id, "symbol": t.symbol,
                    "entry_price": t.entry_price, "exit_price": t.exit_price,
                    "pnl_usdt": t.pnl or 0, "pnl_pct": (t.pnl_percent or 0) * 100,
                    "exit_reason": t.exit_reason, "sector": t.sector,
                    "entry_time": t.entry_time, "exit_time": t.exit_time,
                }
                for t in trades
            ]
        finally:
            session.close()

    def get_snapshots_for_period(self, days: int = 30) -> list:
        session = self.get_session()
        try:
            from datetime import timedelta
            cutoff = utc_now() - timedelta(days=days)
            snaps = (
                session.query(PortfolioSnapshot)
                .filter(PortfolioSnapshot.timestamp >= cutoff)
                .order_by(PortfolioSnapshot.timestamp.asc())
                .all()
            )
            return [
                {"timestamp": s.timestamp, "total_value_usdt": s.total_value_usdt}
                for s in snaps
            ]
        finally:
            session.close()
