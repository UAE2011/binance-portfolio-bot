"""
Binance Spot Portfolio Trading Bot — Main Orchestrator

Features integrated from SOL bot:
  - AI trading (OpenAI-compatible API / gemini-2.5-flash)
  - 10% wallet rule, 3% SL, 6% TP, trailing stop, 50/50 partial TP
  - RSI + MACD + Bollinger + S/R + support break exit + volume spike
  - Multi-timeframe 15m + 1hr + 4h
  - Daily loss limit, weekly report, news sentiment filter
  - Memory after restart, auto-restart watchdog
  - Diversified crypto portfolio (not just SOL)
"""

import os
import sys
import json
import signal
import asyncio
from datetime import datetime, timezone

from config.settings import Settings
from src.utils import setup_logging, utc_now
from src.database import Database
from src.exchange import BinanceExchange
from src.indicators import IndicatorSet
from src.regime import RegimeDetector
from src.news_intelligence import NewsIntelligence
from src.risk_manager import RiskManager
from src.portfolio import PortfolioManager
from src.scanner import AssetScanner
from src.strategy import (
    ConfluenceScorer, SignalGenerator,
    SupportResistanceEngine, VolumeSpikeDetector,
)
from src.ai_advisor import AIAdvisor
from src.calibrator import SelfCalibrator
from src.notifier import TelegramNotifier, build_command_handlers
from src.watchdog import Watchdog

logger = setup_logging(Settings.ops.LOG_LEVEL)

STATE_FILE = Settings.ops.STATE_FILE


class TradingBot:
    def __init__(self):
        # Core infrastructure
        self.db = Database(Settings.DATABASE_PATH)
        self.exchange = BinanceExchange(
            Settings.BINANCE_API_KEY,
            Settings.BINANCE_API_SECRET,
            testnet=Settings.TESTNET,
        )

        # Intelligence layers
        self.news = NewsIntelligence()
        self.regime = RegimeDetector(self.exchange, self.db)
        self.ai = AIAdvisor()

        # Risk & portfolio
        self.risk = RiskManager(self.db, self.regime)
        self.risk.set_news_intel(self.news)
        self.portfolio = PortfolioManager(
            self.exchange, self.db, self.regime, self.risk, self.news,
        )

        # Strategy
        self.sr_engine = SupportResistanceEngine()
        self.scanner = AssetScanner(self.exchange, self.news)
        self.scorer = ConfluenceScorer(self.regime, self.news, self.db, self.sr_engine)
        self.signal_gen = SignalGenerator(self.scorer, self.regime, self.news, self.ai)

        # Calibration
        self.calibrator = SelfCalibrator(self.db, self.risk)

        # Notifications
        self.notifier = TelegramNotifier(
            Settings.TELEGRAM_BOT_TOKEN,
            Settings.TELEGRAM_CHAT_ID,
        )

        # Watchdog
        self.watchdog = Watchdog(self.exchange, self.db)

        # Data caches — multi-timeframe: 15m, 1h, 4h
        self.indicator_cache: dict = {}   # symbol -> {tf -> IndicatorSet}
        self.kline_history: dict = {}     # symbol -> {tf -> [candles]}
        self.watchlist: list = []

        # State
        self._shutdown = False
        self._last_scan_time: float = 0
        self._last_news_time: float = 0
        self._last_regime_time: float = 0
        self._last_rebalance_day: int = -1
        self._last_weekly_report_day: int = -1

    async def start(self):
        logger.info("=" * 60)
        logger.info("BINANCE PORTFOLIO BOT STARTING")
        logger.info("Testnet: %s | AI: %s | Model: %s",
                     Settings.TESTNET, Settings.ai.ENABLED, Settings.ai.MODEL)
        logger.info("=" * 60)

        # Setup signal handlers
        for sig in (signal.SIGTERM, signal.SIGINT):
            asyncio.get_event_loop().add_signal_handler(
                sig, lambda: asyncio.create_task(self._shutdown_handler()),
            )

        # Initialize all components
        self.db.init_db()
        await self.exchange.initialize()
        await self.news.update_all()
        await self.regime.initialize()
        self.risk.initialize_from_history()
        self.calibrator.initialize()

        # Restore state from last run (memory after restart)
        await self._restore_state()

        # Sync portfolio with exchange
        await self.portfolio.sync_with_exchange()

        # Register Telegram commands
        handlers = build_command_handlers(
            self.portfolio, self.risk, self.calibrator,
            self.watchdog, self.regime, self.news, self.ai,
        )
        self.notifier.register_commands(handlers)

        # Startup notification
        await self.notifier.send_message(
            f"🟢 *Bot Started*\n"
            f"Mode: {'TESTNET' if Settings.TESTNET else 'LIVE'}\n"
            f"AI: {'Enabled (' + Settings.ai.MODEL + ')' if Settings.ai.ENABLED else 'Disabled'}\n"
            f"Portfolio: ${self.portfolio.portfolio_value:.2f}\n"
            f"Regime: {self.regime.current_regime}\n"
            f"Fear/Greed: {self.news.fear_greed_value} ({self.news.fear_greed_label})\n"
            f"Open positions: {len(self.portfolio.open_positions)}"
        )

        # Initial scan
        await self._initial_scan()

        # Start watchdog
        await self.watchdog.start()

        # Launch all loops
        await asyncio.gather(
            self._main_loop(),
            self._position_monitor_loop(),
            self._scheduled_tasks_loop(),
            self.notifier.start_polling(),
        )

    # ------------------------------------------------------------------
    # State Persistence (Memory After Restart)
    # ------------------------------------------------------------------

    async def _restore_state(self):
        """Restore bot state from database and state file."""
        try:
            # Restore watchlist
            saved_watchlist = self.db.load_state("watchlist")
            if saved_watchlist:
                self.watchlist = json.loads(saved_watchlist)
                logger.info("Restored watchlist: %d symbols", len(self.watchlist))

            # Restore last scan/news/regime times
            last_scan = self.db.load_state("last_scan_time")
            if last_scan:
                self._last_scan_time = float(last_scan)

            last_news = self.db.load_state("last_news_time")
            if last_news:
                self._last_news_time = float(last_news)

            # Restore indicator caches from state file
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE, "r") as f:
                    state = json.load(f)
                logger.info("Restored state file: %s", list(state.keys()))

            logger.info("State restoration complete")
        except Exception as e:
            logger.warning("State restoration failed (fresh start): %s", e)

    def _save_state(self):
        """Persist current state for restart recovery."""
        try:
            self.db.save_state("watchlist", json.dumps(self.watchlist))
            self.db.save_state("last_scan_time", str(self._last_scan_time))
            self.db.save_state("last_news_time", str(self._last_news_time))

            # Save minimal state to file
            state = {
                "saved_at": utc_now().isoformat(),
                "watchlist": self.watchlist,
                "regime": self.regime.current_regime,
                "portfolio_value": self.portfolio.portfolio_value,
                "open_positions": len(self.portfolio.open_positions),
            }
            os.makedirs(os.path.dirname(STATE_FILE) or ".", exist_ok=True)
            with open(STATE_FILE, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error("State save failed: %s", e)

    # ------------------------------------------------------------------
    # Initial Scan & Indicator Seeding
    # ------------------------------------------------------------------

    async def _initial_scan(self):
        """Scan market, build watchlist, seed indicators for all timeframes."""
        logger.info("Running initial market scan...")
        candidates = await self.scanner.scan_universe()
        self.watchlist = self.scanner.get_watchlist_symbols()

        # Add symbols from open positions
        for pos in self.portfolio.open_positions:
            if pos["symbol"] not in self.watchlist:
                self.watchlist.append(pos["symbol"])

        logger.info("Watchlist: %d symbols — %s", len(self.watchlist),
                     ", ".join(self.watchlist[:10]))

        # Seed indicators for all timeframes
        for symbol in self.watchlist:
            await self._seed_indicators(symbol)

        # Compute S/R levels for all watchlist symbols
        for symbol in self.watchlist:
            history_1h = self.kline_history.get(symbol, {}).get("1h", [])
            if history_1h:
                self.sr_engine.compute_levels(symbol, history_1h)

        # Start WebSocket for 15m klines (primary entry timeframe)
        if self.watchlist:
            await self.exchange.start_kline_stream(
                self.watchlist, "15m", self._on_kline_update,
            )

        self._save_state()
        logger.info("Initial scan complete")

    async def _seed_indicators(self, symbol: str):
        """Seed indicators for 15m, 1h, and 4h timeframes."""
        self.indicator_cache[symbol] = {}
        self.kline_history[symbol] = {}

        for tf, limit in [("15m", 200), ("1h", 200), ("4h", 200)]:
            try:
                klines = await self.exchange.get_klines(symbol, tf, limit)
                if not klines:
                    continue
                ind_set = IndicatorSet()
                ind_set.seed_from_klines(klines)
                self.indicator_cache[symbol][tf] = ind_set
                self.kline_history[symbol][tf] = klines[-100:]
            except Exception as e:
                logger.debug("Failed to seed %s %s: %s", symbol, tf, e)

    # ------------------------------------------------------------------
    # WebSocket Kline Handler
    # ------------------------------------------------------------------

    async def _on_kline_update(self, data: dict):
        """Process real-time 15m kline updates from WebSocket."""
        try:
            symbol = data.get("s", "")
            kline = data.get("k", {})
            is_closed = kline.get("x", False)

            candle = {
                "open": float(kline.get("o", 0)),
                "high": float(kline.get("h", 0)),
                "low": float(kline.get("l", 0)),
                "close": float(kline.get("c", 0)),
                "volume": float(kline.get("v", 0)),
            }

            if symbol not in self.indicator_cache:
                return

            ind_15m = self.indicator_cache[symbol].get("15m")
            if ind_15m is None:
                return

            if is_closed:
                ind_15m.update_candle(candle)
                # Store in history
                history = self.kline_history.get(symbol, {}).get("15m", [])
                history.append(candle)
                if len(history) > 200:
                    history.pop(0)
                self.kline_history.setdefault(symbol, {})["15m"] = history
                self.watchdog.beat()
            else:
                ind_15m.update_tick(candle["close"])

        except Exception as e:
            logger.debug("Kline update error: %s", e)

    # ------------------------------------------------------------------
    # Main Trading Loop
    # ------------------------------------------------------------------

    async def _main_loop(self):
        """Main trading cycle — scans watchlist and generates signals."""
        logger.info("Main trading loop started (interval=%ds)",
                     Settings.strategy.SCAN_INTERVAL_SECONDS)
        while not self._shutdown:
            try:
                await self._trading_cycle()
            except Exception as e:
                logger.error("Trading cycle error: %s", e)
            await asyncio.sleep(Settings.strategy.SCAN_INTERVAL_SECONDS)

    async def _trading_cycle(self):
        self.watchdog.beat()

        if self.risk.kill_switch_active:
            return
        if self.risk.is_paused:
            return

        regime_params = self.regime.get_regime_params()
        if not regime_params.get("entries_allowed", False):
            return

        # Refresh 1h and 4h indicators for watchlist
        await self._refresh_higher_timeframes()

        portfolio_context = self.portfolio.get_status()

        for symbol in self.watchlist:
            if symbol in [p["symbol"] for p in self.portfolio.open_positions]:
                continue  # Already holding

            ind_15m = self.indicator_cache.get(symbol, {}).get("15m", IndicatorSet()).latest
            ind_1h = self.indicator_cache.get(symbol, {}).get("1h", IndicatorSet()).latest
            ind_4h = self.indicator_cache.get(symbol, {}).get("4h", IndicatorSet()).latest
            history_1h = self.kline_history.get(symbol, {}).get("1h", [])

            if not ind_1h:
                continue

            signal = await self.signal_gen.evaluate(
                symbol, ind_15m, ind_1h, ind_4h, history_1h,
                portfolio_context,
            )

            if signal:
                trade = await self.portfolio.execute_entry(signal)
                if trade:
                    await self.notifier.notify_entry(trade)
                    self.calibrator.check_and_calibrate()
                    self._save_state()

    async def _refresh_higher_timeframes(self):
        """Refresh 1h and 4h indicators from REST API."""
        for symbol in self.watchlist:
            for tf in ["1h", "4h"]:
                try:
                    klines = await self.exchange.get_klines(symbol, tf, 5)
                    if not klines:
                        continue
                    ind_set = self.indicator_cache.get(symbol, {}).get(tf)
                    if ind_set is None:
                        continue
                    # Update with latest candles (deduplicated)
                    history = self.kline_history.get(symbol, {}).get(tf, [])
                    for k in klines:
                        # Simple dedup: check if candle open time differs
                        if not history or k["close"] != history[-1]["close"] or k["volume"] != history[-1]["volume"]:
                            ind_set.update_candle(k)
                            history.append(k)
                            if len(history) > 200:
                                history.pop(0)
                    self.kline_history.setdefault(symbol, {})[tf] = history
                except Exception as e:
                    logger.debug("Refresh %s %s failed: %s", symbol, tf, e)

    # ------------------------------------------------------------------
    # Position Monitor Loop
    # ------------------------------------------------------------------

    async def _position_monitor_loop(self):
        """Monitor open positions: trailing stops, partial TP, support break exit."""
        logger.info("Position monitor loop started (15s interval)")
        while not self._shutdown:
            try:
                await self._monitor_positions()
            except Exception as e:
                logger.error("Position monitor error: %s", e)
            await asyncio.sleep(15)

    async def _monitor_positions(self):
        self.watchdog.beat()

        for trade in list(self.portfolio.open_positions):
            symbol = trade["symbol"]
            try:
                current_price = await self.exchange.get_price(symbol)
                if current_price <= 0:
                    continue

                ind = self.indicator_cache.get(symbol, {}).get("1h", IndicatorSet())
                current_atr = ind.latest.get("atr", current_price * 0.03)

                # --- 1. Update trailing stop ---
                stop_update = self.risk.update_trailing_stop(trade, current_price, current_atr)
                if stop_update["stop_loss"] != trade["stop_loss"]:
                    self.db.update_trade(trade["id"], stop_update)
                    trade.update(stop_update)

                # --- 2. Check stop-loss hit ---
                if current_price <= trade["stop_loss"]:
                    result = await self.portfolio.execute_exit(
                        trade, trade.get("remaining_quantity", trade["quantity"]),
                        "STOP_LOSS", current_price,
                    )
                    if result:
                        await self.notifier.notify_exit(result)
                        self.calibrator.check_and_calibrate()
                        self._save_state()
                    continue

                # --- 3. Check 50/50 partial take-profit ---
                tranches = self.risk.check_take_profit_tranches(
                    trade, current_price, current_atr,
                )
                for tranche in tranches:
                    result = await self.portfolio.execute_exit(
                        trade, tranche["quantity"],
                        f"TP_50PCT_{tranche['level']}", current_price,
                    )
                    if result:
                        await self.notifier.notify_exit(result)

                # --- 4. Support break exit ---
                if Settings.risk.SUPPORT_BREAK_EXIT:
                    sr_break = self.sr_engine.check_support_break(symbol, current_price)
                    if sr_break and sr_break.get("broken"):
                        logger.warning(
                            "SUPPORT BREAK: %s broke below $%.4f (%.2f%%)",
                            symbol, sr_break["support_level"], sr_break["break_pct"],
                        )
                        result = await self.portfolio.execute_exit(
                            trade, trade.get("remaining_quantity", trade["quantity"]),
                            "SUPPORT_BREAK", current_price,
                        )
                        if result:
                            await self.notifier.notify_exit(result)
                            await self.notifier.send_alert(
                                f"⚠️ *Support Break Exit*\n"
                                f"{symbol} broke support at ${sr_break['support_level']:.4f}\n"
                                f"Exited at ${current_price:.4f}"
                            )
                            self._save_state()
                        continue

                # --- 5. Volume spike alert (informational) ---
                vol = ind.latest.get("volume", 0)
                vol_sma = ind.latest.get("volume_sma", 1)
                spike = VolumeSpikeDetector.detect_spike(vol, vol_sma)
                if spike["is_spike"] and spike["strength"] in ("STRONG", "EXTREME"):
                    pnl_pct = ((current_price / trade["entry_price"]) - 1) * 100
                    if pnl_pct < -1:
                        await self.notifier.send_alert(
                            f"📊 *Volume Spike Warning*\n"
                            f"{symbol}: {spike['ratio']:.1f}x volume ({spike['strength']})\n"
                            f"P&L: {pnl_pct:+.2f}% — monitor closely"
                        )

            except Exception as e:
                logger.error("Monitor error for %s: %s", symbol, e)

        # Breaking news alerts for held positions
        alerts = self.news.check_breaking_news(
            [t["symbol"] for t in self.portfolio.open_positions]
        )
        if alerts:
            await self.notifier.notify_news_alert(alerts)

    # ------------------------------------------------------------------
    # Scheduled Tasks Loop
    # ------------------------------------------------------------------

    async def _scheduled_tasks_loop(self):
        """Periodic tasks: regime, news, scan, rebalance, weekly report."""
        logger.info("Scheduled tasks loop started")
        while not self._shutdown:
            try:
                now = utc_now()
                current_ts = now.timestamp()

                # News update every 15 minutes
                news_interval = Settings.strategy.NEWS_CHECK_INTERVAL_MINUTES * 60
                if current_ts - self._last_news_time >= news_interval:
                    await self.news.update_all()
                    self._last_news_time = current_ts
                    logger.info("News updated: F&G=%d (%s), sentiment=%.2f",
                                self.news.fear_greed_value,
                                self.news.fear_greed_label,
                                self.news.get_sentiment_score())

                # Regime check every hour
                if current_ts - self._last_regime_time >= 3600:
                    old_regime = self.regime.current_regime
                    await self.regime.detect_regime()
                    self._last_regime_time = current_ts
                    if self.regime.current_regime != old_regime:
                        await self.notifier.notify_regime_change(
                            old_regime, self.regime.current_regime,
                        )

                # Full market scan every SCAN_INTERVAL_HOURS
                scan_interval = Settings.strategy.SCAN_INTERVAL_HOURS * 3600
                if current_ts - self._last_scan_time >= scan_interval:
                    candidates = await self.scanner.scan_universe()
                    new_watchlist = self.scanner.get_watchlist_symbols()
                    # Add open position symbols
                    for pos in self.portfolio.open_positions:
                        if pos["symbol"] not in new_watchlist:
                            new_watchlist.append(pos["symbol"])
                    # Seed indicators for new symbols
                    for sym in new_watchlist:
                        if sym not in self.indicator_cache:
                            await self._seed_indicators(sym)
                            # Compute S/R for new symbol
                            h = self.kline_history.get(sym, {}).get("1h", [])
                            if h:
                                self.sr_engine.compute_levels(sym, h)
                    self.watchlist = new_watchlist
                    self._last_scan_time = current_ts
                    # Restart kline stream with updated watchlist
                    if self.watchlist:
                        await self.exchange.start_kline_stream(
                            self.watchlist, "15m", self._on_kline_update,
                        )
                    self._save_state()
                    logger.info("Market scan complete: %d symbols", len(self.watchlist))

                # Refresh S/R levels every 2 hours
                if now.hour % 2 == 0 and now.minute < 2:
                    for symbol in self.watchlist:
                        history_1h = self.kline_history.get(symbol, {}).get("1h", [])
                        if history_1h:
                            self.sr_engine.compute_levels(symbol, history_1h)

                # Portfolio sync every 5 minutes
                if now.minute % 5 == 0 and now.second < 60:
                    await self.portfolio.sync_with_exchange()

                # Rebalance check (weekly)
                if now.weekday() == Settings.strategy.REBALANCE_DAY and now.hour == 0:
                    if self._last_rebalance_day != now.day:
                        actions = await self.portfolio.check_rebalancing()
                        if actions:
                            results = await self.portfolio.execute_rebalancing(actions)
                            logger.info("Rebalanced: %d actions", len(results))
                        self._last_rebalance_day = now.day

                # Smart DCA during extreme fear
                await self.portfolio.smart_dca()

                # HMM retrain (weekly)
                if now.weekday() == Settings.strategy.HMM_RETRAIN_DAY and now.hour == 3:
                    if self.regime.should_retrain():
                        await self.regime.train_hmm()

                # AI portfolio analysis (every 2 hours)
                if Settings.ai.ENABLED and now.hour % 2 == 0 and now.minute < 2:
                    ai_analysis = await self.ai.analyze_portfolio(
                        self.portfolio.get_status(),
                        self.portfolio.open_positions,
                        self.regime.current_regime,
                        self.news.fear_greed_value,
                        self.news.get_sentiment_score(),
                        self.portfolio.sector_exposure,
                    )
                    if ai_analysis:
                        logger.info("AI portfolio health: %s/10, risk: %s",
                                    ai_analysis.get("health_score", "?"),
                                    ai_analysis.get("risk_level", "?"))

                # Daily report at 00:00 UTC
                if now.hour == 0 and now.minute < 2:
                    await self.notifier.send_daily_report(
                        self.portfolio, self.regime, self.news, self.risk,
                    )

                # Weekly report
                if (now.weekday() == Settings.ops.WEEKLY_REPORT_DAY
                        and now.hour == Settings.ops.WEEKLY_REPORT_HOUR
                        and now.minute < 2):
                    if self._last_weekly_report_day != now.day:
                        await self.notifier.send_weekly_report(
                            self.portfolio, self.regime, self.news,
                            self.risk, self.calibrator, self.ai,
                        )
                        self._last_weekly_report_day = now.day

                # Drawdown check
                dd = self.risk.check_drawdown(self.portfolio.portfolio_value)
                if dd["action"] == "KILL_SWITCH":
                    logger.critical("KILL SWITCH ACTIVATED — drawdown %.2f%%",
                                    dd["drawdown_pct"] * 100)
                    await self.portfolio.liquidate_all("KILL_SWITCH")
                    await self.notifier.notify_drawdown_warning(dd)
                elif dd["action"] == "EMERGENCY_LIQUIDATE_50":
                    logger.warning("Emergency: liquidating 50%% — drawdown %.2f%%",
                                   dd["drawdown_pct"] * 100)
                    positions = list(self.portfolio.open_positions)
                    positions.sort(key=lambda p: p.get("confluence_score", 0))
                    for pos in positions[:len(positions) // 2]:
                        price = await self.exchange.get_price(pos["symbol"])
                        await self.portfolio.execute_exit(
                            pos, pos.get("remaining_quantity", pos["quantity"]),
                            "EMERGENCY_DRAWDOWN", price,
                        )
                    await self.notifier.notify_drawdown_warning(dd)
                elif dd["action"] in ("DEFENSIVE", "WARNING"):
                    await self.notifier.notify_drawdown_warning(dd)

                self._save_state()

            except Exception as e:
                logger.error("Scheduled task error: %s", e)

            await asyncio.sleep(60)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def _shutdown_handler(self):
        logger.info("Shutdown signal received...")
        self._shutdown = True
        self._save_state()
        await self.notifier.send_message("🔴 *Bot Shutting Down*\nState saved for restart recovery.")
        logger.info("Shutdown complete")


def main():
    bot = TradingBot()
    try:
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt — shutting down")
    except Exception as e:
        logger.critical("Fatal error: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
