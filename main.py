"""
Binance Spot Portfolio Trading Bot — Main Orchestrator

Features:
  - AI trading (Groq FREE — llama-3.3-70b + llama-3.1-8b)
  - 10% wallet rule, 2% SL, 3% TP, trailing stop, 50/50 partial TP
  - RSI + MACD + Bollinger + S/R + support break exit + volume spike
  - Multi-timeframe 5m + 15m + 1h
  - Daily loss limit, weekly report, news sentiment filter
  - Memory after restart, auto-restart watchdog
  - Diversified crypto portfolio
  - REAL-TIME ACTIVITY FEED to Telegram
"""

import os
import sys
import json
import signal
import asyncio
import time
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
        self.scanner = AssetScanner(self.exchange, self.regime, self.news)
        self.scorer = ConfluenceScorer(self.regime, self.news, self.db, self.sr_engine)
        self.signal_gen = SignalGenerator(self.scorer, self.regime, self.news, self.ai)

        # Calibration
        self.calibrator = SelfCalibrator(self.db, self.risk, self.regime)

        # Notifications
        self.notifier = TelegramNotifier(
            Settings.TELEGRAM_BOT_TOKEN,
            Settings.TELEGRAM_CHAT_ID,
        )

        # Watchdog
        self.watchdog = Watchdog(self.exchange, self.db)

        # Data caches — multi-timeframe
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

        # Activity feed control — avoid spamming Telegram
        self._cycle_count: int = 0
        self._last_activity_msg_time: float = 0
        self._ACTIVITY_COOLDOWN: int = 300  # Send activity summary every 5 min max

    async def start(self):
        logger.info("=" * 60)
        logger.info("BINANCE PORTFOLIO BOT STARTING")
        logger.info("Testnet: %s | AI: %s | Fast: %s | Strong: %s",
                     Settings.TESTNET, Settings.ai.ENABLED,
                     Settings.ai.FAST_MODEL, Settings.ai.STRONG_MODEL)
        logger.info("=" * 60)

        # Setup signal handlers (not supported on Windows outside Docker)
        try:
            loop = asyncio.get_event_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(
                    sig, lambda: asyncio.create_task(self._shutdown_handler()),
                )
        except NotImplementedError:
            logger.warning("Signal handlers not supported on this OS, using KeyboardInterrupt fallback")

        # Initialize all components
        self.db.init_db()
        await self.exchange.initialize()
        await self.notifier.send_message(
            "<b>INITIALIZING...</b>\n"
            "Connecting to Binance, loading news, training regime model..."
        )
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
            f"<b>BOT STARTED</b>\n\n"
            f"<b>Mode:</b> <code>{'TESTNET' if Settings.TESTNET else 'LIVE'}</code>\n"
            f"<b>AI Fast:</b> <code>{Settings.ai.FAST_MODEL}</code>\n"
            f"<b>AI Strong:</b> <code>{Settings.ai.STRONG_MODEL}</code>\n"
            f"<b>Portfolio:</b> <code>${self.portfolio.portfolio_value:.2f}</code>\n"
            f"<b>Regime:</b> <code>{self.regime.current_regime}</code>\n"
            f"<b>Fear/Greed:</b> <code>{self.news.fear_greed_value} ({self.news.fear_greed_label})</code>\n"
            f"<b>Open Positions:</b> <code>{len(self.portfolio.open_positions)}</code>\n"
            f"<b>Veto Power:</b> <code>{Settings.ai.VETO_POWER}</code>\n"
            f"<b>Confluence Threshold:</b> <code>{Settings.strategy.CONFLUENCE_SCORE_THRESHOLD}/100</code>\n"
            f"<b>Scan Interval:</b> <code>{Settings.strategy.SCAN_INTERVAL_SECONDS}s</code>\n"
            f"<b>Timeframes:</b> <code>{', '.join(Settings.strategy.TIMEFRAMES)}</code>"
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
            self._activity_feed_loop(),
            self.notifier.start_polling(),
        )

    # ------------------------------------------------------------------
    # State Persistence (Memory After Restart)
    # ------------------------------------------------------------------

    async def _restore_state(self):
        """Restore bot state from database and state file."""
        try:
            saved_watchlist = self.db.load_state("watchlist")
            if saved_watchlist:
                self.watchlist = json.loads(saved_watchlist)
                logger.info("Restored watchlist: %d symbols", len(self.watchlist))

            last_scan = self.db.load_state("last_scan_time")
            if last_scan:
                self._last_scan_time = float(last_scan)

            last_news = self.db.load_state("last_news_time")
            if last_news:
                self._last_news_time = float(last_news)

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
        await self.notifier.send_message(
            "<b>SCANNING MARKET...</b>\n"
            f"Filtering {Settings.strategy.MAX_WATCHLIST_SIZE} top assets by volume, "
            f"min 24h vol: ${Settings.strategy.MIN_24H_VOLUME:,.0f}"
        )

        candidates = await self.scanner.scan_universe()
        self.watchlist = self.scanner.get_watchlist_symbols()

        # Add symbols from open positions
        for pos in self.portfolio.open_positions:
            if pos["symbol"] not in self.watchlist:
                self.watchlist.append(pos["symbol"])

        logger.info("Watchlist: %d symbols — %s", len(self.watchlist),
                     ", ".join(self.watchlist[:10]))

        await self.notifier.send_message(
            f"<b>WATCHLIST BUILT</b>\n"
            f"Tracking <code>{len(self.watchlist)}</code> assets\n"
            f"Top 10: <code>{', '.join(self.watchlist[:10])}</code>\n\n"
            f"Seeding indicators for all timeframes..."
        )

        # Seed indicators for all timeframes (batched to avoid rate limits)
        seeded = 0
        batch_size = 5
        for i in range(0, len(self.watchlist), batch_size):
            batch = self.watchlist[i:i + batch_size]
            for symbol in batch:
                await self._seed_indicators(symbol)
                seeded += 1
            logger.info("Seeded %d/%d symbols...", seeded, len(self.watchlist))
            await asyncio.sleep(2)  # 2s pause between batches to respect rate limits

        # Compute S/R levels for all watchlist symbols
        for symbol in self.watchlist:
            tf_key = Settings.strategy.PRIMARY_TIMEFRAME
            history = self.kline_history.get(symbol, {}).get(tf_key, [])
            if history:
                self.sr_engine.compute_levels(symbol, history)

        # Start WebSocket for entry timeframe klines
        entry_tf = Settings.strategy.ENTRY_TIMEFRAME
        if self.watchlist:
            await self.exchange.start_kline_stream(
                self.watchlist, entry_tf, self._on_kline_update,
            )

        self._save_state()

        await self.notifier.send_message(
            f"<b>READY TO TRADE</b>\n"
            f"Indicators seeded for {seeded} assets\n"
            f"WebSocket streaming {entry_tf} candles\n"
            f"S/R levels computed\n\n"
            f"<i>Bot is now actively scanning every {Settings.strategy.SCAN_INTERVAL_SECONDS}s...</i>"
        )
        logger.info("Initial scan complete")

    async def _seed_indicators(self, symbol: str):
        """Seed indicators for configured timeframes."""
        self.indicator_cache[symbol] = {}
        self.kline_history[symbol] = {}

        for tf in Settings.strategy.TIMEFRAMES:
            try:
                klines = await self.exchange.get_klines(symbol, tf, 200)
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
        """Process real-time kline updates from WebSocket."""
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

            entry_tf = Settings.strategy.ENTRY_TIMEFRAME
            ind = self.indicator_cache[symbol].get(entry_tf)
            if ind is None:
                return

            if is_closed:
                ind.update_candle(candle)
                history = self.kline_history.get(symbol, {}).get(entry_tf, [])
                history.append(candle)
                if len(history) > 200:
                    history.pop(0)
                self.kline_history.setdefault(symbol, {})[entry_tf] = history
                self.watchdog.beat()
            else:
                ind.update_tick(candle["close"])

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
        self._cycle_count += 1

        if self.risk.kill_switch_active:
            logger.info("Cycle %d: Kill switch active — skipping", self._cycle_count)
            prev = getattr(self, "_last_cycle_stats", {})
            self._last_cycle_stats = {
                **prev,
                "cycle": self._cycle_count,
                "regime": self.regime.current_regime,
                "portfolio_value": self.portfolio.portfolio_value,
                "positions": len(self.portfolio.open_positions),
            }
            return
        if self.risk.is_paused:
            logger.info("Cycle %d: Trading paused — skipping", self._cycle_count)
            prev = getattr(self, "_last_cycle_stats", {})
            self._last_cycle_stats = {
                **prev,
                "cycle": self._cycle_count,
                "regime": self.regime.current_regime,
                "portfolio_value": self.portfolio.portfolio_value,
                "positions": len(self.portfolio.open_positions),
            }
            return

        regime_params = self.regime.get_regime_params()
        # Regime NEVER blocks entries — it only adjusts position sizing
        # entries_allowed is always True in aggressive mode

        # Refresh higher timeframes
        await self._refresh_higher_timeframes()

        portfolio_context = self.portfolio.get_status()
        held_symbols = {p["symbol"] for p in self.portfolio.open_positions}

        # Track cycle stats for activity feed
        symbols_scanned = 0
        signals_found = 0
        trades_executed = 0
        top_scores = []  # (symbol, score) for top candidates

        for symbol in self.watchlist:
            if symbol in held_symbols:
                continue  # Already holding

            entry_tf = Settings.strategy.ENTRY_TIMEFRAME
            primary_tf = Settings.strategy.PRIMARY_TIMEFRAME
            trend_tf = Settings.strategy.TREND_TIMEFRAME

            ind_entry = self.indicator_cache.get(symbol, {}).get(entry_tf, IndicatorSet()).latest
            ind_primary = self.indicator_cache.get(symbol, {}).get(primary_tf, IndicatorSet()).latest
            ind_trend = self.indicator_cache.get(symbol, {}).get(trend_tf, IndicatorSet()).latest
            history_primary = self.kline_history.get(symbol, {}).get(primary_tf, [])

            if not ind_primary:
                continue

            symbols_scanned += 1

            signal = await self.signal_gen.evaluate(
                symbol, ind_entry, ind_primary, ind_trend, history_primary,
                portfolio_context,
                cycle_id=self._cycle_count,
            )

            if signal:
                signals_found += 1
                score = signal.get("confluence_score", 0)
                top_scores.append((symbol, score))

                trade = await self.portfolio.execute_entry(signal)
                if trade:
                    trades_executed += 1
                    await self.notifier.notify_entry(trade)
                    self.calibrator.check_and_calibrate()
                    self._save_state()

        # Log cycle summary
        logger.info(
            "Cycle %d: scanned=%d, signals=%d, trades=%d, positions=%d",
            self._cycle_count, symbols_scanned, signals_found,
            trades_executed, len(self.portfolio.open_positions),
        )

        # Store cycle stats for activity feed
        self._last_cycle_stats = {
            "cycle": self._cycle_count,
            "scanned": symbols_scanned,
            "signals": signals_found,
            "trades": trades_executed,
            "positions": len(self.portfolio.open_positions),
            "top_scores": sorted(top_scores, key=lambda x: x[1], reverse=True)[:5],
            "regime": self.regime.current_regime,
            "portfolio_value": self.portfolio.portfolio_value,
        }

    async def _refresh_higher_timeframes(self):
        """Refresh primary and trend timeframe indicators from REST API.
        Only refreshes every 5 minutes to avoid rate limits.
        Processes in batches of 10 with 1s delay between batches.
        """
        now = time.time()
        if now - getattr(self, '_last_tf_refresh', 0) < 300:  # 5 min cooldown
            return
        self._last_tf_refresh = now

        primary_tf = Settings.strategy.PRIMARY_TIMEFRAME
        trend_tf = Settings.strategy.TREND_TIMEFRAME

        batch_size = 10
        for i in range(0, len(self.watchlist), batch_size):
            batch = self.watchlist[i:i + batch_size]
            for symbol in batch:
                for tf in [primary_tf, trend_tf]:
                    try:
                        klines = await self.exchange.get_klines(symbol, tf, 5)
                        if not klines:
                            continue
                        ind_set = self.indicator_cache.get(symbol, {}).get(tf)
                        if ind_set is None:
                            continue
                        history = self.kline_history.get(symbol, {}).get(tf, [])
                        for k in klines:
                            if not history or k.get("open_time", 0) != history[-1].get("open_time", -1):
                                ind_set.update_candle(k)
                                history.append(k)
                                if len(history) > 200:
                                    history.pop(0)
                        self.kline_history.setdefault(symbol, {})[tf] = history
                    except Exception as e:
                        logger.debug("Refresh %s %s failed: %s", symbol, tf, e)
            await asyncio.sleep(1)  # Rate limit: 1s between batches of 10

    # ------------------------------------------------------------------
    # Activity Feed Loop — Real-time Telegram updates
    # ------------------------------------------------------------------

    async def _activity_feed_loop(self):
        """Send periodic activity summaries to Telegram so user knows what bot is doing."""
        logger.info("Activity feed loop started (interval=%ds)", self._ACTIVITY_COOLDOWN)
        await asyncio.sleep(60)  # Wait for first cycle to complete

        while not self._shutdown:
            try:
                now = time.time()
                if now - self._last_activity_msg_time >= self._ACTIVITY_COOLDOWN:
                    await self._send_activity_summary()
                    self._last_activity_msg_time = now
            except Exception as e:
                logger.debug("Activity feed error: %s", e)
            await asyncio.sleep(30)

    async def _send_activity_summary(self):
        """Build and send a concise activity summary."""
        stats = getattr(self, "_last_cycle_stats", None)
        if not stats:
            return

        # Build position summary with live P&L
        pos_lines = []
        for p in self.portfolio.open_positions[:5]:
            try:
                price = await self.exchange.get_price(p["symbol"])
                pnl_pct = ((price / p["entry_price"]) - 1) * 100
                arrow = "▲" if pnl_pct >= 0 else "▼"
                pos_lines.append(
                    f"  {arrow} {p['symbol']}: {pnl_pct:+.2f}%"
                )
            except Exception:
                pos_lines.append(f"  {p['symbol']}: (price unavailable)")

        pos_text = "\n".join(pos_lines) if pos_lines else "  No open positions"

        # Build top candidates
        top_text = ""
        if stats.get("top_scores"):
            top_lines = [f"  {sym}: {score}/100" for sym, score in stats["top_scores"][:3]]
            top_text = "\n<b>Top Candidates:</b>\n" + "\n".join(top_lines)

        # AI status
        ai_status = self.ai.get_status()
        ai_calls = ai_status.get("daily_calls", 0)

        portfolio_val = stats.get("portfolio_value", self.portfolio.portfolio_value)
        cash = self.portfolio.cash_available

        msg = (
            f"<b>ACTIVITY UPDATE</b>\n"
            f"Cycle: <code>{stats['cycle']}</code>\n"
            f"Scanned: <code>{stats.get('scanned', '—')}</code> assets\n"
            f"Signals Found: <code>{stats.get('signals', '—')}</code>\n"
            f"Trades Executed: <code>{stats.get('trades', '—')}</code>\n"
            f"Regime: <code>{stats['regime']}</code>\n"
            f"Portfolio: <code>${portfolio_val:.2f}</code> "
            f"(Cash: <code>${cash:.2f}</code>)\n"
            f"AI Calls Today: <code>{ai_calls}</code>\n\n"
            f"<b>Open Positions ({len(self.portfolio.open_positions)}):</b>\n"
            f"<code>{pos_text}</code>"
        )

        if top_text:
            msg += f"\n{top_text}"

        # Status hint with actionable guidance
        if self.risk.kill_switch_active:
            msg += (
                "\n\n⛔ <b>Kill switch ACTIVE</b> — no new trades\n"
                "<i>Send /resume to unlock trading</i>"
            )
        elif self.risk.is_paused:
            msg += "\n\n⏸ <i>Trading PAUSED by user — send /resume to continue</i>"
        elif portfolio_val < 50:
            msg += (
                f"\n\n⚠️ <i>Portfolio ${portfolio_val:.2f} is very small. "
                f"Add at least $50–$100 USDT to enable reliable trading "
                f"(Binance minimum notional is ~$10/trade).</i>"
            )
        elif stats.get("signals", 0) == 0:
            msg += "\n\n🔍 <i>Scanning... no signals above threshold yet</i>"
        else:
            msg += f"\n\n✅ <i>Active — next scan in {Settings.strategy.SCAN_INTERVAL_SECONDS}s</i>"

        await self.notifier.send_message(msg)

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

                primary_tf = Settings.strategy.PRIMARY_TIMEFRAME
                ind = self.indicator_cache.get(symbol, {}).get(primary_tf, IndicatorSet())
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
                                f"<b>Support Break Exit</b>\n"
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
                            f"<b>Volume Spike Warning</b>\n"
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

                # News update every N minutes
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
                    await self.notifier.send_message(
                        f"<b>MARKET RESCAN</b>\n"
                        f"Refreshing watchlist (every {Settings.strategy.SCAN_INTERVAL_HOURS}h)..."
                    )
                    candidates = await self.scanner.scan_universe()
                    new_watchlist = self.scanner.get_watchlist_symbols()
                    # Add open position symbols
                    for pos in self.portfolio.open_positions:
                        if pos["symbol"] not in new_watchlist:
                            new_watchlist.append(pos["symbol"])
                    # Seed indicators for new symbols
                    new_count = 0
                    for sym in new_watchlist:
                        if sym not in self.indicator_cache:
                            await self._seed_indicators(sym)
                            h = self.kline_history.get(sym, {}).get(
                                Settings.strategy.PRIMARY_TIMEFRAME, [])
                            if h:
                                self.sr_engine.compute_levels(sym, h)
                            new_count += 1
                    self.watchlist = new_watchlist
                    self._last_scan_time = current_ts
                    # Restart kline stream with updated watchlist
                    entry_tf = Settings.strategy.ENTRY_TIMEFRAME
                    if self.watchlist:
                        await self.exchange.start_kline_stream(
                            self.watchlist, entry_tf, self._on_kline_update,
                        )
                    self._save_state()
                    await self.notifier.send_message(
                        f"<b>RESCAN COMPLETE</b>\n"
                        f"Watchlist: <code>{len(self.watchlist)}</code> assets\n"
                        f"New additions: <code>{new_count}</code>\n"
                        f"Top: <code>{', '.join(self.watchlist[:5])}</code>"
                    )
                    logger.info("Market scan complete: %d symbols", len(self.watchlist))

                # Refresh S/R levels every 2 hours
                if now.hour % 2 == 0 and now.minute < 2:
                    for symbol in self.watchlist:
                        history = self.kline_history.get(symbol, {}).get(
                            Settings.strategy.PRIMARY_TIMEFRAME, [])
                        if history:
                            self.sr_engine.compute_levels(symbol, history)

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
                        # Notify user of AI portfolio assessment
                        await self.notifier.send_message(
                            f"<b>AI PORTFOLIO REVIEW</b>\n"
                            f"Health: <code>{ai_analysis.get('health_score', '?')}/10</code>\n"
                            f"Risk: <code>{ai_analysis.get('risk_level', '?')}</code>\n"
                            f"Cash: <code>{ai_analysis.get('cash_recommendation', '?')}</code>\n"
                            f"Assessment: {ai_analysis.get('reasoning', 'N/A')[:300]}"
                        )

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
        await self.notifier.send_message(
            "<b>BOT SHUTTING DOWN</b>\n"
            "State saved for restart recovery."
        )
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
