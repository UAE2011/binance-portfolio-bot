"""
Binance Portfolio Bot — Playbook-optimized trading system.

Architecture:
  Regime Detection (HMM + trend + sentiment)
       ↓
  Operating Mode (AGGRESSIVE / BALANCED / DEFENSIVE / CAPITAL_PRESERVATION)
       ↓
  Asset Scanner + Indicator Engine
       ↓
  Confluence Scorer (100-pt playbook model)
       ↓
  AI Veto Layer (Gemini 2.5 Flash, top-5 signals only)
       ↓
  Risk Manager (Quarter Kelly + Anti-martingale + Portfolio Heat)
       ↓
  Execution + OCO Orders + Partial TP
       ↓
  Active Position Manager (Chandelier Exit trailing + circuit breakers)

Start: docker-compose up -d --build
Telegram: /help for all commands
"""
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

from config.settings import Settings
from src.database import Database
from src.exchange import Exchange
from src.indicators import IndicatorSet
from src.regime import RegimeDetector
from src.risk_manager import RiskManager
from src.portfolio import PortfolioManager
from src.strategy import (
    ConfluenceScorer, SignalGenerator, SupportResistanceEngine, VolumeSpikeDetector
)
from src.news_intelligence import NewsIntelligence
from src.notifier import TelegramNotifier, build_command_handlers
from src.scanner import AssetScanner
from src.calibrator import Calibrator
from src.watchdog import Watchdog
from src.ai_advisor import AIAdvisor
from src.utils import setup_logging, utc_now

logger = setup_logging()

# ─── State tracking ──────────────────────────────────────────────────────────

class BotState:
    def __init__(self):
        self.cycle_id: int = 0
        self.last_regime_check: datetime = None
        self.last_news_update: datetime = None
        self.last_rebalance_check: datetime = None
        self.last_dca_attempt: datetime = None
        self.prev_regime: str = ""
        self.prev_mode: str = ""
        self.indicator_sets: dict = {}   # symbol → IndicatorSet (primary)
        self.indicator_sets_entry: dict = {}  # symbol → IndicatorSet (entry tf)
        self.indicator_sets_trend: dict = {}  # symbol → IndicatorSet (trend tf)
        self.kline_history: dict = {}    # symbol → list of last N candles
        self.running: bool = True
        self.initialized: bool = False

    def save(self, path: str = "data/bot_state.json"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "w") as f:
                json.dump({
                    "cycle_id": self.cycle_id,
                    "prev_regime": self.prev_regime,
                    "prev_mode": self.prev_mode,
                    "timestamp": utc_now().isoformat(),
                }, f)
        except Exception:
            pass

    def load(self, path: str = "data/bot_state.json"):
        try:
            if Path(path).exists():
                with open(path) as f:
                    data = json.load(f)
                self.cycle_id = data.get("cycle_id", 0)
                self.prev_regime = data.get("prev_regime", "")
                self.prev_mode = data.get("prev_mode", "")
        except Exception:
            pass


# ─── Initialization ──────────────────────────────────────────────────────────

async def initialize_indicators(bot_state: BotState, scanner: AssetScanner,
                                exchange: Exchange):
    """Seed indicators for watchlist + core assets."""
    watchlist = await scanner.get_watchlist()
    core = await scanner.get_core_assets()
    all_symbols = list({a["symbol"] for a in watchlist + core})

    primary_tf = Settings.strategy.PRIMARY_TIMEFRAME   # 1h
    entry_tf = Settings.strategy.ENTRY_TIMEFRAME       # 15m
    trend_tf = Settings.strategy.TREND_TIMEFRAME       # 4h

    seeds = 0
    for symbol in all_symbols[:60]:  # seed top 60
        try:
            for tf, store in [(primary_tf, bot_state.indicator_sets),
                              (entry_tf, bot_state.indicator_sets_entry),
                              (trend_tf, bot_state.indicator_sets_trend)]:
                if symbol not in store:
                    store[symbol] = IndicatorSet()
                klines = await exchange.get_klines(symbol, tf, limit=250)
                if klines:
                    store[symbol].seed_from_klines(klines)

            # Keep primary kline history
            klines_primary = await exchange.get_klines(symbol, primary_tf, limit=200)
            bot_state.kline_history[symbol] = klines_primary
            seeds += 1
        except Exception as e:
            logger.debug("Seed error %s: %s", symbol, e)

    logger.info("Seeded indicators for %d symbols", seeds)


# ─── Active Position Management ──────────────────────────────────────────────

async def manage_open_positions(portfolio: PortfolioManager, risk: RiskManager,
                                notifier: TelegramNotifier, calibrator: Calibrator,
                                bot_state: BotState, sr_engine: SupportResistanceEngine):
    """
    For each open position:
    1. Update trailing stop (Chandelier Exit)
    2. Check partial TP (50% at 6% gain)
    3. Check stop-loss hit
    4. Check support break
    5. Execute exits as needed
    """
    if not portfolio.open_positions:
        return

    exits_executed = []

    for trade in list(portfolio.open_positions):
        symbol = trade["symbol"]
        try:
            current_price = await portfolio.exchange.get_price(symbol)
        except Exception:
            continue

        # Update highest price
        trade["highest_price"] = max(trade.get("highest_price", trade["entry_price"]),
                                     current_price)

        # Get current ATR
        ind = bot_state.indicator_sets.get(symbol)
        current_atr = ind.latest.get("atr", 0) if ind and ind.latest else trade.get("atr", 0)

        # 1. Update trailing stop
        updates = risk.update_trailing_stop(trade, current_price, current_atr)
        new_stop = updates.get("stop_loss", trade["stop_loss"])
        new_highest = updates.get("highest_price", trade.get("highest_price"))

        if (new_stop != trade.get("stop_loss")
                or new_highest != trade.get("highest_price")):
            portfolio.db.update_trade(trade["id"], {
                "stop_loss": new_stop,
                "highest_price": new_highest,
            })
            trade["stop_loss"] = new_stop
            trade["highest_price"] = new_highest

        # 2. Check partial TP
        partial_exits = risk.check_take_profit_tranches(trade, current_price, current_atr)
        for pe in partial_exits:
            result = await portfolio.execute_exit(
                trade, pe["quantity"], pe["reason"], current_price
            )
            if result:
                exits_executed.append(result)
                await notifier.notify_exit(result)
                trade["tranche_exits"] = portfolio.db.get_trade_by_id(trade["id"])["tranche_exits"]

        # 3. Stop-loss check
        current_stop = trade.get("stop_loss", 0)
        if current_stop > 0 and current_price <= current_stop:
            remaining = trade.get("remaining_quantity", trade["quantity"])
            result = await portfolio.execute_exit(
                trade, remaining,
                f"STOP_LOSS@{current_stop:.4f}", current_price
            )
            if result:
                exits_executed.append(result)
                calibrator.on_trade_closed()
                await notifier.notify_exit(result)
            continue

        # 4. Take-profit (full)
        tp = trade.get("take_profit", 0)
        if tp > 0 and current_price >= tp:
            remaining = trade.get("remaining_quantity", trade["quantity"])
            result = await portfolio.execute_exit(
                trade, remaining,
                f"TAKE_PROFIT@{tp:.4f}", current_price
            )
            if result:
                exits_executed.append(result)
                calibrator.on_trade_closed()
                await notifier.notify_exit(result)
            continue

        # 5. Support break exit
        if Settings.risk.SUPPORT_BREAK_EXIT:
            sb = sr_engine.check_support_break(symbol, current_price)
            if sb and sb.get("broken") and sb.get("touches", 0) >= 2:
                remaining = trade.get("remaining_quantity", trade["quantity"])
                result = await portfolio.execute_exit(
                    trade, remaining,
                    f"SUPPORT_BREAK@{sb['support_level']:.4f}", current_price
                )
                if result:
                    exits_executed.append(result)
                    calibrator.on_trade_closed()
                    await notifier.notify_exit(result)

        # 6. Breaking news exit
        breaking = portfolio.news.check_breaking_news([symbol])
        for alert in breaking:
            if alert.get("sentiment", 0) < -0.5:
                remaining = trade.get("remaining_quantity", trade["quantity"])
                result = await portfolio.execute_exit(
                    trade, remaining,
                    f"NEWS_ALERT:{alert.get('title', '')[:30]}", current_price
                )
                if result:
                    exits_executed.append(result)
                    calibrator.on_trade_closed()
                    await notifier.notify_exit(result)

    return exits_executed


# ─── Signal Scanning ─────────────────────────────────────────────────────────

async def scan_for_signals(scanner: AssetScanner, bot_state: BotState,
                           exchange: Exchange, signal_gen: SignalGenerator,
                           sr_engine: SupportResistanceEngine) -> list:
    """Fetch latest candles, update indicators, generate signals."""
    watchlist = await scanner.get_watchlist()
    core = await scanner.get_core_assets()

    # Deduplicate, core assets get priority
    seen = set()
    all_assets = []
    for a in core + watchlist:
        if a["symbol"] not in seen:
            seen.add(a["symbol"])
            all_assets.append(a)

    primary_tf = Settings.strategy.PRIMARY_TIMEFRAME
    entry_tf = Settings.strategy.ENTRY_TIMEFRAME
    trend_tf = Settings.strategy.TREND_TIMEFRAME

    signals = []
    bot_state.cycle_id += 1

    for asset in all_assets:
        symbol = asset["symbol"]
        try:
            # Fetch latest candle for each timeframe
            for tf, store in [(primary_tf, bot_state.indicator_sets),
                              (entry_tf, bot_state.indicator_sets_entry),
                              (trend_tf, bot_state.indicator_sets_trend)]:
                if symbol not in store:
                    store[symbol] = IndicatorSet()
                latest_candles = await exchange.get_klines(symbol, tf, limit=3)
                if latest_candles:
                    for candle in latest_candles[-2:]:
                        store[symbol].update_candle(candle)

            # Update kline history (rolling 200)
            latest_primary = await exchange.get_klines(symbol, primary_tf, limit=5)
            if latest_primary:
                existing = bot_state.kline_history.get(symbol, [])
                for c in latest_primary:
                    if not existing or c["open_time"] != existing[-1]["open_time"]:
                        existing.append(c)
                bot_state.kline_history[symbol] = existing[-200:]

            # Compute S/R levels (every 10 cycles)
            if bot_state.cycle_id % 10 == 1:
                history = bot_state.kline_history.get(symbol, [])
                if len(history) >= 50:
                    sr_engine.compute_levels(symbol, history)

            ind_primary = bot_state.indicator_sets[symbol].latest
            ind_entry = bot_state.indicator_sets_entry.get(symbol, bot_state.indicator_sets[symbol]).latest
            ind_trend = bot_state.indicator_sets_trend.get(symbol, bot_state.indicator_sets[symbol]).latest

            if not ind_primary:
                continue

            history = bot_state.kline_history.get(symbol, [])

            signal = await signal_gen.evaluate(
                symbol=symbol,
                indicators_entry=ind_entry or ind_primary,
                indicators_primary=ind_primary,
                indicators_trend=ind_trend or ind_primary,
                kline_history=history,
                cycle_id=bot_state.cycle_id,
            )
            if signal:
                signals.append(signal)

        except Exception as e:
            logger.debug("Scan error %s: %s", symbol, e)

    # Sort by confluence score desc
    signals.sort(key=lambda s: s["confluence_score"], reverse=True)
    logger.info("Cycle %d: %d assets scanned, %d signals above threshold",
                bot_state.cycle_id, len(all_assets), len(signals))
    return signals


# ─── Circuit Breaker Enforcement ─────────────────────────────────────────────

async def enforce_circuit_breakers(portfolio: PortfolioManager, risk: RiskManager,
                                   notifier: TelegramNotifier):
    dd = risk.check_drawdown(portfolio.portfolio_value)
    action = dd.get("action", "NONE")

    if action == "NONE" or action == "WARNING":
        return

    await notifier.notify_circuit_breaker(dd)

    if action == "DEFENSIVE":
        risk.position_size_modifier = 0.5

    elif action == "EMERGENCY_LIQUIDATE_50":
        positions_to_close = portfolio.open_positions[:len(portfolio.open_positions)//2]
        for trade in positions_to_close:
            price = await portfolio.exchange.get_price(trade["symbol"])
            result = await portfolio.execute_exit(
                trade, trade.get("remaining_quantity", trade["quantity"]),
                "CIRCUIT_BREAKER_50PCT", price,
            )
            if result:
                await notifier.notify_exit(result)

    elif action == "CAPITAL_PRESERVATION":
        results = await portfolio.liquidate_all("CIRCUIT_BREAKER_CAPITAL_PRESERVATION")
        for result in results:
            await notifier.notify_exit(result)
        risk.position_size_modifier = 0.15

    elif action == "KILL_SWITCH":
        results = await portfolio.liquidate_all("KILL_SWITCH")
        for result in results:
            await notifier.notify_exit(result)
        risk.kill_switch_active = True
        await notifier.send_alert(
            f"🚨 KILL SWITCH ACTIVATED\n"
            f"Drawdown: {dd['drawdown_pct']:.1%}\n"
            f"All positions closed. Send /resume to restart."
        )


# ─── Scheduled Tasks ─────────────────────────────────────────────────────────

async def scheduled_tasks(portfolio: PortfolioManager, regime: RegimeDetector,
                          news: NewsIntelligence, risk: RiskManager,
                          notifier: TelegramNotifier, bot_state: BotState):
    now = utc_now()

    # News update every 15 min
    news_interval = timedelta(minutes=Settings.strategy.NEWS_CHECK_INTERVAL_MINUTES)
    if (bot_state.last_news_update is None
            or (now - bot_state.last_news_update) >= news_interval):
        await news.update_all()
        regime.update_sentiment(
            fear_greed=news.fear_greed_value,
            news_sentiment=news.get_sentiment_score(),
            btc_dominance=news.btc_dominance / 100,
            altcoin_season_index=news.altcoin_season_index,
        )
        bot_state.last_news_update = now

    # Regime re-detection every hour
    if (bot_state.last_regime_check is None
            or (now - bot_state.last_regime_check) >= timedelta(hours=1)):
        old_regime = regime.current_regime
        old_mode = regime.current_mode
        await regime.detect_regime()
        bot_state.last_regime_check = now

        if regime.current_regime != old_regime:
            await notifier.notify_regime_change(old_regime, regime.current_regime)
            bot_state.prev_regime = old_regime
        if regime.current_mode != old_mode:
            await notifier.notify_mode_change(old_mode, regime.current_mode)
            bot_state.prev_mode = old_mode

    # HMM retrain weekly
    if await regime.should_retrain():
        await regime.train_hmm()

    # Rebalance check every 4 hours
    if (bot_state.last_rebalance_check is None
            or (now - bot_state.last_rebalance_check) >= timedelta(hours=4)):
        actions = await portfolio.check_rebalancing()
        if actions:
            results = await portfolio.execute_rebalancing(actions)
            for r in results:
                await notifier.notify_exit(r)
        bot_state.last_rebalance_check = now

    # Smart DCA during extreme fear (every 6 hours)
    if (bot_state.last_dca_attempt is None
            or (now - bot_state.last_dca_attempt) >= timedelta(hours=6)):
        if news.is_extreme_fear():
            await portfolio.smart_dca()
        bot_state.last_dca_attempt = now

    # Daily report at UTC midnight
    if now.hour == 0 and now.minute < 2:
        await notifier.send_daily_report(portfolio, regime, news, risk)

    # Weekly report
    weekly_day = Settings.ops.WEEKLY_REPORT_DAY
    weekly_hour = Settings.ops.WEEKLY_REPORT_HOUR
    if now.weekday() == weekly_day and now.hour == weekly_hour and now.minute < 2:
        pass  # weekly report — needs calibrator, ai refs (handled in main loop)

    bot_state.save()


# ─── Main Loop ───────────────────────────────────────────────────────────────

async def main():
    logger.info("=" * 60)
    logger.info("Binance Portfolio Bot — Playbook Edition")
    logger.info("=" * 60)

    if Settings.binance.USE_TESTNET:
        logger.warning("⚠️  TESTNET MODE — No real funds at risk")
    else:
        logger.info("🔴 LIVE TRADING MODE")

    # Initialize components
    db = Database(Settings.ops.DB_PATH)
    exchange = Exchange(
        Settings.binance.API_KEY,
        Settings.binance.API_SECRET,
        Settings.binance.USE_TESTNET,
    )
    news = NewsIntelligence()
    regime = RegimeDetector(exchange, db, Settings.strategy.HMM_LOOKBACK_DAYS)
    risk = RiskManager(db, regime)
    risk.set_news_intel(news)
    portfolio = PortfolioManager(exchange, db, regime, risk, news)
    ai = AIAdvisor(
        Settings.ai.API_KEY,
        Settings.ai.BASE_URL,
        Settings.ai.FAST_MODEL,
        Settings.ai.STRONG_MODEL,
    )
    sr_engine = SupportResistanceEngine()
    scorer = ConfluenceScorer(regime, news, db, sr_engine)
    signal_gen = SignalGenerator(scorer, regime, news, ai)
    scanner = AssetScanner(exchange, regime, news)
    notifier = TelegramNotifier(Settings.telegram.BOT_TOKEN, Settings.telegram.CHAT_ID)
    calibrator = Calibrator(db, risk, regime, notifier)
    watchdog = Watchdog(notifier)
    bot_state = BotState()
    bot_state.load()

    # Wire command handlers
    handlers = build_command_handlers(
        portfolio, risk, calibrator, watchdog, regime, news, ai
    )
    notifier.register_commands(handlers)

    # Initialization sequence
    logger.info("Initializing regime detector...")
    await regime.initialize()

    logger.info("Updating news & sentiment...")
    await news.update_all()
    regime.update_sentiment(
        fear_greed=news.fear_greed_value,
        news_sentiment=news.get_sentiment_score(),
        btc_dominance=news.btc_dominance / 100,
        altcoin_season_index=news.altcoin_season_index,
    )

    logger.info("Syncing portfolio...")
    await portfolio.sync_with_exchange()

    logger.info("Loading risk history...")
    risk.initialize_from_history()

    logger.info("Seeding indicators...")
    await initialize_indicators(bot_state, scanner, exchange)

    bot_state.initialized = True
    bot_state.last_regime_check = utc_now()
    bot_state.last_news_update = utc_now()

    # Startup notification
    mode_e = regime.get_mode_emoji()
    startup_msg = (
        f"<b>🤖 Bot Online</b> {'🔴 LIVE' if not Settings.binance.USE_TESTNET else '🟡 TESTNET'}\n\n"
        f"Portfolio: <code>${portfolio.portfolio_value:.2f}</code>\n"
        f"Mode: {mode_e} <code>{regime.current_mode}</code>\n"
        f"Regime: <code>{regime.current_regime}</code>\n"
        f"F&G: <code>{news.fear_greed_value} ({news.fear_greed_label})</code>\n"
        f"BTC Dom: <code>{news.btc_dominance:.1f}%</code>\n"
        f"Kill Switch: <code>{'ON' if risk.kill_switch_active else 'OFF'}</code>\n\n"
        f"<i>Send /help for all commands</i>"
    )
    await notifier.send_message(startup_msg)

    # Start Telegram polling
    poll_task = asyncio.create_task(notifier.start_polling())

    logger.info("✅ Bot initialized. Starting main loop...")
    logger.info("Mode: %s | Regime: %s | Portfolio: $%.2f",
                regime.current_mode, regime.current_regime, portfolio.portfolio_value)

    scan_interval = Settings.strategy.SCAN_INTERVAL_SECONDS

    while bot_state.running:
        try:
            cycle_start = utc_now()
            watchdog.heartbeat()

            # Sync portfolio state
            await portfolio.sync_with_exchange()

            # Circuit breaker check
            await enforce_circuit_breakers(portfolio, risk, notifier)

            # Manage open positions
            await manage_open_positions(
                portfolio, risk, notifier, calibrator, bot_state, sr_engine
            )

            # Scheduled tasks
            await scheduled_tasks(portfolio, regime, news, risk, notifier, bot_state)

            # Don't scan for new signals if paused, kill switch active,
            # or in capital preservation mode
            if (not risk.kill_switch_active and not risk.is_paused
                    and regime.get_regime_params().get("entries_allowed", True)):

                signals = await scan_for_signals(
                    scanner, bot_state, exchange, signal_gen, sr_engine
                )

                # Execute top signals (up to max open positions)
                entries_this_cycle = 0
                max_entries_per_cycle = 2  # don't chase more than 2/cycle

                for signal in signals[:10]:  # process top 10
                    if entries_this_cycle >= max_entries_per_cycle:
                        break

                    # Skip if already in this position
                    already_open = any(
                        t["symbol"] == signal["symbol"]
                        for t in portfolio.open_positions
                    )
                    if already_open:
                        continue

                    result = await portfolio.execute_entry(signal)
                    if result:
                        entries_this_cycle += 1
                        await notifier.notify_entry({**result, **signal})

            # Timing
            elapsed = (utc_now() - cycle_start).total_seconds()
            sleep_time = max(5, scan_interval - elapsed)
            logger.debug("Cycle %d done in %.1fs, sleeping %.1fs",
                         bot_state.cycle_id, elapsed, sleep_time)
            await asyncio.sleep(sleep_time)

        except KeyboardInterrupt:
            bot_state.running = False
            break
        except Exception as e:
            logger.error("Main loop error: %s", e, exc_info=True)
            watchdog.record_error(str(e))
            await asyncio.sleep(30)

    # Shutdown
    logger.info("Shutting down...")
    poll_task.cancel()
    bot_state.save()
    await notifier.send_message("🛑 <b>Bot Offline</b>")


if __name__ == "__main__":
    asyncio.run(main())
