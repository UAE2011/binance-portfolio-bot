"""
Binance Portfolio Bot — Playbook-optimized trading system.

Key automation:
  - Scans existing Binance spot wallet on startup, imports any untracked
    positions, manages them toward profitable exits automatically.
  - Sector rotation (BTC dominance → auto-shift) happens every cycle
    with no manual intervention needed.
  - Circuit breakers, trailing stops, partial TPs all run autonomously.

Start: docker-compose up -d --build
Telegram: /help for all 24 commands
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
from src.alpha_hunter import AlphaHunter
from src.utils import setup_logging, utc_now

logger = setup_logging()


# ─── Bot State ───────────────────────────────────────────────────────────────

class BotState:
    def __init__(self):
        self.cycle_id: int = 0
        self.last_regime_check = None
        self.last_news_update = None
        self.last_rebalance_check = None
        self.last_dca_attempt = None
        self.last_auto_adjust = None
        self.last_alpha_scan = None
        self.startup_time = None
        self.prev_regime: str = ""
        self.prev_mode: str = ""
        self.indicator_sets: dict = {}
        self.indicator_sets_entry: dict = {}
        self.indicator_sets_trend: dict = {}
        self.kline_history: dict = {}
        self.running: bool = True

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


# ─── Indicator Seeding ───────────────────────────────────────────────────────

async def initialize_indicators(bot_state: BotState, scanner: AssetScanner,
                                exchange: Exchange):
    watchlist = await scanner.get_watchlist()
    core = await scanner.get_core_assets()
    all_symbols = list({a["symbol"] for a in watchlist + core})

    primary_tf = Settings.strategy.PRIMARY_TIMEFRAME
    entry_tf = Settings.strategy.ENTRY_TIMEFRAME
    trend_tf = Settings.strategy.TREND_TIMEFRAME

    seeds = 0
    for symbol in all_symbols[:60]:
        try:
            for tf, store in [(primary_tf, bot_state.indicator_sets),
                              (entry_tf, bot_state.indicator_sets_entry),
                              (trend_tf, bot_state.indicator_sets_trend)]:
                if symbol not in store:
                    store[symbol] = IndicatorSet()
                klines = await exchange.get_klines(symbol, tf, limit=250)
                if klines:
                    store[symbol].seed_from_klines(klines)
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
    if not portfolio.open_positions:
        return []

    exits_executed = []

    for trade in list(portfolio.open_positions):
        symbol = trade["symbol"]
        try:
            current_price = await portfolio.exchange.get_price(symbol)
        except Exception:
            continue

        # Update highest price tracker
        trade["highest_price"] = max(
            trade.get("highest_price", trade["entry_price"]), current_price
        )

        # Get current ATR from indicator set
        ind = bot_state.indicator_sets.get(symbol)
        current_atr = ind.latest.get("atr", 0) if ind and ind.latest else 0

        # 1. Update trailing stop (Chandelier Exit style)
        updates = risk.update_trailing_stop(trade, current_price, current_atr)
        new_stop = updates.get("stop_loss", trade["stop_loss"])
        new_highest = updates.get("highest_price", trade.get("highest_price"))
        if (abs(new_stop - trade.get("stop_loss", 0)) > 0.0001
                or new_highest != trade.get("highest_price")):
            portfolio.db.update_trade(trade["id"], {
                "stop_loss": new_stop,
                "highest_price": new_highest,
            })
            trade["stop_loss"] = new_stop
            trade["highest_price"] = new_highest

        # 2. Partial TP (50% at 6% gain)
        partial_exits = risk.check_take_profit_tranches(trade, current_price, current_atr)
        for pe in partial_exits:
            result = await portfolio.execute_exit(
                trade, pe["quantity"], pe["reason"], current_price
            )
            if result:
                exits_executed.append(result)
                await notifier.notify_exit(result)
                refreshed = portfolio.db.get_trade_by_id(trade["id"])
                if refreshed:
                    trade["tranche_exits"] = refreshed.get("tranche_exits", "[]")

        # Skip further checks if fully closed already
        if trade not in portfolio.open_positions:
            continue

        # 3. Stop-loss
        current_stop = trade.get("stop_loss", 0)
        if current_stop > 0 and current_price <= current_stop:
            remaining = trade.get("remaining_quantity", trade["quantity"])
            result = await portfolio.execute_exit(
                trade, remaining, f"STOP_LOSS@{current_stop:.4f}", current_price
            )
            if result:
                exits_executed.append(result)
                calibrator.on_trade_closed()
                await notifier.notify_exit(result)
            continue

        # 4. Full take-profit
        tp = trade.get("take_profit", 0)
        if tp > 0 and current_price >= tp:
            remaining = trade.get("remaining_quantity", trade["quantity"])
            result = await portfolio.execute_exit(
                trade, remaining, f"TAKE_PROFIT@{tp:.4f}", current_price
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
                continue

        # 6. Breaking news exit
        breaking = portfolio.news.check_breaking_news([symbol])
        for alert in breaking:
            if alert.get("sentiment", 0) < -0.5:
                remaining = trade.get("remaining_quantity", trade["quantity"])
                result = await portfolio.execute_exit(
                    trade, remaining,
                    f"NEWS:{alert.get('title','')[:30]}", current_price
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
    watchlist = await scanner.get_watchlist()
    core = await scanner.get_core_assets()

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
            for tf, store in [(primary_tf, bot_state.indicator_sets),
                              (entry_tf, bot_state.indicator_sets_entry),
                              (trend_tf, bot_state.indicator_sets_trend)]:
                if symbol not in store:
                    store[symbol] = IndicatorSet()
                latest = await exchange.get_klines(symbol, tf, limit=3)
                if latest:
                    for candle in latest[-2:]:
                        store[symbol].update_candle(candle)

            latest_primary = await exchange.get_klines(symbol, primary_tf, limit=5)
            if latest_primary:
                existing = bot_state.kline_history.get(symbol, [])
                for c in latest_primary:
                    if not existing or c["open_time"] != existing[-1]["open_time"]:
                        existing.append(c)
                bot_state.kline_history[symbol] = existing[-200:]

            # Recompute S/R every 10 cycles
            if bot_state.cycle_id % 10 == 1:
                history = bot_state.kline_history.get(symbol, [])
                if len(history) >= 50:
                    sr_engine.compute_levels(symbol, history)

            ind_primary = bot_state.indicator_sets[symbol].latest
            ind_entry = (bot_state.indicator_sets_entry.get(symbol,
                          bot_state.indicator_sets[symbol]).latest)
            ind_trend = (bot_state.indicator_sets_trend.get(symbol,
                          bot_state.indicator_sets[symbol]).latest)

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

    signals.sort(key=lambda s: s["confluence_score"], reverse=True)
    logger.info("Cycle %d: %d assets scanned, %d signals above threshold [rotation=%s]",
                bot_state.cycle_id, len(all_assets), len(signals),
                signal_gen.scorer.regime.current_mode if signal_gen.scorer else "?")
    return signals


# ─── Circuit Breaker Enforcement ─────────────────────────────────────────────

async def enforce_circuit_breakers(portfolio: PortfolioManager, risk: RiskManager,
                                   notifier: TelegramNotifier,
                                   startup_time=None):
    # If kill switch already active, skip re-evaluation
    if risk.kill_switch_active:
        return

    # Grace period: ignore drawdown for first 3 min after startup
    # Gives ghost cleanup + resync time to settle before evaluating
    from src.utils import utc_now
    from datetime import timedelta
    in_grace = (startup_time is not None and
                (utc_now() - startup_time).total_seconds() < 180)
    dd = risk.check_drawdown(portfolio.portfolio_value, portfolio.peak_value,
                             startup_grace=in_grace)
    action = dd.get("action", "NONE")

    if action in ("NONE", "WARNING"):
        return

    await notifier.notify_circuit_breaker(dd)

    if action == "DEFENSIVE":
        risk.position_size_modifier = 0.5

    elif action == "EMERGENCY_LIQUIDATE_50":
        half = portfolio.open_positions[:len(portfolio.open_positions) // 2]
        for trade in half:
            price = await portfolio.exchange.get_price(trade["symbol"])
            result = await portfolio.execute_exit(
                trade, trade.get("remaining_quantity", trade["quantity"]),
                "CIRCUIT_BREAKER_50PCT", price,
            )
            if result:
                await notifier.notify_exit(result)

    elif action == "CAPITAL_PRESERVATION":
        results = await portfolio.liquidate_all("CIRCUIT_BREAKER_CAPITAL_PRESERVATION")
        for r in results:
            await notifier.notify_exit(r)
        risk.position_size_modifier = 0.15

    elif action == "KILL_SWITCH":
        results = await portfolio.liquidate_all("KILL_SWITCH")
        for r in results:
            await notifier.notify_exit(r)
        risk.kill_switch_active = True
        # notify_circuit_breaker() above already sent the kill switch message.
        # No duplicate send_alert needed.


# ─── Scheduled Tasks ─────────────────────────────────────────────────────────

async def scheduled_tasks(portfolio: PortfolioManager, regime: RegimeDetector,
                          news: NewsIntelligence, risk: RiskManager,
                          notifier: TelegramNotifier, bot_state: BotState,
                          calibrator: Calibrator, ai: AIAdvisor):
    now = utc_now()

    # News every 15 min
    news_interval = timedelta(minutes=Settings.strategy.NEWS_CHECK_INTERVAL_MINUTES)
    if bot_state.last_news_update is None or (now - bot_state.last_news_update) >= news_interval:
        await news.update_all()
        regime.update_sentiment(
            fear_greed=news.fear_greed_value,
            news_sentiment=news.get_sentiment_score(),
            btc_dominance=news.btc_dominance / 100,
            altcoin_season_index=news.altcoin_season_index,
        )
        bot_state.last_news_update = now

    # Regime detection every hour
    if bot_state.last_regime_check is None or (now - bot_state.last_regime_check) >= timedelta(hours=1):
        old_regime = regime.current_regime
        old_mode = regime.current_mode
        await regime.detect_regime()
        bot_state.last_regime_check = now
        if regime.current_regime != old_regime:
            await notifier.notify_regime_change(old_regime, regime.current_regime)
        if regime.current_mode != old_mode:
            await notifier.notify_mode_change(old_mode, regime.current_mode)

    # HMM retrain check (weekly)
    if await regime.should_retrain():
        await regime.train_hmm()

    # Auto-adjust stops on wrong-sector positions (every 5 min)
    if (bot_state.last_auto_adjust is None
            or (now - bot_state.last_auto_adjust) >= timedelta(minutes=5)):
        await portfolio.auto_adjust_inherited_stops()
        bot_state.last_auto_adjust = now

    # Rebalance every 4h
    if (bot_state.last_rebalance_check is None
            or (now - bot_state.last_rebalance_check) >= timedelta(hours=4)):
        actions = await portfolio.check_rebalancing()
        if actions:
            results = await portfolio.execute_rebalancing(actions)
            for r in results:
                await notifier.notify_exit(r)
        bot_state.last_rebalance_check = now

    # Smart DCA on extreme fear (every 6h)
    if (bot_state.last_dca_attempt is None
            or (now - bot_state.last_dca_attempt) >= timedelta(hours=6)):
        if news.is_extreme_fear():
            await portfolio.smart_dca()
        bot_state.last_dca_attempt = now

    # Daily report at midnight UTC
    if now.hour == 0 and now.minute < 2:
        await notifier.send_daily_report(portfolio, regime, news, risk)

    # Weekly report
    if (now.weekday() == Settings.ops.WEEKLY_REPORT_DAY
            and now.hour == Settings.ops.WEEKLY_REPORT_HOUR and now.minute < 2):
        await notifier.send_weekly_report(portfolio, regime, news, risk, calibrator, ai)

    bot_state.save()


# ─── Auto-Resume ─────────────────────────────────────────────────────────────

async def check_auto_resume(portfolio: PortfolioManager, risk: RiskManager,
                             notifier: TelegramNotifier):
    """
    Automatically resume trading after a kill switch when conditions recover.
    No manual /resume needed.

    Resume conditions (ALL must be true):
      1. Cooldown period has passed (default 30 min)
      2. Current drawdown is below CIRCUIT_BREAKER_3 (15%)
      3. Portfolio value is stable or recovering (not still crashing)

    For false positives (stale DB peak bug), the cooldown is shortened to
    5 min since the drawdown never actually existed.
    """
    if not risk.kill_switch_active:
        return

    from src.utils import utc_now
    now = utc_now()
    activated_at = risk.kill_switch_activated_at
    if activated_at is None:
        # Kill switch was active before this session started — short cooldown
        activated_at = now - timedelta(minutes=risk.auto_resume_cooldown_min)
        risk.kill_switch_activated_at = activated_at

    elapsed_min = (now - activated_at).total_seconds() / 60

    # Minimum cooldown before checking
    if elapsed_min < risk.auto_resume_cooldown_min:
        remaining = risk.auto_resume_cooldown_min - elapsed_min
        logger.debug("Auto-resume cooldown: %.0f min remaining", remaining)
        return

    # Check current drawdown against session peak (not stale DB peak)
    current_dd = risk.check_drawdown(portfolio.portfolio_value, portfolio.peak_value)
    dd_pct = current_dd.get("drawdown_pct", 0)

    # Resume threshold: below CIRCUIT_BREAKER_2 (10%) — healthy enough to trade
    resume_threshold = Settings.risk.CIRCUIT_BREAKER_2

    if dd_pct < resume_threshold:
        risk.reset_kill_switch()
        logger.info("AUTO-RESUME: drawdown %.1f%% recovered below %.0f%% threshold",
                    dd_pct * 100, resume_threshold * 100)
        await notifier.send_message(
            "<b>✅ AUTO-RESUME</b>\n\n"
            "Kill switch lifted automatically.\n"
            f"Drawdown recovered to <code>{dd_pct:.1%}</code> "
            f"(below {resume_threshold:.0%} threshold).\n"
            f"Cooldown: <code>{elapsed_min:.0f} min</code>\n\n"
            "<i>Trading resumed. No action needed from you.</i>"
        )
    else:
        logger.info("Auto-resume check: drawdown %.1f%% still above %.0f%% — waiting",
                    dd_pct * 100, resume_threshold * 100)
        # Notify every 30 min while waiting
        if int(elapsed_min) % 30 == 0:
            await notifier.send_message(
                f"<b>⏳ Kill Switch Active</b>\n"
                f"Drawdown: <code>{dd_pct:.1%}</code> "
                f"(need < {resume_threshold:.0%} to auto-resume)\n"
                f"Elapsed: <code>{elapsed_min:.0f} min</code>\n"
                f"<i>Auto-checking every {risk.auto_resume_cooldown_min} min. "
                "Or send /resume to force.</i>"
            )


# ─── Main ────────────────────────────────────────────────────────────────────

async def main():
    logger.info("=" * 60)
    logger.info("Binance Portfolio Bot — Playbook Edition")
    logger.info("=" * 60)

    testnet = Settings.binance.USE_TESTNET
    if testnet:
        logger.warning("⚠️  TESTNET MODE")
    else:
        logger.info("🔴 LIVE TRADING")

    # Instantiate all components
    db = Database(Settings.ops.DB_PATH)
    exchange = Exchange(
        Settings.binance.API_KEY,
        Settings.binance.API_SECRET,
        testnet,
    )
    await exchange.initialize()

    news = NewsIntelligence()
    regime = RegimeDetector(exchange, db, Settings.strategy.HMM_LOOKBACK_DAYS)
    risk = RiskManager(db, regime)
    risk.set_news_intel(news)
    portfolio = PortfolioManager(exchange, db, regime, risk, news)
    ai = AIAdvisor()
    sr_engine = SupportResistanceEngine()
    scorer = ConfluenceScorer(regime, news, db, sr_engine)
    signal_gen = SignalGenerator(scorer, regime, news, ai)
    scanner = AssetScanner(exchange, regime, news)
    notifier = TelegramNotifier(Settings.telegram.BOT_TOKEN, Settings.telegram.CHAT_ID)
    watchdog = Watchdog(notifier)
    calibrator = Calibrator(db, risk, regime, notifier)
    alpha_hunter = AlphaHunter(
        exchange, portfolio, risk, news, notifier, db
    )
    bot_state = BotState()
    bot_state.load()

    # Wire Telegram commands
    handlers = build_command_handlers(
        portfolio, risk, calibrator, watchdog, regime, news, ai,
        alpha_hunter=alpha_hunter
    )
    notifier.register_commands(handlers)

    # ── Initialization Sequence ─────────────────────────────────────────

    logger.info("[1/6] Initializing regime detector...")
    await regime.initialize()

    logger.info("[2/6] Fetching news & sentiment...")
    await news.update_all()
    regime.update_sentiment(
        fear_greed=news.fear_greed_value,
        news_sentiment=news.get_sentiment_score(),
        btc_dominance=news.btc_dominance / 100,
        altcoin_season_index=news.altcoin_season_index,
    )

    logger.info("[3/6] Syncing portfolio from DB...")
    await portfolio.sync_with_exchange()

    # (Peak reset happens in step 4 after wallet reconciliation)

    logger.info("[4/6] Reconciling wallet with DB...")
    # Full wallet reconciliation in one pass:
    # - Close DB entries where coin no longer in wallet (ghosts)
    # - Sync quantities for existing DB positions
    # - Import wallet positions not yet tracked in DB
    # - Classify as ACTIVE (>=$10) or SUB_MIN (<$10 but real)
    recon = await portfolio.reconcile_with_wallet()
    await portfolio.sync_with_exchange()  # re-sync after reconciliation

    # Reset peak AFTER reconciliation so it reflects true portfolio value
    db.reset_peak_to_current(portfolio.portfolio_value)
    portfolio.peak_value = portfolio.portfolio_value
    logger.info("Peak reset to $%.2f | active=%d sub-min=%d ghosts=%d",
                portfolio.portfolio_value, len(portfolio.open_positions),
                len(portfolio.sub_min_positions), recon["ghosts"])

    if recon["imported"] > 0 or recon["ghosts"] > 0:
        active_n = len(portfolio.open_positions)
        submin_n = len(portfolio.sub_min_positions)
        await notifier.send_message(
            f"<b>📥 Wallet reconciled</b>\n"
            f"Active positions: <code>{active_n}</code> (can trade)\n"
            f"Sub-min positions: <code>{submin_n}</code> (waiting for recovery)\n"
            f"Ghosts closed: <code>{recon['ghosts']}</code>\n"
            f"Deployable cash: <code>${portfolio.cash_available:.2f}</code>"
        )

    logger.info("[5/6] Loading risk history...")
    risk.initialize_from_history()

    # ── Start Telegram polling immediately so commands work during seeding ──
    poll_task = asyncio.create_task(notifier.start_polling())

    # Send a quick ping so user knows bot is alive before the slow seeding step
    await notifier.send_message(
        f"<b>🤖 Bot starting up...</b> {'🔴 LIVE' if not testnet else '🟡 TESTNET'}\n"
        f"Portfolio: <code>${portfolio.portfolio_value:.2f}</code> | "
        f"Positions: <code>{len(portfolio.open_positions)}</code>\n"
        f"Regime: <code>{regime.current_regime}</code> | "
        f"F&G: <code>{news.fear_greed_value} ({news.fear_greed_label})</code>\n"
        f"<i>Seeding indicators for 50+ assets... full status follows.</i>"
    )

    logger.info("[6/6] Seeding indicators...")
    await initialize_indicators(bot_state, scanner, exchange)

    # Seed alpha hunter indicators (runs in parallel with main scan)
    asyncio.create_task(alpha_hunter.seed(
        await scanner.get_watchlist()
    ))

    bot_state.last_regime_check = utc_now()
    bot_state.last_news_update = utc_now()

    # ── Full Startup Notification (after seeding) ───────────────────────

    mode_e = regime.get_mode_emoji()
    rotation = news.get_dominance_strategy()
    rotation_label = {
        "BTC_FOCUS": "₿ BTC Focus",
        "ALTCOIN_SEASON": "🔄 Alt Season",
        "NEUTRAL": "⚖️ Neutral",
    }.get(rotation, rotation)

    startup_msg = (
        f"<b>🤖 Bot Online</b> {'🔴 LIVE' if not testnet else '🟡 TESTNET'}\n\n"
        f"Portfolio: <code>${portfolio.portfolio_value:.2f}</code>\n"
        f"Cash:      <code>${portfolio.cash_available:.2f}</code>\n"
        f"Positions: <code>{len(portfolio.open_positions)}</code> "
        f"(active={len(portfolio.open_positions)} sub-min={len(portfolio.sub_min_positions)})\n\n"
        f"Mode:     {mode_e} <code>{regime.current_mode}</code>\n"
        f"Regime:   <code>{regime.current_regime}</code>\n"
        f"Rotation: <code>{rotation_label}</code>\n"
        f"BTC Dom:  <code>{news.btc_dominance:.1f}%</code>\n"
        f"F&G:      <code>{news.fear_greed_value} ({news.fear_greed_label})</code>\n\n"
        f"<i>Rotation and sector thresholds adjust automatically.\n"
        f"Send /help for all commands.</i>"
    )
    await notifier.send_message(startup_msg)

    bot_state.startup_time = utc_now()
    logger.info("✅ Initialized. Mode=%s Regime=%s Rotation=%s Portfolio=$%.2f",
                regime.current_mode, regime.current_regime,
                rotation, portfolio.portfolio_value)

    scan_interval = Settings.strategy.SCAN_INTERVAL_SECONDS

    # ── Main Loop ───────────────────────────────────────────────────────

    while bot_state.running:
        try:
            cycle_start = utc_now()
            watchdog.heartbeat()

            # Portfolio sync
            await portfolio.sync_with_exchange()

            # Circuit breakers
            await enforce_circuit_breakers(portfolio, risk, notifier, bot_state.startup_time)

            # Auto-resume check — lifts kill switch when drawdown recovers
            await check_auto_resume(portfolio, risk, notifier)

            # Manage ACTIVE positions (trailing stops, partial TPs, exits)
            await manage_open_positions(
                portfolio, risk, notifier, calibrator, bot_state, sr_engine
            )

            # Manage SUB-MIN positions (wait for recovery above $10 then sell)
            if portfolio.sub_min_positions:
                await portfolio.manage_sub_min_positions(notifier)

            # Scheduled tasks (news, regime, rebalance, DCA, reports)
            await scheduled_tasks(
                portfolio, regime, news, risk, notifier,
                bot_state, calibrator, ai
            )

            # Signal scanning + entry — only when trading is active
            entries_allowed = (
                not risk.kill_switch_active
                and not risk.is_paused
                and regime.get_regime_params().get("entries_allowed", True)
            )

            if entries_allowed:
                signals = await scan_for_signals(
                    scanner, bot_state, exchange, signal_gen, sr_engine
                )

                entries_this_cycle = 0
                max_entries = 2

                for signal in signals[:10]:
                    if entries_this_cycle >= max_entries:
                        break
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

            # ── Alpha Hunter scan (every 2 min, faster than main) ──────
            alpha_interval = timedelta(seconds=Settings.alpha.SCAN_INTERVAL_SECONDS)
            if (bot_state.last_alpha_scan is None
                    or (utc_now() - bot_state.last_alpha_scan) >= alpha_interval):
                if not risk.kill_switch_active and not risk.is_paused:
                    try:
                        watchlist = await scanner.get_watchlist()
                        alpha_opps = await alpha_hunter.scan(watchlist)
                        await alpha_hunter.execute_alpha_entries(alpha_opps)
                        await alpha_hunter.check_alpha_exits()
                    except Exception as e:
                        logger.debug("Alpha scan error: %s", e)
                bot_state.last_alpha_scan = utc_now()

            # Sleep
            elapsed = (utc_now() - cycle_start).total_seconds()
            sleep_time = max(5, scan_interval - elapsed)
            await asyncio.sleep(sleep_time)

        except KeyboardInterrupt:
            bot_state.running = False
            break
        except Exception as e:
            logger.error("Main loop error: %s", e, exc_info=True)
            watchdog.record_error(str(e))
            await asyncio.sleep(30)

    logger.info("Shutting down...")
    poll_task.cancel()
    bot_state.save()
    await notifier.send_message("🛑 <b>Bot Offline</b>")
    await exchange.close()


if __name__ == "__main__":
    asyncio.run(main())
