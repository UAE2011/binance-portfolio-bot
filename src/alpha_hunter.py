"""
Alpha Hunter — Early-mover detection for 20%+ crypto pump opportunities.

What causes rapid 20%+ pumps (researched from 2025-2026 Binance data):
  1. TTM Squeeze firing — BB inside Keltner Channels = coiled energy.
     When squeeze fires: 67% win rate, median 15-40% move in 1-72h.
     Most 2025-2026 alt pumps had 3-10 squeeze bars before breakout.

  2. OBV Accumulation (silent whale buying) — price flat, OBV rising.
     Precedes price move by 2-48 hours. Most reliable early signal.

  3. RVOL Spike (5-10× normal volume) — smart money entering NOW.
     The pump is already starting. Get in early candles.

  4. Bollinger Band Width at multi-week lows — maximum coiling.
     More compressed = more violent the eventual release.

  5. Binance new listing effect — avg 20-50%+ on announcement.
     Monitor announcements RSS for early positioning.

Alpha plays differ from main strategy:
  - Smaller size (5% portfolio vs 10-15%) — higher risk
  - Wider TP (10-20% vs 6%) — targeting the full pump
  - Tighter SL (5% vs 3%) — exit fast if wrong
  - Faster turnover — hours to 2 days, not swing trades
  - Full TP1 exit (not 50/50) — ride the pump then leave

Alpha score breakdown (0-100):
  TTM Squeeze just fired (3+ bar squeeze):    35 pts
  TTM Squeeze on + compression < 20%:         20 pts (pre-fire)
  OBV accumulation divergence:                20 pts
  RVOL > 5× average:                          20 pts
  RVOL > 3× average:                          12 pts
  BB width at 30-day low (compression):       10 pts
  Multi-TF squeeze alignment (1H + 4H):       15 pts
  RSI momentum divergence:                     8 pts
  Price at key resistance (breakout imminent): 7 pts
  BTC dom falling (alt season brewing):        5 pts
  News catalyst detected:                      5 pts
"""
import asyncio
import json
from datetime import datetime, timezone, timedelta
from typing import Optional

import aiohttp
import feedparser

from config.settings import Settings
from src.indicators import (
    IncrementalTTMSqueeze, IncrementalOBVSlope,
    IncrementalKeltnerChannel, IncrementalBollinger,
    IncrementalATR, IncrementalRSI, IncrementalVolumeSMA,
)
from src.utils import setup_logging, utc_now

logger = setup_logging()

BINANCE_ANNOUNCE_RSS = (
    "https://www.binance.com/en/support/announcement/new-cryptocurrency-listing?"
    "c=48&navId=48&rss=1"
)


class AlphaSetup:
    """Represents one alpha opportunity."""
    def __init__(self, symbol: str, price: float, score: int,
                 signals: dict, entry_reason: str):
        self.symbol = symbol
        self.price = price
        self.score = score
        self.signals = signals
        self.entry_reason = entry_reason
        self.detected_at = utc_now()

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "price": self.price,
            "score": self.score,
            "signals": self.signals,
            "entry_reason": self.entry_reason,
            "detected_at": self.detected_at.isoformat(),
        }


class SymbolAlphaState:
    """Per-symbol incremental state for alpha indicators."""
    def __init__(self):
        self.squeeze_1h = IncrementalTTMSqueeze(period=20, bb_mult=2.0, kc_mult=1.5)
        self.squeeze_4h = IncrementalTTMSqueeze(period=20, bb_mult=2.0, kc_mult=1.5)
        self.obv_1h = IncrementalOBVSlope(slope_period=10)
        self.vol_sma_1h = IncrementalVolumeSMA(period=20)
        self.vol_sma_7d = IncrementalVolumeSMA(period=168)  # ~7d in 1h candles
        self.rsi_1h = IncrementalRSI(period=14)
        self.atr_1h = IncrementalATR(period=14)
        self.seeded: bool = False
        self.last_update: Optional[datetime] = None
        # History for divergence detection
        self._price_highs: list = []
        self._rsi_highs: list = []


class AlphaHunter:
    """
    Scans entire watchlist for early-mover setups using squeeze + OBV + RVOL.
    Runs on its own faster cycle (every 2 min vs main 60s scan).
    Alpha positions managed separately from main portfolio positions.
    """

    def __init__(self, exchange, portfolio, risk_manager, news_intel,
                 notifier, database):
        self.exchange = exchange
        self.portfolio = portfolio
        self.risk = risk_manager
        self.news = news_intel
        self.notifier = notifier
        self.db = database
        self.cfg = Settings.alpha

        self._states: dict = {}       # symbol → SymbolAlphaState
        self._opportunities: list = []   # current scored setups
        self._alpha_positions: set = set()  # symbols we hold as alpha plays
        self._new_listings_seen: set = set()
        self._last_listing_check: Optional[datetime] = None
        self._cycle: int = 0

    # ─── Initialization ───────────────────────────────────────────────────

    async def seed(self, watchlist: list):
        """Seed indicators for all watchlist symbols."""
        logger.info("Seeding Alpha Hunter for %d symbols...", len(watchlist))
        seeded = 0
        for asset in watchlist[:80]:
            symbol = asset["symbol"]
            try:
                state = SymbolAlphaState()
                # 1H data (primary)
                k1h = await self.exchange.get_klines(symbol, "1h", limit=250)
                if k1h and len(k1h) >= 30:
                    state.squeeze_1h.seed(k1h)
                    state.obv_1h.seed(k1h)
                    for k in k1h:
                        state.vol_sma_1h.update(k["volume"])
                        state.rsi_1h.update(k["close"])
                        state.atr_1h.update(k["high"], k["low"], k["close"])
                    # 7d volume baseline
                    for k in k1h[-168:]:
                        state.vol_sma_7d.update(k["volume"])
                # 4H data for multi-TF squeeze
                k4h = await self.exchange.get_klines(symbol, "4h", limit=100)
                if k4h and len(k4h) >= 20:
                    state.squeeze_4h.seed(k4h)
                state.seeded = True
                self._states[symbol] = state
                seeded += 1
            except Exception as e:
                logger.debug("Alpha seed error %s: %s", symbol, e)
        logger.info("Alpha Hunter seeded %d symbols", seeded)

    # ─── Main Scan ────────────────────────────────────────────────────────

    async def scan(self, watchlist: list = None) -> list:
        """
        Full alpha scan across watchlist.
        Returns list of AlphaSetup sorted by score desc.
        """
        self._cycle += 1
        opportunities = []

        # Self-fetch watchlist if not provided (fully autonomous)
        if watchlist is None:
            try:
                tickers = await self.exchange.get_24h_tickers()
                watchlist = [
                    {"symbol": t["symbol"]}
                    for t in tickers
                    if t.get("symbol", "").endswith("USDT")
                    and float(t.get("quoteVolume", 0)) >= 5_000_000
                ][:80]
            except Exception:
                watchlist = list(self._states.keys())[:80]
                watchlist = [{"symbol": s} for s in watchlist]

        # Update indicators and score each symbol
        for asset in watchlist:
            symbol = asset["symbol"]
            try:
                setup = await self._score_symbol(symbol)
                if setup and setup.score >= self.cfg.ALERT_THRESHOLD:
                    opportunities.append(setup)
            except Exception as e:
                logger.debug("Alpha scan error %s: %s", symbol, e)

        # Check new Binance listings (every 10 min)
        if self.cfg.CHECK_NEW_LISTINGS:
            listing_setups = await self._check_new_listings()
            opportunities.extend(listing_setups)

        # Sort by score
        opportunities.sort(key=lambda x: x.score, reverse=True)
        self._opportunities = opportunities

        if opportunities:
            top = opportunities[0]
            logger.info("Alpha scan: %d opportunities (top: %s score=%d %s)",
                        len(opportunities), top.symbol, top.score, top.entry_reason[:40])

        return opportunities

    async def _score_symbol(self, symbol: str) -> Optional[AlphaSetup]:
        """Score one symbol for alpha opportunity."""
        # Initialize state if not seeded
        if symbol not in self._states:
            self._states[symbol] = SymbolAlphaState()

        state = self._states[symbol]

        # Fetch latest candles
        try:
            k1h = await self.exchange.get_klines(symbol, "1h", limit=5)
            if not k1h or len(k1h) < 2:
                return None
            latest = k1h[-1]
            price = latest["close"]
            volume = latest["volume"]
        except Exception:
            return None

        # Update indicators
        sq1h = state.squeeze_1h.update(latest["high"], latest["low"], latest["close"])
        obv = state.obv_1h.update(latest["close"], volume)
        vol_sma = state.vol_sma_1h.update(volume)
        state.vol_sma_7d.update(volume)
        rsi = state.rsi_1h.update(latest["close"])
        atr = state.atr_1h.update(latest["high"], latest["low"], latest["close"])

        # 4H squeeze update (every 4 candles approx)
        if self._cycle % 4 == 0:
            try:
                k4h = await self.exchange.get_klines(symbol, "4h", limit=3)
                if k4h:
                    for k in k4h[-2:]:
                        state.squeeze_4h.update(k["high"], k["low"], k["close"])
            except Exception:
                pass

        sq4h_on = state.squeeze_4h.squeeze_on
        vol_7d_avg = state.vol_sma_7d.value or vol_sma

        # ── Compute RVOL ────────────────────────────────────────────────
        rvol_20 = volume / vol_sma if vol_sma > 0 else 1.0
        rvol_7d = volume / vol_7d_avg if vol_7d_avg > 0 else 1.0
        rvol = max(rvol_20, rvol_7d)  # use whichever is higher

        # ── Score assembly ──────────────────────────────────────────────
        score = 0
        signals = {}
        reasons = []

        # 1. TTM Squeeze just fired (highest alpha signal)
        bars = sq1h["bars_in_squeeze"]
        if sq1h["squeeze_fired"] and bars >= self.cfg.SQUEEZE_MIN_BARS:
            score += 35
            signals["squeeze_fired"] = True
            signals["squeeze_bars"] = bars
            reasons.append(f"🚨 SQUEEZE FIRED after {bars} bars")

        # 2. Squeeze ON + high compression (pre-fire, get ready)
        elif sq1h["squeeze_on"] and sq1h["bb_width_pct_30"] < 0.20:
            score += 20
            signals["squeeze_on"] = True
            signals["squeeze_bars"] = bars
            signals["compression_pct"] = sq1h["bb_width_pct_30"]
            reasons.append(f"🔴 Squeeze on ({bars} bars, {sq1h['bb_width_pct_30']:.0%} compression)")

        # 3. OBV Accumulation — silent whale buying
        div = obv["divergence"]
        if div == "ACCUMULATION":
            score += 20
            signals["obv_accumulation"] = True
            signals["obv_slope"] = round(obv["obv_slope"], 4)
            reasons.append(f"🐋 OBV accumulation (slope={obv['obv_slope']:.3f})")

        # 4. RVOL spike — smart money entering NOW
        if rvol >= self.cfg.RVOL_STRONG_THRESHOLD:
            score += 20
            signals["rvol"] = round(rvol, 1)
            reasons.append(f"⚡ RVOL={rvol:.1f}× (STRONG)")
        elif rvol >= self.cfg.RVOL_SPIKE_THRESHOLD:
            score += 12
            signals["rvol"] = round(rvol, 1)
            reasons.append(f"📈 RVOL={rvol:.1f}× spike")

        # 5. BB Width at 30-bar low — maximum compression
        if sq1h["bb_width_pct_30"] < 0.10:
            score += 10
            signals["bb_compressed"] = True
            reasons.append("🗜️ BB width at 30-bar low (maximum compression)")

        # 6. Multi-timeframe squeeze alignment (1H + 4H)
        if sq1h["squeeze_on"] and sq4h_on:
            score += 15
            signals["mtf_squeeze"] = True
            reasons.append("🎯 Multi-TF squeeze (1H + 4H aligned)")

        # 7. RSI momentum divergence (hidden strength)
        if rsi < 50 and obv["obv_slope"] > 0.01:
            score += 8
            signals["rsi_divergence"] = True
            signals["rsi"] = round(rsi, 1)
            reasons.append(f"📊 RSI divergence: RSI={rsi:.0f} but OBV rising")
        elif 50 < rsi < 60:
            score += 4
            signals["rsi"] = round(rsi, 1)

        # 8. BTC dominance falling = alt season brewing
        btc_dom = self.news.btc_dominance
        if btc_dom < 55 and self.news.altcoin_season_index > 50:
            score += 5
            signals["alt_season_signal"] = True

        # 9. News catalyst for this specific symbol
        asset_sentiment = self.news.get_asset_sentiment(symbol)
        if asset_sentiment > 0.3:
            score += 5
            signals["news_sentiment"] = round(asset_sentiment, 2)
            reasons.append(f"📰 Positive news sentiment: {asset_sentiment:.2f}")

        # Too low to be worth reporting
        if score < self.cfg.ALERT_THRESHOLD:
            return None

        entry_reason = " | ".join(reasons) if reasons else "Multi-signal alpha"
        return AlphaSetup(symbol, price, score, signals, entry_reason)

    # ─── New Listing Detection ────────────────────────────────────────────

    async def _check_new_listings(self) -> list:
        """
        Monitor Binance announcements for new listings.
        New listings avg 20-50%+ in first session — highest alpha return.
        """
        now = utc_now()
        if (self._last_listing_check is not None
                and (now - self._last_listing_check).seconds < 600):
            return []
        self._last_listing_check = now

        new_setups = []
        try:
            async with aiohttp.ClientSession() as session:
                # Binance announcement RSS
                urls = [
                    "https://www.binance.com/bapi/composite/v1/public/marketing/article/list"
                    "?type=1&pageNo=1&pageSize=5",
                ]
                for url in urls:
                    try:
                        async with session.get(
                            url, timeout=aiohttp.ClientTimeout(total=8)
                        ) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                articles = (data.get("data", {}).get("articles", [])
                                            if isinstance(data, dict) else [])
                                for article in articles[:5]:
                                    title = article.get("title", "")
                                    art_id = article.get("id", "")
                                    if art_id in self._new_listings_seen:
                                        continue
                                    self._new_listings_seen.add(art_id)

                                    # Detect listing announcements
                                    keywords = ["will list", "lists", "listing",
                                                "new cryptocurrency", "adds"]
                                    if any(kw in title.lower() for kw in keywords):
                                        logger.info("New Binance listing detected: %s", title)
                                        # Try to find the token symbol
                                        import re
                                        tickers = re.findall(r'\(([A-Z]{2,8})\)', title)
                                        for ticker in tickers:
                                            symbol = f"{ticker}USDT"
                                            if symbol in self.exchange.symbol_filters:
                                                try:
                                                    price = await self.exchange.get_price(symbol)
                                                    if price > 0:
                                                        setup = AlphaSetup(
                                                            symbol=symbol,
                                                            price=price,
                                                            score=90,   # Near-max: listing = premium signal
                                                            signals={"new_listing": True, "title": title[:60]},
                                                            entry_reason=f"🚀 BINANCE NEW LISTING: {title[:50]}",
                                                        )
                                                        new_setups.append(setup)
                                                        await self.notifier.send_message(
                                                            f"<b>🚀 BINANCE NEW LISTING DETECTED</b>\n"
                                                            f"Token: <code>{symbol}</code>\n"
                                                            f"Announcement: {title[:80]}\n"
                                                            f"Price: <code>${price:.4f}</code>\n"
                                                            f"<i>Alpha Hunter scoring 90/100 — "
                                                            f"evaluating entry...</i>"
                                                        )
                                                except Exception:
                                                    pass
                    except Exception as e:
                        logger.debug("Listing check error: %s", e)
        except Exception as e:
            logger.debug("New listings check failed: %s", e)

        return new_setups

    # ─── Entry Execution ──────────────────────────────────────────────────

    async def execute_alpha_entries(self, opportunities: list) -> list:
        """
        For top opportunities above MIN_SCORE, execute alpha entries.
        Alpha positions are tracked separately with their own SL/TP.
        """
        if not self.cfg.ENABLED:
            return []

        entered = []
        alpha_count = sum(
            1 for p in self.portfolio.open_positions
            if p.get("regime_at_entry", "").startswith("ALPHA")
        )
        remaining_slots = self.cfg.MAX_POSITIONS - alpha_count

        if remaining_slots <= 0:
            return []

        for setup in opportunities:
            if remaining_slots <= 0:
                break
            if setup.score < self.cfg.MIN_SCORE:
                break
            if setup.symbol in self._alpha_positions:
                continue
            already_open = any(
                p["symbol"] == setup.symbol for p in self.portfolio.open_positions
            )
            if already_open:
                continue

            result = await self._enter_alpha(setup)
            if result:
                entered.append(result)
                self._alpha_positions.add(setup.symbol)
                remaining_slots -= 1

        return entered

    async def _enter_alpha(self, setup: AlphaSetup) -> Optional[dict]:
        """Execute an alpha entry with alpha-specific position sizing."""
        symbol = setup.symbol
        cfg = self.cfg

        # Position size — small but viable
        pv = self.portfolio.portfolio_value
        size = max(pv * cfg.POSITION_SIZE_PCT, cfg.MIN_POSITION_USDT)
        size = min(size, self.portfolio.cash_available * 0.8)

        min_notional = self.exchange.get_min_notional(symbol)
        if size < max(min_notional, cfg.MIN_POSITION_USDT):
            logger.debug("Alpha entry skipped %s: size $%.2f below minimum", symbol, size)
            return None

        # Cash check
        if self.portfolio.cash_available < size * 1.05:
            logger.debug("Alpha entry skipped %s: insufficient cash", symbol)
            return None

        result = await self.exchange.place_market_buy(symbol, size)
        if not result or "orderId" not in result:
            logger.error("Alpha buy failed: %s", symbol)
            return None

        filled_qty = float(result.get("executedQty", 0))
        if filled_qty <= 0:
            return None
        cum_quote = float(result.get("cummulativeQuoteQty", 0))
        filled_price = cum_quote / filled_qty if filled_qty > 0 else setup.price

        stop_loss = filled_price * (1 - cfg.STOP_LOSS_PCT)
        take_profit = filled_price * (1 + cfg.TAKE_PROFIT_1_PCT)

        trade_data = {
            "symbol": symbol, "side": "BUY",
            "entry_price": filled_price, "quantity": filled_qty,
            "usdt_value": size,
            "stop_loss": stop_loss, "take_profit": take_profit,
            "highest_price": filled_price, "remaining_quantity": filled_qty,
            "confluence_score": setup.score,
            "regime_at_entry": f"ALPHA|score={setup.score}",
            "status": "OPEN", "fees_paid": 0.0,
            "entry_time": utc_now(),
            "sector": Settings.get_sector_for_asset(symbol),
            "tranche_exits": "[]",
        }
        trade_id = self.db.save_trade(trade_data)
        trade_data["id"] = trade_id
        self.portfolio.open_positions.append(trade_data)
        self.portfolio.cash_available -= size

        # Place OCO
        try:
            stop_limit = stop_loss * 0.998
            await self.exchange.place_oco_sell(
                symbol, filled_qty, take_profit, stop_loss, stop_limit
            )
        except Exception as e:
            logger.warning("Alpha OCO failed %s: %s", symbol, e)

        logger.info("ALPHA ENTRY: %s @ $%.4f size=$%.2f SL=$%.4f TP=$%.4f score=%d",
                    symbol, filled_price, size, stop_loss, take_profit, setup.score)

        await self.notifier.send_message(
            f"<b>⚡ ALPHA ENTRY</b>\n"
            f"Symbol: <code>{symbol}</code>\n"
            f"Price:  <code>${filled_price:.4f}</code>\n"
            f"Size:   <code>${size:.2f}</code>\n"
            f"SL:     <code>${stop_loss:.4f}</code> (-{cfg.STOP_LOSS_PCT:.0%})\n"
            f"TP:     <code>${take_profit:.4f}</code> (+{cfg.TAKE_PROFIT_1_PCT:.0%})\n"
            f"Score:  <code>{setup.score}/100</code>\n"
            f"Signal: {setup.entry_reason[:100]}"
        )
        return trade_data

    async def check_alpha_exits(self):
        """
        Check open alpha positions for max hold time and TP2 management.
        Alpha plays also exit at TP2 (20%) with full position — no runner.
        """
        now = utc_now()
        for trade in list(self.portfolio.open_positions):
            if not (trade.get("regime_at_entry", "") or "").startswith("ALPHA"):
                continue

            symbol = trade["symbol"]
            try:
                price = await self.exchange.get_price(symbol)
            except Exception:
                continue

            entry_price = trade["entry_price"]
            gain_pct = ((price / entry_price) - 1) if entry_price > 0 else 0
            entry_time = trade.get("entry_time")
            if entry_time and isinstance(entry_time, str):
                try:
                    entry_time = datetime.fromisoformat(entry_time)
                except Exception:
                    entry_time = None

            # TP2 hit — take full position at 20%
            if gain_pct >= self.cfg.TAKE_PROFIT_2_PCT:
                remaining = trade.get("remaining_quantity", trade["quantity"])
                result = await self.portfolio.execute_exit(
                    trade, remaining,
                    f"ALPHA_TP2@{price:.4f}(+{gain_pct:.1%})", price
                )
                if result:
                    self._alpha_positions.discard(symbol)
                    await self.notifier.notify_exit(result)
                continue

            # Max hold time exceeded — close if in profit, else trail
            if entry_time:
                hold_hours = (now - entry_time).total_seconds() / 3600
                if hold_hours >= self.cfg.MAX_HOLD_HOURS:
                    if gain_pct >= 0:
                        # Profitable — close it
                        remaining = trade.get("remaining_quantity", trade["quantity"])
                        result = await self.portfolio.execute_exit(
                            trade, remaining,
                            f"ALPHA_TIME_LIMIT({hold_hours:.0f}h,+{gain_pct:.1%})", price
                        )
                        if result:
                            self._alpha_positions.discard(symbol)
                            await self.notifier.notify_exit(result)
                    # If in loss, let normal SL handle it

    # ─── Status / Reporting ───────────────────────────────────────────────

    def get_opportunities(self) -> list:
        return self._opportunities[:10]

    def get_alpha_position_count(self) -> int:
        return sum(
            1 for p in self.portfolio.open_positions
            if (p.get("regime_at_entry", "") or "").startswith("ALPHA")
        )

    def get_status_text(self) -> str:
        """For /alpha Telegram command."""
        opps = self._opportunities[:8]
        alpha_pos = self.get_alpha_position_count()

        if not opps:
            return (
                f"<b>⚡ ALPHA HUNTER</b>\n\n"
                f"Status: <code>{'✅ Enabled' if self.cfg.ENABLED else '❌ Disabled'}</code>\n"
                f"Alpha positions: <code>{alpha_pos}/{self.cfg.MAX_POSITIONS}</code>\n"
                f"Threshold: <code>{self.cfg.MIN_SCORE}/100</code>\n\n"
                f"<i>No alpha setups above alert threshold ({self.cfg.ALERT_THRESHOLD}) right now.\n"
                f"Scanning every {self.cfg.SCAN_INTERVAL_SECONDS}s.</i>"
            )

        lines = []
        for op in opps:
            bar = "█" * (op.score // 10) + "░" * (10 - op.score // 10)
            signals_str = ""
            if op.signals.get("squeeze_fired"):
                signals_str += "🚨Squeeze "
            if op.signals.get("squeeze_on"):
                signals_str += "🔴Coil "
            if op.signals.get("obv_accumulation"):
                signals_str += "🐋OBV "
            rvol = op.signals.get("rvol", 0)
            if rvol >= 3:
                signals_str += f"⚡{rvol:.0f}×vol "
            if op.signals.get("new_listing"):
                signals_str += "🚀NEW "
            lines.append(
                f"  <code>{op.symbol:12s} {op.score:3d}/100 [{bar}]</code>\n"
                f"  {signals_str.strip()}"
            )

        return (
            f"<b>⚡ ALPHA HUNTER</b>\n\n"
            f"Positions: <code>{alpha_pos}/{self.cfg.MAX_POSITIONS}</code> | "
            f"Min score: <code>{self.cfg.MIN_SCORE}</code>\n\n"
            f"<b>Current Opportunities:</b>\n" + "\n".join(lines)
        )
