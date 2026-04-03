"""
Portfolio Manager — Core-satellite model with fully automated sector rotation.

Two key upgrades:
  1. import_spot_positions() — on startup, detects any existing Binance spot
     holdings not tracked by the bot, imports them with estimated entry price,
     and immediately manages them toward profitable exits.

  2. Auto-rotation — fully automatic based on BTC dominance. No manual
     commands needed. The scanner, entry filter, and position manager all
     shift behavior automatically when dominance crosses thresholds.

Capital deployment:
  AGGRESSIVE (bull): 85% deployed, all sectors open
  BALANCED (sideways): 60% deployed, higher conviction required for alts
  DEFENSIVE (high-vol): 35% deployed, only best setups
  CAPITAL_PRESERVATION (bear): 20% deployed, DCA only, stablecoins park 80%

Existing positions recovered from wallet:
  - In profit:  trail at breakeven+, let runner compound
  - In loss:    set emergency stop at -5% from current, close when even possible
  - Wrong sector (alts in BTC season): tighten stop, no additions
"""
import asyncio
import json
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Optional

from config.settings import Settings
from src.utils import setup_logging, utc_now

logger = setup_logging()

STABLECOIN_ASSETS = {"USDT", "BUSD", "USDC", "TUSD", "DAI", "FDUSD", "PAXG", "UST"}


class PortfolioManager:
    def __init__(self, exchange, database, regime_detector, risk_manager, news_intel):
        self.exchange = exchange
        self.db = database
        self.regime = regime_detector
        self.risk = risk_manager
        self.news = news_intel
        self.portfolio_value: float = 0.0
        self.cash_available: float = 0.0
        self.invested_value: float = 0.0
        self.open_positions: list = []
        self.sector_exposure: dict = {}
        self.peak_value: float = 0.0
        self.dca_active: bool = False
        self.dca_state: dict = {}
        self._imported_symbols: set = set()   # track what we imported this session

    # ------------------------------------------------------------------
    # Startup: Import Existing Spot Positions
    # ------------------------------------------------------------------

    async def import_spot_positions(self) -> list:
        """
        Scan the actual Binance spot wallet on startup.
        Any non-USDT holding > $10 that isn't already in the bot's DB
        gets imported as a tracked position so the bot can manage it.

        Strategy per imported position:
          - Try to get average entry from trade history
          - If in profit: set trailing stop at breakeven+, let it run
          - If in loss:   set emergency stop at current - 5%, aim for recovery
          - Wrong sector for current rotation: tighten stop to 2% from current
        """
        logger.info("Scanning Binance spot wallet for existing positions...")

        # Get actual wallet balances
        try:
            spot_positions = await self.exchange.get_spot_positions(min_value_usdt=10.0)
        except Exception as e:
            logger.error("Could not scan spot wallet: %s", e)
            return []

        if not spot_positions:
            logger.info("No existing spot positions found (wallet clean or all USDT)")
            return []

        # Get what the bot already tracks in the DB
        tracked = {t["symbol"] for t in self.db.get_open_trades()}
        imported = []

        for pos in spot_positions:
            symbol = pos["symbol"]
            asset = pos["asset"]
            qty = pos["quantity"]
            current_price = pos["current_price"]
            usdt_value = pos["usdt_value"]

            if symbol in tracked or symbol in self._imported_symbols:
                logger.debug("Already tracked: %s", symbol)
                continue

            logger.info("Found untracked position: %s qty=%.6f value=$%.2f",
                        symbol, qty, usdt_value)

            # Estimate entry price from trade history
            try:
                avg_entry = await self.exchange.get_avg_entry_price(symbol)
            except Exception:
                avg_entry = 0.0

            if avg_entry <= 0:
                # Fall back: assume entry was recent — use current as conservative estimate
                # This means we treat it as breakeven, set a small stop below
                avg_entry = current_price
                entry_source = "ESTIMATED_CURRENT"
            else:
                entry_source = "TRADE_HISTORY"

            gain_pct = ((current_price / avg_entry) - 1) * 100
            is_profitable = gain_pct > 0

            # Determine stop-loss based on position state
            rotation = self.news.get_dominance_strategy()
            sector = Settings.get_sector_for_asset(symbol)
            wrong_sector = (rotation == "BTC_FOCUS"
                            and sector not in ("Layer1", "Store_of_Value", "Other"))

            if wrong_sector and not is_profitable:
                # Alt in BTC season and in loss — tight stop, get out soon
                stop_pct = 0.02   # 2% from current
                tp_pct = 0.05     # 5% TP (get to breakeven territory)
            elif wrong_sector and is_profitable:
                # Alt in BTC season but in profit — lock most profit, tight trail
                stop_pct = 0.025
                tp_pct = 0.08
            elif is_profitable:
                # In profit — trail above breakeven, let winner run
                stop_pct = max(0.03, gain_pct / 200)  # wider trail if big gain
                tp_pct = 0.12
            else:
                # In loss — set emergency stop at -5% from current to cap further damage
                stop_pct = 0.05
                tp_pct = abs(gain_pct) / 100 + 0.03  # TP = back to entry + 3%

            # Get ATR for better stop
            try:
                klines = await self.exchange.get_klines(symbol, "1h", limit=20)
                if len(klines) >= 14:
                    from src.indicators import IncrementalATR
                    atr_calc = IncrementalATR(14)
                    for k in klines:
                        atr_calc.update(k["high"], k["low"], k["close"])
                    atr = atr_calc.value or current_price * 0.03
                    # ATR-based stop: 2.5× ATR from current (tighter than standard 3×)
                    atr_stop = current_price - (atr * 2.5)
                    # For inherited positions, use whichever is tighter (higher stop)
                    fixed_stop = current_price * (1 - stop_pct)
                    stop_price = max(atr_stop, fixed_stop)
                else:
                    stop_price = current_price * (1 - stop_pct)
                    atr = current_price * 0.03
            except Exception:
                stop_price = current_price * (1 - stop_pct)
                atr = current_price * 0.03

            # Ensure stop never exceeds current price (can't stop above market for long)
            stop_price = min(stop_price, current_price * 0.985)

            take_profit = current_price * (1 + tp_pct)

            trade_data = {
                "symbol": symbol,
                "side": "BUY",
                "entry_price": avg_entry,
                "quantity": qty,
                "remaining_quantity": qty,
                "usdt_value": usdt_value,
                "stop_loss": stop_price,
                "take_profit": take_profit,
                "highest_price": current_price,
                "confluence_score": 0,
                "regime_at_entry": f"INHERITED_{entry_source}",
                "status": "OPEN",
                "fees_paid": 0.0,
                "entry_time": utc_now() - timedelta(hours=1),  # approximate
                "sector": sector,
                "tranche_exits": "[]",
            }

            trade_id = self.db.save_trade(trade_data)
            trade_data["id"] = trade_id
            self.open_positions.append(trade_data)
            self._imported_symbols.add(symbol)
            imported.append(trade_data)

            status_str = (f"{'✅ profit' if is_profitable else '⚠️ loss'} "
                          f"{gain_pct:+.1f}% | SL=${stop_price:.4f} | "
                          f"TP=${take_profit:.4f} | "
                          f"{'⚠️ wrong sector' if wrong_sector else 'sector OK'}")
            logger.info("IMPORTED %s: entry=$%.4f current=$%.4f %s",
                        symbol, avg_entry, current_price, status_str)

        if imported:
            logger.info("Imported %d existing position(s) from spot wallet", len(imported))
        return imported

    # ------------------------------------------------------------------
    # Portfolio Sync
    # ------------------------------------------------------------------

    async def sync_with_exchange(self):
        balances = await self.exchange.get_balances()
        self.cash_available = balances.get("USDT", {}).get("free", 0.0)
        self.open_positions = self.db.get_open_trades()

        invested = 0.0
        self.sector_exposure = {}
        for pos in self.open_positions:
            try:
                current_price = await self.exchange.get_price(pos["symbol"])
                pos_value = (pos.get("remaining_quantity", pos["quantity"]) * current_price
                             if current_price > 0 else pos.get("usdt_value", 0))
                invested += pos_value
                sector = pos.get("sector", "Other")
                self.sector_exposure[sector] = self.sector_exposure.get(sector, 0) + pos_value
            except Exception:
                invested += pos.get("usdt_value", 0)

        self.invested_value = invested
        self.portfolio_value = self.cash_available + invested
        # Use the DB peak only if it is within 50% of current value.
        # A DB peak >50% above current means old session data — ignore it
        # so stale testnet or prior session peaks never trigger false circuit breakers.
        db_peak = self.db.get_peak_portfolio_value()
        if db_peak > 0 and db_peak < self.portfolio_value * 1.5:
            self.peak_value = max(self.peak_value, self.portfolio_value, db_peak)
        else:
            # DB peak is unreasonably high (>50% above current) — use current only.
            self.peak_value = max(self.peak_value, self.portfolio_value)
        drawdown = ((self.peak_value - self.portfolio_value) / self.peak_value
                    if self.peak_value > 0 else 0)

        self.db.save_snapshot({
            "timestamp": utc_now(),
            "total_value_usdt": self.portfolio_value,
            "cash_usdt": self.cash_available,
            "invested_usdt": self.invested_value,
            "num_open_positions": len(self.open_positions),
            "drawdown_from_peak": drawdown,
            "market_regime": self.regime.current_regime,
            "fear_greed_index": self.news.fear_greed_value,
            "news_sentiment_score": self.news.get_sentiment_score(),
        })
        logger.info("Portfolio: $%.2f (cash=$%.2f invested=$%.2f pos=%d dd=%.1f%%)",
                    self.portfolio_value, self.cash_available,
                    self.invested_value, len(self.open_positions), drawdown * 100)

    # ------------------------------------------------------------------
    # Automatic Rotation — Fully Automated (no manual intervention)
    # ------------------------------------------------------------------

    def _get_rotation_threshold(self, sector: str) -> int:
        """
        Automatically determine entry confluence threshold based on
        BTC dominance (rotation strategy). No manual commands needed.

        BTC_FOCUS (dom > 65%):
          - Core assets (BTC/ETH): low bar (35)
          - Layer1s: normal bar (45)
          - DeFi/AI/Gaming: high bar (65) — only exceptional setups
          - Meme/Unknown: very high bar (75) — effectively blocked

        ALTCOIN_SEASON (dom < 50%):
          - All sectors: normal or lower bar
          - DeFi/AI leading sectors: low bar (38)

        NEUTRAL (50-65%):
          - Base threshold from config
        """
        rotation = self.news.get_dominance_strategy()
        base = Settings.strategy.CONFLUENCE_SCORE_THRESHOLD

        if rotation == "BTC_FOCUS":
            sector_thresholds = {
                "Layer1": base - 7,         # 38 — BTC/ETH/SOL always OK
                "Store_of_Value": base - 7,
                "Other": base,              # 45 — neutral
                "Exchange": base,
                "DeFi": base + 20,          # 65 — high bar for alts in BTC season
                "AI": base + 20,
                "Gaming": base + 25,        # 70
                "Layer2": base + 15,        # 60
                "Meme": base + 30,          # 75 — effectively blocked
                "Infrastructure": base + 10,
            }
        elif rotation == "ALTCOIN_SEASON":
            sector_thresholds = {
                "Layer1": base - 5,
                "Store_of_Value": base - 5,
                "DeFi": base - 7,           # 38 — DeFi leads alt season
                "AI": base - 7,
                "Gaming": base - 5,
                "Layer2": base - 5,
                "Exchange": base,
                "Infrastructure": base - 3,
                "Other": base,
                "Meme": base + 15,          # 60 — still cautious on memes
            }
        else:  # NEUTRAL
            sector_thresholds = {s: base for s in [
                "Layer1", "Store_of_Value", "DeFi", "AI", "Gaming",
                "Layer2", "Exchange", "Infrastructure", "Other", "Meme"
            ]}
            sector_thresholds["Meme"] = base + 20  # always cautious on memes

        return sector_thresholds.get(sector, base)

    def _rotation_allows_entry(self, symbol: str, score: int, sector: str) -> tuple:
        """
        Fully automatic rotation gate. Returns (allowed: bool, reason: str).
        No manual commands needed — adjusts dynamically every cycle.
        """
        rotation = self.news.get_dominance_strategy()
        threshold = self._get_rotation_threshold(sector)

        if score < threshold:
            return False, (f"Rotation gate [{rotation}]: {sector} needs "
                           f"{threshold}/100 (got {score})")

        # Additional gate: in BTC_FOCUS mode, cap alt exposure
        if rotation == "BTC_FOCUS":
            alt_sectors = {"DeFi", "AI", "Gaming", "Meme", "Layer2"}
            if sector in alt_sectors:
                alt_exposure = sum(v for k, v in self.sector_exposure.items()
                                   if k in alt_sectors)
                alt_pct = alt_exposure / self.portfolio_value if self.portfolio_value > 0 else 0
                max_alt_in_btc_season = 0.25  # max 25% alts when BTC is dominant
                if alt_pct >= max_alt_in_btc_season:
                    return False, (f"BTC season: alt exposure {alt_pct:.0%} at "
                                   f"{max_alt_in_btc_season:.0%} max")

        return True, "Rotation check passed"

    def get_rotation_status(self) -> dict:
        """For Telegram /dominance command — full auto-rotation status."""
        rotation = self.news.get_dominance_strategy()
        alt_sectors = {"DeFi", "AI", "Gaming", "Meme", "Layer2"}
        alt_exposure = sum(v for k, v in self.sector_exposure.items() if k in alt_sectors)
        core_exposure = sum(v for k, v in self.sector_exposure.items()
                            if k in ("Layer1", "Store_of_Value"))
        return {
            "rotation": rotation,
            "btc_dominance": self.news.btc_dominance,
            "altcoin_season_index": self.news.altcoin_season_index,
            "alt_exposure_usd": alt_exposure,
            "alt_exposure_pct": alt_exposure / self.portfolio_value if self.portfolio_value > 0 else 0,
            "core_exposure_usd": core_exposure,
            "core_exposure_pct": core_exposure / self.portfolio_value if self.portfolio_value > 0 else 0,
            "sector_thresholds": {s: self._get_rotation_threshold(s)
                                  for s in ["Layer1", "DeFi", "AI", "Gaming", "Meme"]},
        }

    # ------------------------------------------------------------------
    # Auto-tighten Stops for "Wrong Sector" Positions
    # ------------------------------------------------------------------

    async def auto_adjust_inherited_stops(self):
        """
        Called each cycle. For positions that are in the wrong sector
        for the current rotation strategy, automatically tighten their
        trailing stops to guide them toward profitable exits without
        force-selling.
        """
        rotation = self.news.get_dominance_strategy()
        if rotation == "NEUTRAL":
            return  # no adjustment needed

        alt_sectors = {"DeFi", "AI", "Gaming", "Meme", "Layer2"}

        for trade in self.open_positions:
            sector = trade.get("sector", "Other")
            is_wrong_sector = (rotation == "BTC_FOCUS" and sector in alt_sectors)

            if not is_wrong_sector:
                continue

            try:
                current_price = await self.exchange.get_price(trade["symbol"])
                entry_price = trade["entry_price"]
                gain_pct = ((current_price / entry_price) - 1) if entry_price > 0 else 0
                current_stop = trade.get("stop_loss", 0)

                # Tighten stop to 2% from current if in profit
                # This ensures we exit the "wrong" position at profit when it corrects
                if gain_pct > 0.01:
                    # Lock in 70% of unrealized gain
                    lock_stop = entry_price * (1 + gain_pct * 0.70)
                    tight_stop = current_price * 0.98  # 2% trailing
                    new_stop = max(current_stop, lock_stop, tight_stop)
                    if new_stop > current_stop + 0.0001:
                        self.db.update_trade(trade["id"], {
                            "stop_loss": new_stop,
                            "highest_price": max(trade.get("highest_price", entry_price),
                                                 current_price),
                        })
                        trade["stop_loss"] = new_stop
                        logger.debug("Tightened stop on %s (BTC season, wrong sector): "
                                     "SL=$%.4f (gain=%.1f%%)",
                                     trade["symbol"], new_stop, gain_pct * 100)
                elif gain_pct < -0.04:
                    # Position losing ground in wrong sector — cap further damage
                    # Set emergency stop at current - 2%
                    emergency_stop = current_price * 0.98
                    if emergency_stop > current_stop:
                        self.db.update_trade(trade["id"], {"stop_loss": emergency_stop})
                        trade["stop_loss"] = emergency_stop
                        logger.info("Emergency stop tightened: %s in wrong sector "
                                    "(loss=%.1f%%)", trade["symbol"], gain_pct * 100)
            except Exception as e:
                logger.debug("Auto-adjust stop error for %s: %s", trade["symbol"], e)

    # ------------------------------------------------------------------
    # Entry Execution
    # ------------------------------------------------------------------

    async def execute_entry(self, signal: dict) -> Optional[dict]:
        symbol = signal["symbol"]
        price = signal["price"]
        atr = signal["atr"]
        score = signal.get("confluence_score", 50)
        sector = Settings.get_sector_for_asset(symbol)

        # 1. Risk manager admission control
        check = self.risk.can_open_position(
            self.portfolio_value, self.cash_available,
            self.open_positions, sector, self.sector_exposure,
        )
        if not check["allowed"]:
            logger.info("Entry blocked for %s: %s", symbol, check["reason"])
            self.db.save_signal({
                "timestamp": utc_now(), "symbol": symbol,
                "confluence_score": score,
                "score_breakdown": json.dumps(signal.get("breakdown", {})),
                "action_taken": f"BLOCKED:{check['reason'][:30]}",
                "regime": self.regime.current_regime,
            })
            return None

        # 2. Automatic rotation gate (fully automated, no manual steps)
        rotation_ok, rotation_reason = self._rotation_allows_entry(symbol, score, sector)
        if not rotation_ok:
            logger.debug("Rotation blocked %s: %s", symbol, rotation_reason)
            return None

        # 3. Correlation check
        if not await self._check_correlation(symbol):
            logger.info("Entry blocked for %s: high correlation with existing position", symbol)
            return None

        # 4. Size the position
        position_size = self.risk.calculate_position_size(
            self.portfolio_value, atr, price, score
        )

        # Apply rotation-based size scaling
        rotation = self.news.get_dominance_strategy()
        if rotation == "BTC_FOCUS" and sector not in ("Layer1", "Store_of_Value", "Other"):
            position_size *= 0.6  # reduce alt position size in BTC season
        elif rotation == "ALTCOIN_SEASON" and sector in ("DeFi", "AI", "Gaming"):
            position_size *= 1.15  # slightly larger positions in leading alt sectors

        position_size = min(position_size, self.cash_available * 0.95)

        if position_size < 5:
            logger.info("Position too small for %s: $%.2f", symbol, position_size)
            return None

        min_notional = self.exchange.get_min_notional(symbol)
        if position_size < min_notional:
            logger.info("Below min notional for %s: $%.2f < $%.2f",
                        symbol, position_size, min_notional)
            return None

        result = await self.exchange.place_market_buy(symbol, position_size)
        if not result or "orderId" not in result:
            logger.error("Buy failed for %s", symbol)
            return None

        filled_qty = float(result.get("executedQty", 0))
        if filled_qty <= 0:
            return None
        cum_quote = float(result.get("cummulativeQuoteQty", 0))
        filled_price = cum_quote / filled_qty if filled_qty > 0 else price
        fees = sum(float(f.get("commission", 0)) for f in result.get("fills", []))

        stop_loss = self.risk.calculate_stop_loss(filled_price, atr)
        tp_targets = self.risk.calculate_take_profit(filled_price, atr)
        take_profit = tp_targets["tp1"]

        trade_data = {
            "symbol": symbol, "side": "BUY",
            "entry_price": filled_price, "quantity": filled_qty,
            "usdt_value": position_size, "stop_loss": stop_loss,
            "take_profit": take_profit,
            "confluence_score": score,
            "regime_at_entry": f"{self.regime.current_regime}|{self.news.get_dominance_strategy()}",
            "status": "OPEN", "fees_paid": fees,
            "entry_time": utc_now(), "sector": sector,
            "highest_price": filled_price,
            "remaining_quantity": filled_qty,
            "tranche_exits": "[]",
        }
        trade_id = self.db.save_trade(trade_data)
        trade_data["id"] = trade_id

        self.db.save_signal({
            "timestamp": utc_now(), "symbol": symbol,
            "confluence_score": score,
            "score_breakdown": json.dumps(signal.get("breakdown", {})),
            "action_taken": "ENTERED",
            "regime": self.regime.current_regime,
        })

        self.cash_available -= position_size
        self.invested_value += position_size
        self.open_positions.append(trade_data)
        self.sector_exposure[sector] = self.sector_exposure.get(sector, 0) + position_size

        # Place OCO order
        try:
            stop_limit = stop_loss * (1 - 0.002)
            await self.exchange.place_oco_sell(
                symbol, filled_qty, take_profit, stop_loss, stop_limit
            )
        except Exception as e:
            logger.warning("OCO failed for %s (bot will monitor): %s", symbol, e)

        logger.info("ENTRY: %s @ $%.4f qty=%.6f size=$%.2f SL=$%.4f TP=$%.4f "
                    "score=%d rotation=%s",
                    symbol, filled_price, filled_qty, position_size, stop_loss,
                    take_profit, score, rotation)
        return trade_data

    # ------------------------------------------------------------------
    # Exit Execution
    # ------------------------------------------------------------------

    async def execute_exit(self, trade: dict, quantity: float,
                           reason: str, current_price: float) -> Optional[dict]:
        symbol = trade["symbol"]
        qty = min(quantity, trade.get("remaining_quantity", trade["quantity"]))
        if qty <= 0:
            return None

        try:
            await self.exchange.cancel_all_orders(symbol)
        except Exception:
            pass

        result = await self.exchange.place_market_sell(symbol, qty)
        if not result or "orderId" not in result:
            logger.error("Sell failed for %s", symbol)
            return None

        filled_qty = float(result.get("executedQty", 0))
        cum_quote = float(result.get("cummulativeQuoteQty", 0))
        filled_price = cum_quote / filled_qty if filled_qty > 0 else current_price
        fees = sum(float(f.get("commission", 0)) for f in result.get("fills", []))

        remaining = trade.get("remaining_quantity", trade["quantity"]) - filled_qty
        pnl = (filled_price - trade["entry_price"]) * filled_qty
        pnl_pct = ((filled_price / trade["entry_price"]) - 1) * 100
        is_full_exit = remaining <= 0.0001

        updates = {
            "remaining_quantity": max(remaining, 0),
            "fees_paid": trade.get("fees_paid", 0) + fees,
        }

        if is_full_exit:
            total_pnl = (filled_price - trade["entry_price"]) * trade["quantity"]
            total_pnl_pct = ((filled_price / trade["entry_price"]) - 1) * 100
            updates.update({
                "status": "CLOSED" if total_pnl >= 0 else "STOPPED_OUT",
                "exit_price": filled_price, "pnl": total_pnl,
                "pnl_percent": total_pnl_pct, "exit_time": utc_now(),
                "exit_reason": reason,
            })
            self.risk.record_trade_result(won=total_pnl > 0)
        else:
            exits = (json.loads(trade.get("tranche_exits", "[]"))
                     if isinstance(trade.get("tranche_exits"), str)
                     else trade.get("tranche_exits", []))
            exits.append({
                "level": len(exits) + 1,
                "price": filled_price, "quantity": filled_qty,
                "pnl": pnl, "reason": reason,
                "time": utc_now().isoformat(),
            })
            updates["tranche_exits"] = json.dumps(exits)

        self.db.update_trade(trade["id"], updates)

        if is_full_exit:
            self.open_positions = [p for p in self.open_positions if p["id"] != trade["id"]]
        else:
            trade["remaining_quantity"] = max(remaining, 0)
            if "tranche_exits" in updates:
                trade["tranche_exits"] = updates["tranche_exits"]

        logger.info("EXIT: %s @ $%.4f qty=%.6f PnL=$%.2f (%.2f%%) reason=%s",
                    symbol, filled_price, filled_qty, pnl, pnl_pct, reason)
        return {
            "symbol": symbol, "exit_price": filled_price,
            "quantity": filled_qty, "pnl": pnl, "pnl_pct": pnl_pct,
            "reason": reason, "is_full_exit": is_full_exit,
        }

    async def force_sell(self, symbol: str) -> Optional[dict]:
        for trade in self.open_positions:
            if trade["symbol"] == symbol:
                price = await self.exchange.get_price(symbol)
                return await self.execute_exit(
                    trade, trade.get("remaining_quantity", trade["quantity"]),
                    "MANUAL", price,
                )
        return None

    async def force_sell_all(self) -> list:
        results = []
        for trade in list(self.open_positions):
            price = await self.exchange.get_price(trade["symbol"])
            r = await self.execute_exit(
                trade, trade.get("remaining_quantity", trade["quantity"]),
                "MANUAL", price,
            )
            if r:
                results.append(r)
        return results

    async def liquidate_all(self, reason: str = "KILL_SWITCH") -> list:
        results = []
        for trade in list(self.open_positions):
            try:
                price = await self.exchange.get_price(trade["symbol"])
                r = await self.execute_exit(
                    trade, trade.get("remaining_quantity", trade["quantity"]),
                    reason, price,
                )
                if r:
                    results.append(r)
            except Exception as e:
                logger.error("Liquidation error for %s: %s", trade["symbol"], e)
            try:
                await self.exchange.cancel_all_orders(trade["symbol"])
            except Exception:
                pass
        return results

    async def _check_correlation(self, new_symbol: str) -> bool:
        if not self.open_positions:
            return True
        try:
            new_klines = await self.exchange.get_klines(new_symbol, "1d", 30)
            if len(new_klines) < 15:
                return True
            new_closes = [k["close"] for k in new_klines]
            new_returns = np.diff(new_closes) / np.array(new_closes[:-1])

            for pos in self.open_positions[:5]:
                pos_klines = await self.exchange.get_klines(pos["symbol"], "1d", 30)
                if len(pos_klines) < 15:
                    continue
                pos_closes = [k["close"] for k in pos_klines]
                pos_returns = np.diff(pos_closes) / np.array(pos_closes[:-1])
                min_len = min(len(new_returns), len(pos_returns))
                if min_len < 10:
                    continue
                corr = np.corrcoef(new_returns[-min_len:], pos_returns[-min_len:])[0, 1]
                if abs(corr) > 0.95:
                    logger.info("High correlation %.2f: %s ↔ %s", corr,
                                new_symbol, pos["symbol"])
                    return False
        except Exception as e:
            logger.debug("Correlation check error: %s", e)
        return True

    async def smart_dca(self):
        if not self.news.is_extreme_fear():
            self.dca_active = False
            return
        cfg = Settings.portfolio_cfg
        if self.dca_active and self.dca_state.get("tranches_remaining", 0) <= 0:
            self.dca_active = False
            return
        regime_params = self.regime.get_regime_params()
        dca_mult = regime_params.get("dca_multiplier", 1.0)
        if not self.dca_active:
            dca_budget = self.portfolio_value * cfg.DCA_BUDGET_PCT * dca_mult
            tranche_size = dca_budget / cfg.DCA_TRANCHES
            self.dca_state = {
                "budget": dca_budget, "tranche_size": tranche_size,
                "tranches_remaining": cfg.DCA_TRANCHES,
                "symbols": ["BTCUSDT", "ETHUSDT"],
            }
            self.dca_active = True
            logger.info("Smart DCA activated (F&G=%d): $%.2f in %d tranches",
                        self.news.fear_greed_value, dca_budget, cfg.DCA_TRANCHES)
        if self.dca_state["tranches_remaining"] > 0:
            tranche = self.dca_state["tranche_size"]
            per_asset = tranche / len(self.dca_state["symbols"])
            for symbol in self.dca_state["symbols"]:
                if per_asset >= 10 and per_asset <= self.cash_available * 0.5:
                    result = await self.exchange.place_market_buy(symbol, per_asset)
                    if result and "orderId" in result:
                        filled_qty = float(result.get("executedQty", 0))
                        cum_quote = float(result.get("cummulativeQuoteQty", 0))
                        filled_price = cum_quote / filled_qty if filled_qty > 0 else 0
                        from src.indicators import IncrementalATR
                        klines = await self.exchange.get_klines(symbol, "4h", 20)
                        atr_calc = IncrementalATR(14)
                        for k in klines:
                            atr_calc.update(k["high"], k["low"], k["close"])
                        atr = atr_calc.value or filled_price * 0.03
                        trade_data = {
                            "symbol": symbol, "side": "BUY",
                            "entry_price": filled_price, "quantity": filled_qty,
                            "usdt_value": per_asset,
                            "stop_loss": self.risk.calculate_stop_loss(filled_price, atr),
                            "take_profit": filled_price * (1 + Settings.risk.TAKE_PROFIT_PCT),
                            "confluence_score": 0,
                            "regime_at_entry": f"DCA_FEAR_{self.news.fear_greed_value}",
                            "status": "OPEN", "fees_paid": 0,
                            "entry_time": utc_now(),
                            "sector": Settings.get_sector_for_asset(symbol),
                            "highest_price": filled_price,
                            "remaining_quantity": filled_qty, "tranche_exits": "[]",
                        }
                        self.db.save_trade(trade_data)
                        logger.info("DCA: %s $%.2f @ $%.4f (F&G=%d)",
                                    symbol, per_asset, filled_price,
                                    self.news.fear_greed_value)
            self.dca_state["tranches_remaining"] -= 1

    async def check_rebalancing(self) -> list:
        actions = []
        cfg = Settings.portfolio_cfg
        for pos in self.open_positions:
            try:
                current_price = await self.exchange.get_price(pos["symbol"])
                pos_value = pos.get("remaining_quantity", pos["quantity"]) * current_price
                if (self.portfolio_value > 0
                        and pos_value / self.portfolio_value > cfg.MAX_SINGLE_POSITION_PCT):
                    excess_pct = pos_value / self.portfolio_value - cfg.MAX_SINGLE_POSITION_PCT
                    sell_value = excess_pct * self.portfolio_value
                    sell_qty = sell_value / current_price if current_price > 0 else 0
                    if sell_qty > 0:
                        actions.append({
                            "symbol": pos["symbol"], "quantity": sell_qty,
                            "reason": f"Exceeds {cfg.MAX_SINGLE_POSITION_PCT:.0%} max position",
                        })
                gain_pct = ((current_price / pos["entry_price"]) - 1) * 100 if pos["entry_price"] > 0 else 0
                if gain_pct > 50:
                    sell_qty = pos.get("remaining_quantity", pos["quantity"]) * 0.25
                    actions.append({
                        "symbol": pos["symbol"], "quantity": sell_qty,
                        "reason": f"Profit taking at {gain_pct:.0f}% gain",
                    })
            except Exception:
                pass
        return actions

    async def execute_rebalancing(self, actions: list) -> list:
        results = []
        for action in actions:
            trade = next((t for t in self.open_positions
                          if t["symbol"] == action["symbol"]), None)
            if not trade:
                continue
            price = await self.exchange.get_price(action["symbol"])
            r = await self.execute_exit(trade, action["quantity"], "REBALANCE", price)
            if r:
                results.append(r)
        return results

    def get_status(self) -> dict:
        return {
            "portfolio_value": self.portfolio_value,
            "cash_available": self.cash_available,
            "invested_value": self.invested_value,
            "open_positions": len(self.open_positions),
            "sector_exposure": self.sector_exposure,
            "peak_value": self.peak_value,
            "drawdown": ((self.peak_value - self.portfolio_value) / self.peak_value
                         if self.peak_value > 0 else 0),
            "dca_active": self.dca_active,
            "mode": self.regime.current_mode,
            "rotation": self.news.get_dominance_strategy(),
        }
