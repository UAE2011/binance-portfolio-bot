"""
Portfolio Manager — Smart wallet-aware position management.

Three-tier position system:
  ACTIVE      — value >= $10 notional, can trade normally
  SUB_MIN     — value $1-$10, real position but below Binance minimum.
                Track it. Wait for price to recover above $10 then exit.
  GHOST       — in DB but asset has zero qty in wallet. Close in DB.

Startup sequence:
  1. scan_wallet_positions() — classify everything in wallet into ACTIVE/SUB_MIN
  2. clean_ghost_positions() — close DB positions where wallet qty = 0
  3. sync_db_with_wallet()   — for DB positions, update qty from wallet reality
  4. import_untracked()      — import wallet positions not yet in DB

This gives the bot a true picture of capital:
  deployable_cash   = USDT balance
  active_invested   = sum of ACTIVE position values
  sub_min_locked    = sum of SUB_MIN position values (can't sell yet)
  total_portfolio   = deployable_cash + active_invested + sub_min_locked
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
MIN_NOTIONAL = 10.0        # Binance minimum order value
DUST_THRESHOLD = 0.50      # below this USD value = ignore entirely


class PortfolioManager:
    def __init__(self, exchange, database, regime_detector, risk_manager, news_intel):
        self.exchange = exchange
        self.db = database
        self.regime = regime_detector
        self.risk = risk_manager
        self.news = news_intel

        # Core state
        self.portfolio_value: float = 0.0
        self.cash_available: float = 0.0
        self.invested_value: float = 0.0
        self.sub_min_value: float = 0.0      # locked in sub-minimum positions
        self.open_positions: list = []        # ACTIVE positions (can trade)
        self.sub_min_positions: list = []     # SUB_MIN positions (tracking only)
        self.sector_exposure: dict = {}
        self.peak_value: float = 0.0
        self.dca_active: bool = False
        self.dca_state: dict = {}
        self._imported_symbols: set = set()

    # ──────────────────────────────────────────────────────────────────────
    # Startup: Full wallet reconciliation
    # ──────────────────────────────────────────────────────────────────────

    async def reconcile_with_wallet(self) -> dict:
        """
        Master startup reconciliation. Does everything in one pass:
        1. Read actual Binance wallet
        2. Classify each holding: ACTIVE / SUB_MIN / GHOST
        3. Clean ghost DB entries
        4. Import untracked wallet positions
        5. Sync quantities for already-tracked positions
        Returns summary dict for Telegram notification.
        """
        logger.info("Reconciling DB with Binance spot wallet...")

        # Get full wallet
        try:
            balances = await self.exchange.get_balances()
        except Exception as e:
            logger.error("Wallet reconciliation failed: %s", e)
            return {"ghosts": 0, "imported": 0, "synced": 0, "sub_min": 0}

        # Classify every non-USDT balance
        wallet_assets = {}   # asset → {qty, value, symbol, price}
        for asset, data in balances.items():
            if asset in STABLECOIN_ASSETS:
                continue
            qty = data.get("total", 0)
            if qty <= 0:
                continue
            symbol = f"{asset}USDT"
            if self.exchange.symbol_filters and symbol not in self.exchange.symbol_filters:
                continue
            try:
                price = await self.exchange.get_price(symbol)
                if price <= 0:
                    continue
                value = qty * price
                if value < DUST_THRESHOLD:
                    continue   # genuine dust, ignore
                wallet_assets[asset] = {
                    "symbol": symbol, "asset": asset,
                    "qty": qty, "price": price, "value": value,
                    "tier": "ACTIVE" if value >= MIN_NOTIONAL else "SUB_MIN",
                }
            except Exception:
                continue

        # Get current DB open trades
        db_open = self.db.get_open_trades()
        db_symbols = {t["symbol"]: t for t in db_open}

        ghosts = synced = imported = sub_min_count = 0

        # Pass 1: Clean ghost DB positions (in DB but asset=0 in wallet)
        for trade in db_open:
            symbol = trade["symbol"]
            asset = symbol.replace("USDT", "").replace("BUSD", "")
            if asset not in wallet_assets:
                # Asset not in wallet at all — genuine ghost
                self.db.update_trade(trade["id"], {
                    "status": "CLOSED_GHOST",
                    "exit_reason": "Reconciliation: asset not found in wallet",
                    "exit_time": utc_now(),
                    "pnl": 0, "pnl_percent": 0,
                })
                logger.info("Ghost closed: %s (not in wallet)", symbol)
                ghosts += 1

        # Pass 2: Sync quantities for existing DB positions
        db_open_fresh = self.db.get_open_trades()
        db_symbols_fresh = {t["symbol"]: t for t in db_open_fresh}

        for asset, wdata in wallet_assets.items():
            symbol = wdata["symbol"]
            if symbol in db_symbols_fresh:
                trade = db_symbols_fresh[symbol]
                db_qty = trade.get("remaining_quantity", trade.get("quantity", 0))
                wallet_qty = wdata["qty"]
                # Sync if more than 5% off
                if db_qty > 0 and abs(wallet_qty - db_qty) / db_qty > 0.05:
                    self.db.update_trade(trade["id"], {
                        "remaining_quantity": wallet_qty,
                    })
                    logger.info("Synced qty %s: DB=%.6f → wallet=%.6f",
                                symbol, db_qty, wallet_qty)
                    synced += 1

        # Pass 3: Import wallet positions not yet in DB
        db_symbols_fresh = {t["symbol"] for t in self.db.get_open_trades()}
        for asset, wdata in wallet_assets.items():
            symbol = wdata["symbol"]
            if symbol in db_symbols_fresh or symbol in self._imported_symbols:
                continue

            tier = wdata["tier"]
            qty = wdata["qty"]
            price = wdata["price"]
            value = wdata["value"]

            # Get avg entry price from trade history
            try:
                avg_entry = await self.exchange.get_avg_entry_price(symbol)
            except Exception:
                avg_entry = 0.0

            if avg_entry <= 0:
                avg_entry = price
                entry_source = "ESTIMATED_CURRENT"
            else:
                entry_source = "TRADE_HISTORY"

            gain_pct = (price / avg_entry - 1) if avg_entry > 0 else 0
            sector = Settings.get_sector_for_asset(symbol)

            # Set SL/TP appropriate for the tier
            if tier == "SUB_MIN":
                # Below min — set wide stop, wait for recovery above $10
                stop_loss = price * 0.85     # 15% stop — don't panic sell dust
                take_profit = price * (1 + max(0.20, abs(gain_pct) + 0.05))
                status = "OPEN_SUB_MIN"
            else:
                # Active position — normal trailing logic
                if gain_pct > 0:
                    stop_loss = max(price * 0.97, avg_entry * 1.002)  # lock some profit
                else:
                    stop_loss = price * 0.95   # 5% emergency stop
                take_profit = price * (1 + max(0.10, abs(gain_pct) + 0.03))
                status = "OPEN"

            trade_data = {
                "symbol": symbol, "side": "BUY",
                "entry_price": avg_entry, "quantity": qty,
                "remaining_quantity": qty, "usdt_value": value,
                "stop_loss": stop_loss, "take_profit": take_profit,
                "highest_price": price, "confluence_score": 0,
                "regime_at_entry": f"INHERITED_{entry_source}_{tier}",
                "status": status, "fees_paid": 0.0,
                "entry_time": utc_now() - timedelta(hours=1),
                "sector": sector, "tranche_exits": "[]",
            }
            trade_id = self.db.save_trade(trade_data)
            trade_data["id"] = trade_id
            self._imported_symbols.add(symbol)
            imported += 1

            logger.info("Imported %s: tier=%s qty=%.6f value=$%.2f gain=%.1f%%",
                        symbol, tier, qty, value, gain_pct * 100)
            if tier == "SUB_MIN":
                sub_min_count += 1

        logger.info("Reconciliation: %d ghosts closed, %d synced, %d imported (%d sub-min)",
                    ghosts, synced, imported, sub_min_count)
        return {"ghosts": ghosts, "synced": synced,
                "imported": imported, "sub_min": sub_min_count}

    # Keep old method names as wrappers for compatibility
    async def clean_ghost_positions(self) -> int:
        result = await self.reconcile_with_wallet()
        return result.get("ghosts", 0)

    async def import_spot_positions(self) -> list:
        # Already handled by reconcile_with_wallet — return empty to avoid double import
        return []

    # ──────────────────────────────────────────────────────────────────────
    # Portfolio Sync
    # ──────────────────────────────────────────────────────────────────────

    async def sync_with_exchange(self):
        balances = await self.exchange.get_balances()

        # Only update cash if API returned real data
        usdt_free = balances.get("USDT", {}).get("free")
        if usdt_free is not None:
            self.cash_available = float(usdt_free)

        # Classify DB positions
        all_open = self.db.get_open_trades()
        self.open_positions = []
        self.sub_min_positions = []

        invested = 0.0
        sub_min = 0.0
        self.sector_exposure = {}

        for pos in all_open:
            symbol = pos["symbol"]
            asset = symbol.replace("USDT", "").replace("BUSD", "")

            # Use actual wallet quantity
            wallet_qty = balances.get(asset, {}).get("total", 0) if balances else 0
            db_qty = pos.get("remaining_quantity", pos.get("quantity", 0))
            actual_qty = min(wallet_qty, db_qty) if wallet_qty > 0 else db_qty

            try:
                price = await self.exchange.get_price(symbol)
                pos_value = actual_qty * price if price > 0 else 0
            except Exception:
                pos_value = pos.get("usdt_value", 0)
                price = 0

            sector = pos.get("sector", "Other")
            self.sector_exposure[sector] = self.sector_exposure.get(sector, 0) + pos_value

            # Classify by current value
            if pos_value >= MIN_NOTIONAL:
                self.open_positions.append(pos)
                invested += pos_value
            elif pos_value >= DUST_THRESHOLD:
                self.sub_min_positions.append(pos)
                sub_min += pos_value
            # else: genuine dust — ignore

        self.invested_value = invested
        self.sub_min_value = sub_min
        self.portfolio_value = self.cash_available + invested + sub_min

        # Peak protection — never use a DB peak > 50% above current
        db_peak = self.db.get_peak_portfolio_value()
        if db_peak > 0 and db_peak < self.portfolio_value * 1.5:
            self.peak_value = max(self.peak_value, self.portfolio_value, db_peak)
        else:
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

        logger.info(
            "Portfolio: $%.2f | cash=$%.2f | active=$%.2f(%dp) | sub-min=$%.2f(%dp) | dd=%.1f%%",
            self.portfolio_value, self.cash_available,
            self.invested_value, len(self.open_positions),
            self.sub_min_value, len(self.sub_min_positions),
            drawdown * 100,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Sub-minimum position manager — wait for recovery then exit
    # ──────────────────────────────────────────────────────────────────────

    async def manage_sub_min_positions(self, notifier=None) -> list:
        """
        For each sub-minimum position:
        - If value crossed $10: sell immediately (finally tradeable)
        - If loss > 40%: accept it as permanent dust, mark closed
        - Otherwise: hold and wait
        Returns list of executed exits.
        """
        exits = []
        for pos in list(self.sub_min_positions):
            symbol = pos["symbol"]
            try:
                price = await self.exchange.get_price(symbol)
                asset = symbol.replace("USDT", "").replace("BUSD", "")
                from config.settings import Settings
                wallet_qty = (await self.exchange.get_balances()).get(asset, {}).get("total", 0)
                if wallet_qty <= 0:
                    continue
                current_value = wallet_qty * price

                # Crossed $10 → can now sell
                if current_value >= MIN_NOTIONAL:
                    logger.info("Sub-min %s recovered to $%.2f → selling", symbol, current_value)
                    result = await self.exchange.place_market_sell(symbol, wallet_qty)
                    if result and "orderId" in result:
                        filled_qty = float(result.get("executedQty", 0))
                        cum_quote = float(result.get("cummulativeQuoteQty", 0))
                        filled_price = cum_quote / filled_qty if filled_qty > 0 else price
                        pnl = (filled_price - pos["entry_price"]) * filled_qty
                        pnl_pct = (filled_price / pos["entry_price"] - 1) * 100

                        self.db.update_trade(pos["id"], {
                            "status": "CLOSED",
                            "exit_price": filled_price,
                            "exit_reason": "SUB_MIN_RECOVERED",
                            "exit_time": utc_now(),
                            "pnl": pnl, "pnl_percent": pnl_pct,
                        })
                        exits.append({
                            "symbol": symbol, "pnl": pnl, "pnl_pct": pnl_pct,
                            "reason": "Sub-min recovered above $10",
                        })
                        if notifier:
                            sign = "🟢" if pnl >= 0 else "🔴"
                            await notifier.send_message(
                                f"{sign} <b>Sub-min exit</b> {symbol}\n"
                                f"Recovered to ${current_value:.2f} → sold\n"
                                f"P&L: <code>${pnl:+.2f} ({pnl_pct:+.1f}%)</code>"
                            )

                # Permanent loss > 50% — accept as dust
                elif pos["entry_price"] > 0 and price / pos["entry_price"] < 0.50:
                    self.db.update_trade(pos["id"], {
                        "status": "CLOSED_DUST",
                        "exit_reason": f"Permanent loss >50% at ${current_value:.2f}",
                        "exit_time": utc_now(),
                        "pnl": -(pos.get("usdt_value", 0) * 0.5),
                        "pnl_percent": -50,
                    })
                    logger.info("Accepted %s as permanent dust (>50%% loss)", symbol)

            except Exception as e:
                logger.debug("Sub-min check error %s: %s", symbol, e)

        return exits

    # ──────────────────────────────────────────────────────────────────────
    # Auto-rotation: tighten stops on wrong-sector positions
    # ──────────────────────────────────────────────────────────────────────

    async def auto_adjust_inherited_stops(self):
        rotation = self.news.get_dominance_strategy()
        if rotation == "NEUTRAL":
            return
        alt_sectors = {"DeFi", "AI", "Gaming", "Meme", "Layer2"}
        for trade in self.open_positions:
            sector = trade.get("sector", "Other")
            if rotation == "BTC_FOCUS" and sector in alt_sectors:
                try:
                    price = await self.exchange.get_price(trade["symbol"])
                    entry = trade["entry_price"]
                    gain = (price / entry - 1) if entry > 0 else 0
                    current_stop = trade.get("stop_loss", 0)
                    if gain > 0.01:
                        new_stop = max(current_stop, price * 0.98,
                                       entry * (1 + gain * 0.70))
                        if new_stop > current_stop + 0.0001:
                            self.db.update_trade(trade["id"], {"stop_loss": new_stop})
                            trade["stop_loss"] = new_stop
                    elif gain < -0.04:
                        emergency = price * 0.98
                        if emergency > current_stop:
                            self.db.update_trade(trade["id"], {"stop_loss": emergency})
                            trade["stop_loss"] = emergency
                except Exception:
                    pass

    # ──────────────────────────────────────────────────────────────────────
    # Rotation gate — automatic, no manual commands
    # ──────────────────────────────────────────────────────────────────────

    def _get_rotation_threshold(self, sector: str) -> int:
        rotation = self.news.get_dominance_strategy()
        base = Settings.strategy.CONFLUENCE_SCORE_THRESHOLD
        if rotation == "BTC_FOCUS":
            return {"Layer1": base-7, "Store_of_Value": base-7, "Other": base,
                    "Exchange": base, "DeFi": base+20, "AI": base+20,
                    "Gaming": base+25, "Layer2": base+15, "Meme": base+30,
                    "Infrastructure": base+10}.get(sector, base)
        elif rotation == "ALTCOIN_SEASON":
            return {"Layer1": base-5, "Store_of_Value": base-5, "DeFi": base-7,
                    "AI": base-7, "Gaming": base-5, "Layer2": base-5,
                    "Exchange": base, "Infrastructure": base-3,
                    "Other": base, "Meme": base+15}.get(sector, base)
        return base

    def _rotation_allows_entry(self, symbol: str, score: int, sector: str) -> tuple:
        rotation = self.news.get_dominance_strategy()
        threshold = self._get_rotation_threshold(sector)
        if score < threshold:
            return False, f"Rotation [{rotation}]: {sector} needs {threshold} (got {score})"
        if rotation == "BTC_FOCUS":
            alt_sectors = {"DeFi", "AI", "Gaming", "Meme", "Layer2"}
            if sector in alt_sectors:
                alt_exp = sum(v for k, v in self.sector_exposure.items() if k in alt_sectors)
                if self.portfolio_value > 0 and alt_exp / self.portfolio_value >= 0.25:
                    return False, "BTC season: alt exposure at 25% max"
        return True, "OK"

    def get_rotation_status(self) -> dict:
        rotation = self.news.get_dominance_strategy()
        alt_sectors = {"DeFi", "AI", "Gaming", "Meme", "Layer2"}
        alt_exp = sum(v for k, v in self.sector_exposure.items() if k in alt_sectors)
        core_exp = sum(v for k, v in self.sector_exposure.items()
                       if k in ("Layer1", "Store_of_Value"))
        return {
            "rotation": rotation,
            "btc_dominance": self.news.btc_dominance,
            "altcoin_season_index": self.news.altcoin_season_index,
            "alt_exposure_pct": alt_exp / self.portfolio_value if self.portfolio_value > 0 else 0,
            "core_exposure_pct": core_exp / self.portfolio_value if self.portfolio_value > 0 else 0,
            "sector_thresholds": {s: self._get_rotation_threshold(s)
                                  for s in ["Layer1", "DeFi", "AI", "Gaming", "Meme"]},
        }

    # ──────────────────────────────────────────────────────────────────────
    # Entry Execution
    # ──────────────────────────────────────────────────────────────────────

    async def execute_entry(self, signal: dict) -> Optional[dict]:
        symbol = signal["symbol"]
        price = signal["price"]
        atr = signal["atr"]
        score = signal.get("confluence_score", 50)
        sector = Settings.get_sector_for_asset(symbol)

        # Admission control
        check = self.risk.can_open_position(
            self.portfolio_value, self.cash_available,
            self.open_positions, sector, self.sector_exposure,
        )
        if not check["allowed"]:
            logger.info("Entry blocked %s: %s", symbol, check["reason"])
            self.db.save_signal({
                "timestamp": utc_now(), "symbol": symbol,
                "confluence_score": score,
                "score_breakdown": json.dumps(signal.get("breakdown", {})),
                "action_taken": f"BLOCKED:{check['reason'][:40]}",
                "regime": self.regime.current_regime,
            })
            return None

        rotation_ok, rotation_reason = self._rotation_allows_entry(symbol, score, sector)
        if not rotation_ok:
            logger.debug("Rotation blocked %s: %s", symbol, rotation_reason)
            return None

        if not await self._check_correlation(symbol):
            logger.info("Correlation blocked %s", symbol)
            return None

        # Skip if already in sub-min for this symbol
        if any(p["symbol"] == symbol for p in self.sub_min_positions):
            logger.debug("Already have sub-min position in %s, skipping", symbol)
            return None

        # Size the position
        position_size = self.risk.calculate_position_size(
            self.portfolio_value, atr, price, score
        )
        rotation = self.news.get_dominance_strategy()
        if rotation == "BTC_FOCUS" and sector not in ("Layer1", "Store_of_Value", "Other"):
            position_size *= 0.6
        elif rotation == "ALTCOIN_SEASON" and sector in ("DeFi", "AI", "Gaming"):
            position_size *= 1.15
        position_size = min(position_size, self.cash_available * 0.95)

        min_notional = self.exchange.get_min_notional(symbol)
        if position_size < max(min_notional, 10.0):
            logger.info("Position too small %s: $%.2f", symbol, position_size)
            return None

        result = await self.exchange.place_market_buy(symbol, position_size)
        if not result or "orderId" not in result:
            return None

        filled_qty = float(result.get("executedQty", 0))
        if filled_qty <= 0:
            return None
        cum_quote = float(result.get("cummulativeQuoteQty", 0))
        filled_price = cum_quote / filled_qty if filled_qty > 0 else price
        fees = sum(float(f.get("commission", 0)) for f in result.get("fills", []))

        stop_loss = self.risk.calculate_stop_loss(filled_price, atr)
        tp = self.risk.calculate_take_profit(filled_price, atr)

        trade_data = {
            "symbol": symbol, "side": "BUY",
            "entry_price": filled_price, "quantity": filled_qty,
            "usdt_value": position_size, "stop_loss": stop_loss,
            "take_profit": tp["tp1"], "highest_price": filled_price,
            "remaining_quantity": filled_qty, "confluence_score": score,
            "regime_at_entry": f"{self.regime.current_regime}|{rotation}",
            "status": "OPEN", "fees_paid": fees,
            "entry_time": utc_now(), "sector": sector, "tranche_exits": "[]",
        }
        trade_id = self.db.save_trade(trade_data)
        trade_data["id"] = trade_id

        self.db.save_signal({
            "timestamp": utc_now(), "symbol": symbol,
            "confluence_score": score,
            "score_breakdown": json.dumps(signal.get("breakdown", {})),
            "action_taken": "ENTERED", "regime": self.regime.current_regime,
        })

        self.cash_available -= position_size
        self.invested_value += position_size
        self.open_positions.append(trade_data)
        self.sector_exposure[sector] = self.sector_exposure.get(sector, 0) + position_size

        try:
            stop_limit = stop_loss * 0.998
            await self.exchange.place_oco_sell(
                symbol, filled_qty, tp["tp1"], stop_loss, stop_limit
            )
        except Exception as e:
            logger.warning("OCO failed %s: %s", symbol, e)

        logger.info("ENTRY: %s @ $%.4f size=$%.2f SL=$%.4f TP=$%.4f score=%d",
                    symbol, filled_price, position_size, stop_loss, tp["tp1"], score)
        return trade_data

    # ──────────────────────────────────────────────────────────────────────
    # Exit Execution
    # ──────────────────────────────────────────────────────────────────────

    async def execute_exit(self, trade: dict, quantity: float,
                           reason: str, current_price: float) -> Optional[dict]:
        symbol = trade["symbol"]
        qty = min(quantity, trade.get("remaining_quantity", trade["quantity"]))
        if qty <= 0:
            return None

        # Check value meets notional minimum
        estimated_value = qty * current_price
        if estimated_value < MIN_NOTIONAL * 0.9:
            # Move to sub-min tracking instead of trying to sell
            logger.info("Exit blocked %s: value $%.2f below min notional — tracking as sub-min",
                        symbol, estimated_value)
            self.db.update_trade(trade["id"], {"status": "OPEN_SUB_MIN"})
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
        pnl_pct = (filled_price / trade["entry_price"] - 1) * 100
        is_full_exit = remaining <= 0.0001

        updates = {"remaining_quantity": max(remaining, 0),
                   "fees_paid": trade.get("fees_paid", 0) + fees}
        if is_full_exit:
            updates.update({
                "status": "CLOSED" if pnl >= 0 else "STOPPED_OUT",
                "exit_price": filled_price, "pnl": pnl,
                "pnl_percent": pnl_pct, "exit_time": utc_now(),
                "exit_reason": reason,
            })
            self.risk.record_trade_result(won=pnl > 0)
        else:
            exits = (json.loads(trade.get("tranche_exits", "[]"))
                     if isinstance(trade.get("tranche_exits"), str)
                     else trade.get("tranche_exits", []))
            exits.append({"level": len(exits)+1, "price": filled_price,
                          "quantity": filled_qty, "pnl": pnl, "reason": reason,
                          "time": utc_now().isoformat()})
            updates["tranche_exits"] = json.dumps(exits)

        self.db.update_trade(trade["id"], updates)

        if is_full_exit:
            self.open_positions = [p for p in self.open_positions if p["id"] != trade["id"]]
        else:
            trade["remaining_quantity"] = max(remaining, 0)

        logger.info("EXIT: %s @ $%.4f qty=%.6f P&L=$%.2f (%.2f%%) %s",
                    symbol, filled_price, filled_qty, pnl, pnl_pct, reason)
        return {"symbol": symbol, "exit_price": filled_price, "quantity": filled_qty,
                "pnl": pnl, "pnl_pct": pnl_pct, "reason": reason, "is_full_exit": is_full_exit}

    async def force_sell(self, symbol: str) -> Optional[dict]:
        for trade in self.open_positions:
            if trade["symbol"] == symbol:
                price = await self.exchange.get_price(symbol)
                return await self.execute_exit(
                    trade, trade.get("remaining_quantity", trade["quantity"]), "MANUAL", price)
        return None

    async def force_sell_all(self) -> list:
        results = []
        for trade in list(self.open_positions):
            price = await self.exchange.get_price(trade["symbol"])
            r = await self.execute_exit(
                trade, trade.get("remaining_quantity", trade["quantity"]), "MANUAL", price)
            if r:
                results.append(r)
        return results

    async def liquidate_all(self, reason: str = "KILL_SWITCH") -> list:
        results = []
        for trade in list(self.open_positions):
            try:
                price = await self.exchange.get_price(trade["symbol"])
                if price and price > 0:
                    r = await self.execute_exit(
                        trade, trade.get("remaining_quantity", trade["quantity"]), reason, price)
                    if r:
                        results.append(r)
                    await asyncio.sleep(0.5)
            except Exception as e:
                logger.error("Liquidation error %s: %s", trade["symbol"], e)
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
                    return False
        except Exception:
            pass
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
            self.dca_state = {"budget": dca_budget, "tranche_size": tranche_size,
                              "tranches_remaining": cfg.DCA_TRANCHES,
                              "symbols": ["BTCUSDT", "ETHUSDT"]}
            self.dca_active = True
            logger.info("Smart DCA: $%.2f in %d tranches (F&G=%d)",
                        dca_budget, cfg.DCA_TRANCHES, self.news.fear_greed_value)
        if self.dca_state["tranches_remaining"] > 0:
            tranche = self.dca_state["tranche_size"]
            per_asset = tranche / len(self.dca_state["symbols"])
            for symbol in self.dca_state["symbols"]:
                if per_asset >= 10 and per_asset <= self.cash_available * 0.5:
                    result = await self.exchange.place_market_buy(symbol, per_asset)
                    if result and "orderId" in result:
                        logger.info("DCA %s $%.2f (F&G=%d)", symbol, per_asset,
                                    self.news.fear_greed_value)
            self.dca_state["tranches_remaining"] -= 1

    async def check_rebalancing(self) -> list:
        actions = []
        for pos in self.open_positions:
            try:
                price = await self.exchange.get_price(pos["symbol"])
                pos_value = pos.get("remaining_quantity", pos["quantity"]) * price
                if (self.portfolio_value > 0
                        and pos_value / self.portfolio_value
                        > Settings.portfolio_cfg.MAX_SINGLE_POSITION_PCT):
                    excess = pos_value / self.portfolio_value - Settings.portfolio_cfg.MAX_SINGLE_POSITION_PCT
                    sell_qty = (excess * self.portfolio_value) / price if price > 0 else 0
                    if sell_qty > 0:
                        actions.append({"symbol": pos["symbol"], "quantity": sell_qty,
                                        "reason": "Drift rebalance"})
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
            "sub_min_value": self.sub_min_value,
            "open_positions": len(self.open_positions),
            "sub_min_positions": len(self.sub_min_positions),
            "sector_exposure": self.sector_exposure,
            "peak_value": self.peak_value,
            "drawdown": ((self.peak_value - self.portfolio_value) / self.peak_value
                         if self.peak_value > 0 else 0),
            "dca_active": self.dca_active,
            "mode": self.regime.current_mode,
            "rotation": self.news.get_dominance_strategy(),
        }
