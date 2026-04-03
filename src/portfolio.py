"""
Portfolio Manager — Core-satellite model with sector rotation.

Architecture:
  Core (50%):        BTC + ETH — always some exposure
  Satellite (35%):   Quality alts with momentum signals
  Speculative (5%):  High-risk/high-reward opportunities
  Dry Powder (10%):  Cash for DCA on fear spikes

Capital preservation:
  - CAPITAL_PRESERVATION mode → 80% stablecoins, DCA only
  - DEFENSIVE mode → 35% deployed max
  - Anti-martingale scaling via risk_manager
  - Smart DCA: 3× base amount during extreme fear (F&G < 15)

Sector rotation:
  - BTC Dom > 65% → BTC_FOCUS (80-90% BTC/ETH)
  - BTC Dom < 50% → ALTCOIN_SEASON (rotate to quality alts)
"""
import asyncio
import json
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Optional

from config.settings import Settings
from src.utils import setup_logging, utc_now, format_pnl, format_pct

logger = setup_logging()


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
        self._last_sync_time: float = 0.0

    async def sync_with_exchange(self):
        balances = await self.exchange.get_balances()
        self.cash_available = balances.get("USDT", {}).get("free", 0.0)
        self.open_positions = self.db.get_open_trades()

        invested = 0.0
        self.sector_exposure = {}
        for pos in self.open_positions:
            try:
                current_price = await self.exchange.get_price(pos["symbol"])
                pos_value = (pos["remaining_quantity"] * current_price
                             if current_price > 0 else pos["usdt_value"])
                invested += pos_value
                sector = pos.get("sector", "Other")
                self.sector_exposure[sector] = self.sector_exposure.get(sector, 0) + pos_value
            except Exception:
                invested += pos.get("usdt_value", 0)

        self.invested_value = invested
        self.portfolio_value = self.cash_available + invested
        self.peak_value = max(self.peak_value, self.portfolio_value,
                              self.db.get_peak_portfolio_value())

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
        logger.info("Portfolio: $%.2f (cash=$%.2f invested=$%.2f pos=%d)",
                    self.portfolio_value, self.cash_available,
                    self.invested_value, len(self.open_positions))

    async def execute_entry(self, signal: dict) -> Optional[dict]:
        symbol = signal["symbol"]
        price = signal["price"]
        atr = signal["atr"]
        score = signal.get("confluence_score", 50)
        sector = Settings.get_sector_for_asset(symbol)

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

        if not await self._check_correlation(symbol):
            logger.info("Entry blocked for %s: high correlation", symbol)
            return None

        # Validate against sector rotation advice
        rotation = self.news.get_dominance_strategy()
        if rotation == "BTC_FOCUS" and sector not in ("Layer1", "Store_of_Value", "Other"):
            btc_exposed = sum(v for k, v in self.sector_exposure.items()
                              if k in ("Layer1", "Store_of_Value"))
            if btc_exposed / self.portfolio_value < 0.5 if self.portfolio_value > 0 else False:
                pass  # allow — portfolio not yet in BTC focus
            elif score < 65:  # only allow alts with very high conviction in BTC season
                logger.info("Skipping %s (BTC focus mode, score %d < 65)", symbol, score)
                return None

        position_size = self.risk.calculate_position_size(
            self.portfolio_value, atr, price, score
        )
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
            "regime_at_entry": self.regime.current_regime,
            "status": "OPEN", "fees_paid": fees,
            "entry_time": utc_now(), "sector": sector,
            "highest_price": filled_price,
            "remaining_quantity": filled_qty,
            "tranche_exits": "[]",
        }
        trade_id = self.db.save_trade(trade_data)
        trade_data["id"] = trade_id

        # Record for anti-martingale (will be updated on exit)
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

        # Place OCO for exchange-level SL/TP protection
        try:
            stop_limit = stop_loss * (1 - 0.002)
            oco = await self.exchange.place_oco_sell(
                symbol, filled_qty, take_profit, stop_loss, stop_limit
            )
            if oco and ("orderListId" in oco or "orders" in oco):
                logger.info("OCO placed for %s: TP=$%.4f SL=$%.4f", symbol, take_profit, stop_loss)
        except Exception as e:
            logger.warning("OCO failed for %s (bot will monitor): %s", symbol, e)

        logger.info("ENTRY: %s @ $%.4f qty=%.6f size=$%.2f SL=$%.4f score=%d mode=%s",
                    symbol, filled_price, filled_qty, position_size, stop_loss,
                    score, self.regime.current_mode)
        return trade_data

    async def execute_exit(self, trade: dict, quantity: float,
                           reason: str, current_price: float) -> Optional[dict]:
        symbol = trade["symbol"]
        qty = min(quantity, trade.get("remaining_quantity", trade["quantity"]))
        if qty <= 0:
            return None

        # Cancel open orders before selling
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
                "status": "CLOSED" if pnl >= 0 else "STOPPED_OUT",
                "exit_price": filled_price,
                "pnl": total_pnl,
                "pnl_percent": total_pnl_pct,
                "exit_time": utc_now(),
                "exit_reason": reason,
            })
            # Update anti-martingale
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
        """Reject if correlation > 0.95 with existing position."""
        if not self.open_positions:
            return True
        try:
            new_klines = await self.exchange.get_klines(new_symbol, "1d", 30)
            if len(new_klines) < 20:
                return True
            new_returns = np.diff([k["close"] for k in new_klines]) / [k["close"] for k in new_klines[:-1]]
            for pos in self.open_positions[:5]:  # check against first 5 only
                pos_klines = await self.exchange.get_klines(pos["symbol"], "1d", 30)
                if len(pos_klines) < 20:
                    continue
                pos_returns = np.diff([k["close"] for k in pos_klines]) / [k["close"] for k in pos_klines[:-1]]
                min_len = min(len(new_returns), len(pos_returns))
                if min_len < 10:
                    continue
                corr = np.corrcoef(new_returns[-min_len:], pos_returns[-min_len:])[0, 1]
                if abs(corr) > 0.95:
                    logger.info("High correlation %.2f: %s ↔ %s", corr, new_symbol, pos["symbol"])
                    return False
        except Exception as e:
            logger.warning("Correlation check error: %s", e)
        return True

    async def smart_dca(self):
        """DCA during extreme fear (F&G < 15) with 3× normal amount."""
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
                "budget": dca_budget,
                "tranche_size": tranche_size,
                "tranches_remaining": cfg.DCA_TRANCHES,
                "symbols": ["BTCUSDT", "ETHUSDT"],
                "started": utc_now().isoformat(),
            }
            self.dca_active = True
            logger.info("Smart DCA activated (F&G=%d): $%.2f in %d tranches of $%.2f",
                        self.news.fear_greed_value, dca_budget,
                        cfg.DCA_TRANCHES, tranche_size)

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
                            "remaining_quantity": filled_qty,
                            "tranche_exits": "[]",
                        }
                        self.db.save_trade(trade_data)
                        logger.info("DCA: %s $%.2f @ $%.4f (F&G=%d)",
                                    symbol, per_asset, filled_price, self.news.fear_greed_value)

            self.dca_state["tranches_remaining"] -= 1

    async def check_rebalancing(self) -> list:
        """Trim positions that exceed 15% of portfolio (drift rebalance)."""
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
                            "type": "TRIM",
                            "symbol": pos["symbol"],
                            "quantity": sell_qty,
                            "reason": f"Exceeds {cfg.MAX_SINGLE_POSITION_PCT:.0%} max position",
                        })
                gain_pct = ((current_price / pos["entry_price"]) - 1) * 100 if pos["entry_price"] > 0 else 0
                if gain_pct > 50:
                    sell_qty = pos.get("remaining_quantity", pos["quantity"]) * 0.25
                    actions.append({
                        "type": "PROFIT_TAKE",
                        "symbol": pos["symbol"],
                        "quantity": sell_qty,
                        "reason": f"Profit taking at {gain_pct:.0f}% gain",
                    })
            except Exception:
                pass
        return actions

    async def execute_rebalancing(self, actions: list) -> list:
        results = []
        for action in actions:
            symbol = action["symbol"]
            trade = next((t for t in self.open_positions if t["symbol"] == symbol), None)
            if not trade:
                continue
            price = await self.exchange.get_price(symbol)
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
        }
