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

    async def sync_with_exchange(self):
        balances = await self.exchange.get_balances()
        self.cash_available = balances.get("USDT", {}).get("free", 0.0)
        self.open_positions = self.db.get_open_trades()

        invested = 0.0
        self.sector_exposure = {}
        for pos in self.open_positions:
            current_price = await self.exchange.get_price(pos["symbol"])
            pos_value = pos["remaining_quantity"] * current_price if current_price > 0 else pos["usdt_value"]
            invested += pos_value
            sector = pos.get("sector", "Other")
            self.sector_exposure[sector] = self.sector_exposure.get(sector, 0) + pos_value

        self.invested_value = invested
        self.portfolio_value = self.cash_available + invested
        self.peak_value = max(self.peak_value, self.portfolio_value, self.db.get_peak_portfolio_value())

        drawdown = (self.peak_value - self.portfolio_value) / self.peak_value if self.peak_value > 0 else 0

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
            "Portfolio synced: total=$%.2f, cash=$%.2f, invested=$%.2f, positions=%d",
            self.portfolio_value, self.cash_available, self.invested_value,
            len(self.open_positions),
        )

    async def execute_entry(self, signal: dict) -> Optional[dict]:
        symbol = signal["symbol"]
        price = signal["price"]
        atr = signal["atr"]
        sector = Settings.get_sector_for_asset(symbol)

        check = self.risk.can_open_position(
            self.portfolio_value, self.cash_available,
            len(self.open_positions), sector, self.sector_exposure,
        )
        if not check["allowed"]:
            logger.info("Entry blocked for %s: %s", symbol, check["reason"])
            self.db.save_signal({
                "timestamp": utc_now(), "symbol": symbol,
                "confluence_score": signal["confluence_score"],
                "score_breakdown": json.dumps(signal.get("breakdown", {})),
                "action_taken": f"SKIPPED_{check['reason'].upper().replace(' ', '_')[:30]}",
                "regime": self.regime.current_regime,
            })
            return None

        if not await self._check_correlation(symbol):
            logger.info("Entry blocked for %s: high correlation with existing position", symbol)
            return None

        position_size = self.risk.calculate_position_size(
            self.portfolio_value, atr, price
        )
        position_size = min(position_size, self.cash_available * 0.95)

        if position_size < 20:
            logger.info("Position size too small for %s: $%.2f", symbol, position_size)
            return None

        result = await self.exchange.place_market_buy(symbol, position_size)
        if not result or "orderId" not in result:
            logger.error("Failed to execute buy for %s", symbol)
            return None

        filled_qty = float(result.get("executedQty", 0))
        filled_price = float(result.get("cummulativeQuoteQty", 0)) / filled_qty if filled_qty > 0 else price
        fees = sum(float(f.get("commission", 0)) for f in result.get("fills", []))

        stop_loss = self.risk.calculate_stop_loss(filled_price, atr)

        trade_data = {
            "symbol": symbol, "side": "BUY",
            "entry_price": filled_price, "quantity": filled_qty,
            "usdt_value": position_size, "stop_loss": stop_loss,
            "take_profit": filled_price + (atr * 5.0),
            "confluence_score": signal["confluence_score"],
            "regime_at_entry": self.regime.current_regime,
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
            "confluence_score": signal["confluence_score"],
            "score_breakdown": json.dumps(signal.get("breakdown", {})),
            "action_taken": "ENTERED",
            "regime": self.regime.current_regime,
        })

        self.cash_available -= position_size
        self.invested_value += position_size
        self.open_positions.append(trade_data)
        self.sector_exposure[sector] = self.sector_exposure.get(sector, 0) + position_size

        logger.info(
            "ENTRY: %s @ $%.4f, qty=%.6f, size=$%.2f, SL=$%.4f, score=%d",
            symbol, filled_price, filled_qty, position_size, stop_loss,
            signal["confluence_score"],
        )
        return trade_data

    async def execute_exit(self, trade: dict, quantity: float,
                           reason: str, current_price: float) -> Optional[dict]:
        symbol = trade["symbol"]
        qty = min(quantity, trade.get("remaining_quantity", trade["quantity"]))

        if qty <= 0:
            return None

        result = await self.exchange.place_market_sell(symbol, qty)
        if not result or "orderId" not in result:
            logger.error("Failed to execute sell for %s", symbol)
            return None

        filled_qty = float(result.get("executedQty", 0))
        filled_price = float(result.get("cummulativeQuoteQty", 0)) / filled_qty if filled_qty > 0 else current_price
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
        else:
            exits = json.loads(trade.get("tranche_exits", "[]")) if isinstance(trade.get("tranche_exits"), str) else trade.get("tranche_exits", [])
            level_num = len(exits) + 1
            exits.append({
                "level": level_num,
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
            if isinstance(updates.get("tranche_exits"), str):
                trade["tranche_exits"] = updates["tranche_exits"]

        logger.info(
            "EXIT: %s @ $%.4f, qty=%.6f, PnL=$%.2f (%.2f%%), reason=%s",
            symbol, filled_price, filled_qty, pnl, pnl_pct, reason,
        )

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
            result = await self.execute_exit(
                trade, trade.get("remaining_quantity", trade["quantity"]),
                "MANUAL", price,
            )
            if result:
                results.append(result)
        return results

    async def liquidate_all(self, reason: str = "KILL_SWITCH") -> list:
        results = []
        for trade in list(self.open_positions):
            price = await self.exchange.get_price(trade["symbol"])
            result = await self.execute_exit(
                trade, trade.get("remaining_quantity", trade["quantity"]),
                reason, price,
            )
            if result:
                results.append(result)
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
            if len(new_klines) < 20:
                return True
            new_returns = np.diff([k["close"] for k in new_klines]) / [k["close"] for k in new_klines[:-1]]

            for pos in self.open_positions:
                pos_klines = await self.exchange.get_klines(pos["symbol"], "1d", 30)
                if len(pos_klines) < 20:
                    continue
                pos_returns = np.diff([k["close"] for k in pos_klines]) / [k["close"] for k in pos_klines[:-1]]
                min_len = min(len(new_returns), len(pos_returns))
                if min_len < 10:
                    continue
                corr = np.corrcoef(new_returns[-min_len:], pos_returns[-min_len:])[0, 1]
                if abs(corr) > 0.85:
                    logger.info(
                        "High correlation (%.2f) between %s and %s",
                        corr, new_symbol, pos["symbol"],
                    )
                    return False
        except Exception as e:
            logger.warning("Correlation check failed: %s", e)
        return True

    async def check_rebalancing(self) -> list:
        actions = []
        for pos in self.open_positions:
            current_price = await self.exchange.get_price(pos["symbol"])
            pos_value = pos.get("remaining_quantity", pos["quantity"]) * current_price
            if self.portfolio_value > 0 and pos_value / self.portfolio_value > 0.15:
                excess_pct = (pos_value / self.portfolio_value) - 0.15
                sell_value = excess_pct * self.portfolio_value
                sell_qty = sell_value / current_price if current_price > 0 else 0
                if sell_qty > 0:
                    actions.append({
                        "type": "TRIM",
                        "symbol": pos["symbol"],
                        "quantity": sell_qty,
                        "reason": "Position exceeds 15% of portfolio",
                    })

            gain_pct = ((current_price / pos["entry_price"]) - 1) * 100 if pos["entry_price"] > 0 else 0
            if gain_pct > 50:
                sell_qty = pos.get("remaining_quantity", pos["quantity"]) * 0.25
                actions.append({
                    "type": "PROFIT_TAKE",
                    "symbol": pos["symbol"],
                    "quantity": sell_qty,
                    "reason": f"Profit taking at {gain_pct:.1f}% gain",
                })

        for pos in self.open_positions:
            for other in self.open_positions:
                if pos["symbol"] >= other["symbol"]:
                    continue
                try:
                    p1 = await self.exchange.get_klines(pos["symbol"], "1d", 30)
                    p2 = await self.exchange.get_klines(other["symbol"], "1d", 30)
                    if len(p1) < 20 or len(p2) < 20:
                        continue
                    r1 = np.diff([k["close"] for k in p1]) / [k["close"] for k in p1[:-1]]
                    r2 = np.diff([k["close"] for k in p2]) / [k["close"] for k in p2[:-1]]
                    ml = min(len(r1), len(r2))
                    corr = np.corrcoef(r1[-ml:], r2[-ml:])[0, 1]
                    if abs(corr) > 0.85:
                        smaller = pos if pos["usdt_value"] < other["usdt_value"] else other
                        sell_qty = smaller.get("remaining_quantity", smaller["quantity"]) * 0.5
                        actions.append({
                            "type": "DECORRELATE",
                            "symbol": smaller["symbol"],
                            "quantity": sell_qty,
                            "reason": f"High correlation ({corr:.2f}) with {other['symbol'] if smaller == pos else pos['symbol']}",
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
            result = await self.execute_exit(
                trade, action["quantity"], "REBALANCE", price
            )
            if result:
                results.append(result)
        return results

    async def smart_dca(self):
        if not self.news.is_extreme_fear():
            self.dca_active = False
            return

        if self.dca_active and self.dca_state.get("tranches_remaining", 0) <= 0:
            self.dca_active = False
            return

        if not self.dca_active:
            dca_budget = self.portfolio_value * 0.05
            tranche_size = dca_budget / 5
            self.dca_state = {
                "budget": dca_budget,
                "tranche_size": tranche_size,
                "tranches_remaining": 5,
                "symbols": ["BTCUSDT", "ETHUSDT"],
                "started": utc_now().isoformat(),
            }
            self.dca_active = True
            logger.info("Smart DCA activated: budget=$%.2f, 5 tranches of $%.2f",
                        dca_budget, tranche_size)

        if self.dca_state["tranches_remaining"] > 0:
            tranche = self.dca_state["tranche_size"]
            per_asset = tranche / len(self.dca_state["symbols"])
            for symbol in self.dca_state["symbols"]:
                if per_asset >= 20 and per_asset <= self.cash_available * 0.5:
                    result = await self.exchange.place_market_buy(symbol, per_asset)
                    if result and "orderId" in result:
                        filled_qty = float(result.get("executedQty", 0))
                        filled_price = float(result.get("cummulativeQuoteQty", 0)) / filled_qty if filled_qty > 0 else 0
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
                            "take_profit": filled_price + (atr * 5.0),
                            "confluence_score": 0, "regime_at_entry": "DCA",
                            "status": "OPEN", "fees_paid": 0,
                            "entry_time": utc_now(),
                            "sector": Settings.get_sector_for_asset(symbol),
                            "highest_price": filled_price,
                            "remaining_quantity": filled_qty,
                            "tranche_exits": "[]",
                        }
                        self.db.save_trade(trade_data)
                        logger.info("DCA buy: %s $%.2f @ $%.4f", symbol, per_asset, filled_price)

            self.dca_state["tranches_remaining"] -= 1

    def get_status(self) -> dict:
        return {
            "portfolio_value": self.portfolio_value,
            "cash_available": self.cash_available,
            "invested_value": self.invested_value,
            "open_positions": len(self.open_positions),
            "sector_exposure": self.sector_exposure,
            "peak_value": self.peak_value,
            "drawdown": (self.peak_value - self.portfolio_value) / self.peak_value if self.peak_value > 0 else 0,
            "dca_active": self.dca_active,
        }
