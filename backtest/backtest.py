import asyncio
import json
import sys
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.indicators import IndicatorSet, detect_swing_points, cluster_levels
from src.utils import setup_logging

logger = setup_logging()


class BacktestEngine:
    def __init__(self, initial_capital: float = 10000, max_positions: int = 8,
                 risk_per_trade: float = 0.02, atr_stop_mult: float = 2.5,
                 confluence_threshold: int = 70):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.max_positions = max_positions
        self.risk_per_trade = risk_per_trade
        self.atr_stop_mult = atr_stop_mult
        self.confluence_threshold = confluence_threshold

        self.positions: list = []
        self.closed_trades: list = []
        self.equity_curve: list = []
        self.peak_equity: float = initial_capital

    def run(self, data: dict, start_date: str = None, end_date: str = None):
        logger.info("Starting backtest with $%.2f capital", self.initial_capital)

        btc_data = data.get("BTCUSDT")
        if btc_data is None:
            logger.error("BTCUSDT data required for regime detection")
            return

        all_dates = sorted(btc_data.keys())
        if start_date:
            all_dates = [d for d in all_dates if d >= start_date]
        if end_date:
            all_dates = [d for d in all_dates if d <= end_date]

        btc_indicators = IndicatorSet()
        symbol_indicators = {}
        for symbol in data:
            symbol_indicators[symbol] = IndicatorSet()

        warmup = min(210, len(all_dates) // 3)
        for i, date in enumerate(all_dates):
            for symbol, candles in data.items():
                if date in candles:
                    symbol_indicators[symbol].update_candle(candles[date])

            if i < warmup:
                continue

            btc_candle = btc_data.get(date)
            if not btc_candle:
                continue
            btc_ind = symbol_indicators["BTCUSDT"].latest

            regime = self._detect_regime_simple(btc_ind)

            self._check_exits(data, date, symbol_indicators)

            if regime in ("BULL", "SIDEWAYS") and len(self.positions) < self.max_positions:
                for symbol in data:
                    if symbol == "BTCUSDT" and len(data) > 3:
                        continue
                    if any(p["symbol"] == symbol for p in self.positions):
                        continue
                    if date not in data[symbol]:
                        continue

                    ind = symbol_indicators[symbol].latest
                    if not ind:
                        continue

                    score = self._score_signal(ind, regime)
                    if score >= self.confluence_threshold:
                        self._open_position(symbol, ind, date, score, regime)

            total_value = self.capital
            for pos in self.positions:
                candle = data[pos["symbol"]].get(date)
                if candle:
                    total_value += pos["quantity"] * candle["close"]
            self.equity_curve.append({"date": date, "equity": total_value})
            self.peak_equity = max(self.peak_equity, total_value)

        for pos in list(self.positions):
            last_date = all_dates[-1]
            candle = data[pos["symbol"]].get(last_date)
            if candle:
                self._close_position(pos, candle["close"], last_date, "END_OF_TEST")

        return self._generate_report()

    def _detect_regime_simple(self, btc_ind: dict) -> str:
        if not btc_ind:
            return "SIDEWAYS"
        close = btc_ind.get("close", 0)
        sma50 = btc_ind.get("sma50", 0)
        sma200 = btc_ind.get("sma200", 0)
        adx = btc_ind.get("adx", 0)

        if sma50 == 0 or sma200 == 0:
            return "SIDEWAYS"

        if close > sma50 and close > sma200 and adx > 20:
            return "BULL"
        elif close < sma50 and close < sma200 and adx > 20:
            return "BEAR"
        return "SIDEWAYS"

    def _score_signal(self, ind: dict, regime: str) -> int:
        score = 0

        rsi = ind.get("rsi", 50)
        if 30 <= rsi <= 45:
            score += 10
        elif 45 < rsi <= 55:
            score += 5

        macd_hist = ind.get("macd_histogram", 0)
        macd_line = ind.get("macd", 0)
        macd_signal = ind.get("macd_signal", 0)
        if macd_line > macd_signal and macd_hist > 0:
            score += 10
        elif macd_hist > 0:
            score += 5

        price = ind.get("close", 0)
        ema9 = ind.get("ema9", 0)
        ema21 = ind.get("ema21", 0)
        if price > ema9 > ema21 and ema9 > 0:
            score += 10
        elif price > ema21 and ema21 > 0:
            score += 5

        volume = ind.get("volume", 0)
        vol_sma = ind.get("volume_sma", 1)
        if vol_sma > 0:
            ratio = volume / vol_sma
            if ratio > 2.0:
                score += 15
            elif ratio > 1.5:
                score += 10
            elif ratio > 1.2:
                score += 5

        if regime == "BULL":
            score += 20
        elif regime == "SIDEWAYS":
            score += 10

        adx = ind.get("adx", 0)
        if adx > 25:
            score += 5

        bb_lower = ind.get("bb_lower", 0)
        if bb_lower > 0 and price < bb_lower * 1.02:
            score += 10

        return min(score, 100)

    def _open_position(self, symbol: str, ind: dict, date: str,
                       score: int, regime: str):
        price = ind["close"]
        atr = ind.get("atr", price * 0.03)
        stop_loss = price - (atr * self.atr_stop_mult)

        risk_amount = self.capital * self.risk_per_trade
        stop_distance = price - stop_loss
        if stop_distance <= 0:
            return

        position_size = risk_amount / (stop_distance / price)
        position_size = min(position_size, self.capital * 0.12)
        position_size = min(position_size, self.capital * 0.95)

        if position_size < 20:
            return

        quantity = position_size / price
        self.capital -= position_size

        self.positions.append({
            "symbol": symbol,
            "entry_price": price,
            "quantity": quantity,
            "usdt_value": position_size,
            "stop_loss": stop_loss,
            "highest_price": price,
            "entry_date": date,
            "score": score,
            "regime": regime,
            "atr": atr,
        })

    def _check_exits(self, data: dict, date: str, indicators: dict):
        for pos in list(self.positions):
            candle = data[pos["symbol"]].get(date)
            if not candle:
                continue

            current_price = candle["close"]
            high = candle["high"]
            low = candle["low"]

            if low <= pos["stop_loss"]:
                self._close_position(pos, pos["stop_loss"], date, "STOP_LOSS")
                continue

            pos["highest_price"] = max(pos["highest_price"], high)
            atr = pos.get("atr", current_price * 0.03)
            new_stop = pos["highest_price"] - (atr * self.atr_stop_mult)
            pos["stop_loss"] = max(pos["stop_loss"], new_stop)

            profit_pct = (current_price / pos["entry_price"]) - 1
            if profit_pct >= 1.5 * self.atr_stop_mult * (atr / pos["entry_price"]):
                pos["stop_loss"] = max(pos["stop_loss"], pos["entry_price"] * 1.01)

            tp1 = pos["entry_price"] + (atr * 2.0)
            if current_price >= tp1 and pos.get("tp1_hit") is None:
                sell_qty = pos["quantity"] * 0.25
                sell_value = sell_qty * current_price
                self.capital += sell_value
                pos["quantity"] -= sell_qty
                pos["tp1_hit"] = True

            tp2 = pos["entry_price"] + (atr * 3.5)
            if current_price >= tp2 and pos.get("tp2_hit") is None:
                sell_qty = pos["quantity"] * 0.33
                sell_value = sell_qty * current_price
                self.capital += sell_value
                pos["quantity"] -= sell_qty
                pos["tp2_hit"] = True

    def _close_position(self, pos: dict, exit_price: float, date: str, reason: str):
        value = pos["quantity"] * exit_price
        self.capital += value
        pnl = value - (pos["quantity"] * pos["entry_price"])
        pnl_pct = ((exit_price / pos["entry_price"]) - 1) * 100

        self.closed_trades.append({
            "symbol": pos["symbol"],
            "entry_price": pos["entry_price"],
            "exit_price": exit_price,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "entry_date": pos["entry_date"],
            "exit_date": date,
            "reason": reason,
            "score": pos["score"],
        })
        self.positions.remove(pos)

    def _generate_report(self) -> dict:
        if not self.closed_trades:
            return {"error": "No trades executed"}

        wins = [t for t in self.closed_trades if t["pnl"] > 0]
        losses = [t for t in self.closed_trades if t["pnl"] <= 0]

        total_pnl = sum(t["pnl"] for t in self.closed_trades)
        win_rate = len(wins) / len(self.closed_trades) if self.closed_trades else 0
        avg_win = np.mean([t["pnl_pct"] for t in wins]) if wins else 0
        avg_loss = np.mean([abs(t["pnl_pct"]) for t in losses]) if losses else 0
        gross_profit = sum(t["pnl"] for t in wins)
        gross_loss = abs(sum(t["pnl"] for t in losses)) or 1
        profit_factor = gross_profit / gross_loss

        equities = [e["equity"] for e in self.equity_curve]
        peak = equities[0]
        max_dd = 0
        for eq in equities:
            peak = max(peak, eq)
            dd = (peak - eq) / peak
            max_dd = max(max_dd, dd)

        final_equity = equities[-1] if equities else self.capital
        total_return = ((final_equity / self.initial_capital) - 1) * 100

        days = len(self.equity_curve)
        if days > 365:
            cagr = ((final_equity / self.initial_capital) ** (365 / days) - 1) * 100
        else:
            cagr = total_return

        daily_returns = pd.Series(equities).pct_change().dropna()
        sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(365)) if daily_returns.std() > 0 else 0
        neg_returns = daily_returns[daily_returns < 0]
        sortino = (daily_returns.mean() / neg_returns.std() * np.sqrt(365)) if len(neg_returns) > 0 and neg_returns.std() > 0 else 0

        calmar = cagr / (max_dd * 100) if max_dd > 0 else 0

        report = {
            "total_trades": len(self.closed_trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "total_return_pct": total_return,
            "cagr": cagr,
            "max_drawdown": max_dd,
            "profit_factor": profit_factor,
            "avg_win_pct": avg_win,
            "avg_loss_pct": avg_loss,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "initial_capital": self.initial_capital,
            "final_equity": final_equity,
            "days_tested": days,
        }

        logger.info("=" * 60)
        logger.info("BACKTEST RESULTS")
        logger.info("=" * 60)
        logger.info("Total Trades: %d", report["total_trades"])
        logger.info("Win Rate: %.1f%%", report["win_rate"] * 100)
        logger.info("Total Return: %.1f%%", report["total_return_pct"])
        logger.info("CAGR: %.1f%%", report["cagr"])
        logger.info("Max Drawdown: %.1f%%", report["max_drawdown"] * 100)
        logger.info("Profit Factor: %.2f", report["profit_factor"])
        logger.info("Sharpe Ratio: %.2f", report["sharpe_ratio"])
        logger.info("Sortino Ratio: %.2f", report["sortino_ratio"])
        logger.info("Calmar Ratio: %.2f", report["calmar_ratio"])
        logger.info("Final Equity: $%.2f", report["final_equity"])
        logger.info("=" * 60)

        return report


async def fetch_backtest_data(symbols: list, interval: str = "1d",
                              days: int = 365) -> dict:
    import aiohttp
    data = {}
    base_url = "https://api.binance.com"

    async with aiohttp.ClientSession() as session:
        for symbol in symbols:
            try:
                end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
                start_time = end_time - (days * 86400 * 1000)
                params = {
                    "symbol": symbol, "interval": interval,
                    "startTime": start_time, "endTime": end_time, "limit": 1000,
                }
                async with session.get(f"{base_url}/api/v3/klines", params=params) as resp:
                    if resp.status == 200:
                        klines = await resp.json()
                        symbol_data = {}
                        for k in klines:
                            date = datetime.fromtimestamp(
                                k[0] / 1000, tz=timezone.utc
                            ).strftime("%Y-%m-%d")
                            symbol_data[date] = {
                                "open": float(k[1]), "high": float(k[2]),
                                "low": float(k[3]), "close": float(k[4]),
                                "volume": float(k[5]),
                            }
                        data[symbol] = symbol_data
                        logger.info("Fetched %d candles for %s", len(symbol_data), symbol)
            except Exception as e:
                logger.error("Failed to fetch %s: %s", symbol, e)
            await asyncio.sleep(0.5)
    return data


async def run_backtest():
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT",
               "DOTUSDT", "AVAXUSDT", "LINKUSDT", "MATICUSDT", "ATOMUSDT"]

    logger.info("Fetching historical data for %d symbols...", len(symbols))
    data = await fetch_backtest_data(symbols, "1d", 365)

    if not data:
        logger.error("No data fetched")
        return

    engine = BacktestEngine(
        initial_capital=10000,
        max_positions=8,
        risk_per_trade=0.02,
        atr_stop_mult=2.5,
        confluence_threshold=65,
    )
    report = engine.run(data)

    Path("data").mkdir(exist_ok=True)
    with open("data/backtest_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    trades_df = pd.DataFrame(engine.closed_trades)
    trades_df.to_csv("data/backtest_trades.csv", index=False)

    equity_df = pd.DataFrame(engine.equity_curve)
    equity_df.to_csv("data/backtest_equity.csv", index=False)

    logger.info("Reports saved to data/")


if __name__ == "__main__":
    asyncio.run(run_backtest())
