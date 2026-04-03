"""
Telegram Notifier — Playbook-enhanced monitoring with 22 commands.

New commands from playbook:
  /mode       — Current operating mode (AGGRESSIVE/BALANCED/DEFENSIVE/CAPITAL_PRESERVATION)
  /heat       — Portfolio heat (total % at risk across all positions)
  /kelly      — Kelly fraction, scale factor, anti-martingale status
  /signals    — Recent signals with confluence scores
  /perf       — Performance metrics (win rate, Sharpe, R:R, compound growth)
  /dominance  — BTC dominance, altcoin season index, rotation advice
  /circuit    — Circuit breaker status and drawdown levels
  /compound   — Compound growth projection at current win rate
  /market     — Market overview (regime, F&G, BTC dom, altseason)
"""
import asyncio
import json
import math
from datetime import datetime, timezone
from typing import Optional, Callable

import aiohttp

from config.settings import Settings
from src.utils import setup_logging, utc_now

logger = setup_logging()


def _fmt_pnl(v: float) -> str:
    return f"+${v:.2f}" if v >= 0 else f"-${abs(v):.2f}"


def _fmt_pct(v: float) -> str:
    return f"+{v:.2f}%" if v >= 0 else f"{v:.2f}%"


def _mode_emoji(mode: str) -> str:
    return {"AGGRESSIVE": "🚀", "BALANCED": "⚖️",
            "DEFENSIVE": "🛡️", "CAPITAL_PRESERVATION": "🏦"}.get(mode, "⚖️")


def _regime_emoji(regime: str) -> str:
    return {"BULL": "🐂", "BEAR": "🐻", "SIDEWAYS": "↔️",
            "HIGH_VOLATILITY": "⚡"}.get(regime, "⚖️")


class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self.enabled = bool(bot_token and chat_id)
        self.command_handlers: dict = {}
        self._polling = False
        self._last_update_id = 0

    async def send_message(self, text: str, parse_mode: str = "HTML"):
        if not self.enabled:
            return
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "chat_id": self.chat_id,
                    "text": text[:4096],
                    "parse_mode": parse_mode,
                    "disable_web_page_preview": True,
                }
                async with session.post(
                    f"{self.base_url}/sendMessage", json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        logger.warning("Telegram send failed (%d): %s", resp.status, body[:200])
        except Exception as e:
            logger.warning("Telegram error: %s", e)

    async def send_photo(self, photo_bytes: bytes, caption: str = ""):
        if not self.enabled:
            return
        try:
            async with aiohttp.ClientSession() as session:
                data = aiohttp.FormData()
                data.add_field("chat_id", self.chat_id)
                data.add_field("photo", photo_bytes, filename="chart.png",
                               content_type="image/png")
                if caption:
                    data.add_field("caption", caption[:1024])
                    data.add_field("parse_mode", "HTML")
                async with session.post(
                    f"{self.base_url}/sendPhoto", data=data,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        logger.warning("Telegram photo failed (%d): %s", resp.status, body[:200])
        except Exception as e:
            logger.warning("Telegram photo error: %s", e)

    async def send_alert(self, text: str):
        await self.send_message(f"<b>⚠️ ALERT</b>\n{text}")

    # ------------------------------------------------------------------
    # Trade Notifications
    # ------------------------------------------------------------------

    async def notify_entry(self, trade: dict):
        ai = trade.get("ai_analysis") or {}
        ai_verdict = ai.get("verdict", "N/A")
        ai_conf = ai.get("confidence")
        ai_str = f"{ai_verdict} ({ai_conf:.0%})" if isinstance(ai_conf, float) else ai_verdict
        usdt_val = trade.get("usdt_value", 0)
        score = trade.get("confluence_score", 0)
        mode = trade.get("regime_at_entry", "")

        try:
            from src.charts import generate_trade_card
            card_data = {
                "symbol": trade["symbol"], "price": trade["entry_price"],
                "confluence_score": score, "stop_loss": trade.get("stop_loss", 0),
                "take_profit": trade.get("take_profit", 0),
                "position_size_usd": usdt_val, "regime": mode, "ai_verdict": ai_str,
            }
            chart_bytes = generate_trade_card(card_data, action="ENTRY")
            caption = (f"<b>🟢 ENTRY</b> | {trade['symbol']}\n"
                       f"${trade['entry_price']:.4f} | ${usdt_val:.2f} | "
                       f"Score: {score}/100 | AI: {ai_str}")
            await self.send_photo(chart_bytes, caption)
        except Exception:
            msg = (
                f"<b>🟢 NEW ENTRY</b>\n"
                f"Symbol: <code>{trade['symbol']}</code>\n"
                f"Price: <code>${trade['entry_price']:.4f}</code>\n"
                f"Size: <code>${usdt_val:.2f}</code>\n"
                f"SL: <code>${trade.get('stop_loss', 0):.4f}</code> | "
                f"TP: <code>${trade.get('take_profit', 0):.4f}</code>\n"
                f"Score: <code>{score}/100</code> | Mode: <code>{mode}</code>\n"
                f"AI: <code>{ai_str}</code>"
            )
            await self.send_message(msg)

    async def notify_exit(self, result: dict):
        pnl = result.get("pnl", 0)
        pnl_pct = result.get("pnl_pct", 0)
        is_full = result.get("is_full_exit", True)
        reason = result.get("reason", "Unknown")
        exit_type = "FULL EXIT" if is_full else "PARTIAL (50%)"
        emoji = "🟢" if pnl >= 0 else "🔴"

        try:
            from src.charts import generate_trade_card
            card_data = {
                "symbol": result["symbol"], "price": result.get("exit_price", 0),
                "pnl": pnl, "pnl_pct": pnl_pct,
                "hold_time": result.get("hold_time", "N/A"),
                "exit_reason": reason,
            }
            chart_bytes = generate_trade_card(card_data, action="EXIT")
            caption = (f"<b>{emoji} {exit_type}</b> | {result['symbol']}\n"
                       f"P&L: {_fmt_pnl(pnl)} ({_fmt_pct(pnl_pct)}) | {reason}")
            await self.send_photo(chart_bytes, caption)
        except Exception:
            msg = (
                f"<b>{emoji} {exit_type}</b>\n"
                f"Symbol: <code>{result['symbol']}</code>\n"
                f"Price: <code>${result.get('exit_price', 0):.4f}</code>\n"
                f"P&L: <code>{_fmt_pnl(pnl)}</code> ({_fmt_pct(pnl_pct)})\n"
                f"Reason: <code>{reason}</code>"
            )
            await self.send_message(msg)

    async def notify_regime_change(self, old_regime: str, new_regime: str):
        from src.regime import MODE_AGGRESSIVE, MODE_BALANCED, MODE_DEFENSIVE, MODE_CAPITAL_PRESERVATION
        msg = (
            f"<b>🔄 REGIME CHANGE</b>\n"
            f"From: <code>{_regime_emoji(old_regime)} {old_regime}</code>\n"
            f"To: <code>{_regime_emoji(new_regime)} {new_regime}</code>\n\n"
            f"<i>Strategy parameters auto-adjusted for new regime.</i>"
        )
        await self.send_message(msg)

    async def notify_mode_change(self, old_mode: str, new_mode: str):
        msg = (
            f"<b>⚙️ OPERATING MODE CHANGE</b>\n"
            f"From: <code>{_mode_emoji(old_mode)} {old_mode}</code>\n"
            f"To: <code>{_mode_emoji(new_mode)} {new_mode}</code>"
        )
        await self.send_message(msg)

    async def notify_circuit_breaker(self, cb_data: dict):
        action = cb_data.get("action", "")
        dd = cb_data.get("drawdown_pct", 0)
        emoji = "🚨" if action in ("KILL_SWITCH", "CAPITAL_PRESERVATION") else "⚠️"
        msg = (
            f"<b>{emoji} CIRCUIT BREAKER: {action}</b>\n"
            f"Drawdown: <code>{dd*100:.1f}%</code>\n"
        )
        if action == "DEFENSIVE":
            msg += "<i>Position sizes reduced 50%. Monitoring closely.</i>"
        elif action == "EMERGENCY_LIQUIDATE_50":
            msg += "<i>Closing 50% of positions to protect capital.</i>"
        elif action == "CAPITAL_PRESERVATION":
            msg += "<i>Going to cash. No new entries until regime improves.</i>"
        elif action == "KILL_SWITCH":
            msg += "<i>All trading halted. Send /resume to restart.</i>"
        await self.send_message(msg)

    async def notify_drawdown_warning(self, dd: dict):
        await self.notify_circuit_breaker(dd)

    async def notify_calibration(self, cal_data: dict):
        msg = (
            f"<b>🎯 CALIBRATION</b>\n"
            f"Parameter: <code>{cal_data.get('parameter_name', 'N/A')}</code>\n"
            f"Old: <code>{cal_data.get('old_value', 0):.4f}</code>\n"
            f"New: <code>{cal_data.get('new_value', 0):.4f}</code>\n"
            f"Reason: {cal_data.get('reason', 'N/A')}"
        )
        await self.send_message(msg)

    async def notify_news_alert(self, alerts: list):
        for alert in alerts[:3]:
            title = alert.get("title", alert.get("headline", "N/A"))
            symbol = alert.get("symbol", "Market")
            sentiment = alert.get("sentiment", 0)
            sent_emoji = "🟢" if sentiment > 0.2 else "🔴" if sentiment < -0.2 else "⚪"
            msg = (f"<b>📰 NEWS ALERT</b>\n"
                   f"{sent_emoji} {title}\n"
                   f"Symbol: <code>{symbol}</code> | "
                   f"Sentiment: <code>{sentiment:+.2f}</code>")
            await self.send_message(msg)

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------

    async def send_daily_report(self, portfolio, regime, news, risk):
        status = portfolio.get_status()
        daily_pnl = portfolio.db.get_daily_pnl()
        trades_today = portfolio.db.get_trades_for_period(1)
        wins = sum(1 for t in trades_today if (t.get("pnl") or 0) > 0)
        losses = sum(1 for t in trades_today if (t.get("pnl") or 0) < 0)
        win_rate = wins / len(trades_today) * 100 if trades_today else 0

        mode_e = _mode_emoji(regime.current_mode)
        reg_e = _regime_emoji(regime.current_regime)

        msg = (
            f"<b>📊 DAILY REPORT</b> — {utc_now().strftime('%Y-%m-%d')}\n\n"
            f"<b>Portfolio</b>\n"
            f"  Value: <code>${status.get('portfolio_value', 0):,.2f}</code>\n"
            f"  Cash:  <code>${status.get('cash_available', 0):,.2f}</code>\n"
            f"  P&L Today: <code>{_fmt_pnl(daily_pnl)}</code>\n"
            f"  Drawdown: <code>{status.get('drawdown', 0):.1%}</code>\n\n"
            f"<b>Trades</b>\n"
            f"  Count: <code>{len(trades_today)}</code> (W:{wins} L:{losses})\n"
            f"  Win Rate: <code>{win_rate:.0f}%</code>\n"
            f"  Positions: <code>{len(portfolio.open_positions)}</code>\n\n"
            f"<b>Market</b>\n"
            f"  Regime: {reg_e} <code>{regime.current_regime}</code>\n"
            f"  Mode: {mode_e} <code>{regime.current_mode}</code>\n"
            f"  F&G: <code>{news.fear_greed_value} ({news.fear_greed_label})</code>\n"
            f"  BTC Dom: <code>{news.btc_dominance:.1f}%</code>\n"
            f"  AltSeason: <code>{news.altcoin_season_index}</code>\n"
        )
        await self.send_message(msg)

    async def send_weekly_report(self, portfolio, regime, news, risk, calibrator, ai):
        status = portfolio.get_status()
        weekly_pnl = portfolio.db.get_daily_pnl() * 7  # approximation
        trades = portfolio.db.get_trade_stats(50)
        risk_status = risk.get_status_summary()

        msg = (
            f"<b>📈 WEEKLY REPORT</b> — {utc_now().strftime('%Y-W%V')}\n\n"
            f"<b>Performance</b>\n"
            f"  Portfolio: <code>${status.get('portfolio_value', 0):,.2f}</code>\n"
            f"  Weekly P&L: <code>{_fmt_pnl(weekly_pnl)}</code>\n"
            f"  Win Rate: <code>{trades.get('win_rate', 0):.1%}</code>\n"
            f"  Avg Win: <code>{trades.get('avg_win', 0):.2%}</code>\n"
            f"  Avg Loss: <code>{trades.get('avg_loss', 0):.2%}</code>\n"
            f"  Profit Factor: <code>{trades.get('profit_factor', 0):.2f}</code>\n\n"
            f"<b>Risk Engine</b>\n"
            f"  Kelly Fraction: <code>{risk_status['kelly_fraction']:.3f}</code>\n"
            f"  Scale Factor: <code>{risk_status['scale_factor']:.2f}×</code>\n"
            f"  Consecutive Wins: <code>{risk_status['consecutive_wins']}</code>\n"
            f"  Consecutive Losses: <code>{risk_status['consecutive_losses']}</code>\n\n"
            f"<b>AI</b>\n"
            f"  Daily Calls: <code>{ai.get_status()['daily_calls']}</code>\n"
            f"  Model: <code>{ai.get_status()['strong_model']}</code>\n"
        )
        await self.send_message(msg)

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    def register_commands(self, handlers: dict):
        self.command_handlers = handlers

    async def start_polling(self):
        if not self.enabled:
            return
        self._polling = True
        logger.info("Telegram polling started")
        while self._polling:
            try:
                updates = await self._get_updates()
                for update in updates:
                    await self._handle_update(update)
                await asyncio.sleep(2)
            except Exception as e:
                logger.error("Polling error: %s", e)
                await asyncio.sleep(10)

    async def _get_updates(self) -> list:
        try:
            async with aiohttp.ClientSession() as session:
                params = {"offset": self._last_update_id + 1, "timeout": 5, "limit": 10}
                async with session.get(
                    f"{self.base_url}/getUpdates", params=params,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("result", [])
        except Exception:
            pass
        return []

    async def _handle_update(self, update: dict):
        self._last_update_id = update.get("update_id", self._last_update_id)
        message = update.get("message", {})
        text = message.get("text", "")
        if not text:
            return
        parts = text.strip().split(maxsplit=1)
        cmd = parts[0].lower().split("@")[0]
        args = parts[1] if len(parts) > 1 else ""
        handler = self.command_handlers.get(cmd)
        if handler:
            try:
                response = await handler(args)
                if response:
                    await self.send_message(response)
            except Exception as e:
                await self.send_message(f"<b>Error:</b> {e}")


# ------------------------------------------------------------------
# Command Handlers Builder
# ------------------------------------------------------------------

def build_command_handlers(portfolio, risk_manager, calibrator,
                            watchdog, regime, news, ai,
                            alpha_hunter=None) -> dict:

    async def cmd_status(args):
        s = portfolio.get_status()
        reg_e = _regime_emoji(regime.current_regime)
        mode_e = _mode_emoji(regime.current_mode)
        pos_lines = []
        for p in portfolio.open_positions[:8]:
            try:
                price = await portfolio.exchange.get_price(p["symbol"])
                pnl_pct = ((price / p["entry_price"]) - 1) * 100
                arrow = "▲" if pnl_pct >= 0 else "▼"
                pos_lines.append(f"  {arrow} {p['symbol']}: {pnl_pct:+.2f}%")
            except Exception:
                pos_lines.append(f"  {p['symbol']}: N/A")
        pos_text = "\n".join(pos_lines) if pos_lines else "  No active positions"

        # Sub-min positions
        submin_lines = []
        for p in portfolio.sub_min_positions[:5]:
            try:
                price = await portfolio.exchange.get_price(p["symbol"])
                qty = p.get("remaining_quantity", p.get("quantity", 0))
                val = qty * price
                pnl_pct = ((price / p["entry_price"]) - 1) * 100 if p["entry_price"] > 0 else 0
                submin_lines.append(
                    f"  📌 {p['symbol']}: ${val:.2f} ({pnl_pct:+.1f}%) — waiting $10"
                )
            except Exception:
                submin_lines.append(f"  📌 {p['symbol']}")
        submin_text = "\n".join(submin_lines) if submin_lines else ""

        sub_min_v = s.get("sub_min_value", 0)
        deployable = s["cash_available"]
        min_pos = max(11.0, s["portfolio_value"] * 0.01)
        affordable = int(deployable / min_pos)

        msg = (
            f"<b>📊 PORTFOLIO STATUS</b>\n\n"
            f"Total:    <code>${s['portfolio_value']:.2f}</code>\n"
            f"💵 Cash:  <code>${deployable:.2f}</code> (~{affordable} new positions)\n"
            f"📈 Active: <code>${s['invested_value']:.2f}</code>\n"
            f"📌 Sub-min: <code>${sub_min_v:.2f}</code> (locked, waiting)\n"
            f"Drawdown: <code>{s['drawdown']:.1%}</code>\n\n"
            f"Regime: {reg_e} <code>{regime.current_regime}</code>\n"
            f"Mode:   {mode_e} <code>{regime.current_mode}</code>\n\n"
            f"<b>Active ({len(portfolio.open_positions)}):</b>\n<code>{pos_text}</code>"
        )
        if submin_text:
            msg += f"\n\n<b>Sub-min (can't sell yet):</b>\n<code>{submin_text}</code>"
        return msg

    async def cmd_mode(args):
        """Show current operating mode with parameters."""
        mode_e = _mode_emoji(regime.current_mode)
        params = regime.get_regime_params()
        return (
            f"<b>⚙️ OPERATING MODE</b>\n\n"
            f"{mode_e} <b>{regime.current_mode}</b>\n\n"
            f"Entries Allowed: <code>{'✅' if params['entries_allowed'] else '❌'}</code>\n"
            f"Position Multiplier: <code>{params['position_multiplier']:.1f}×</code>\n"
            f"Max Exposure: <code>{params['max_exposure']:.0%}</code>\n"
            f"Cash Reserve: <code>{params['cash_reserve']:.0%}</code>\n"
            f"Confluence Threshold: <code>{params['confluence_threshold']}/100</code>\n"
            f"Stop ATR Mult: <code>{params['stop_atr_mult']:.1f}×</code>\n"
            f"TP ATR Mult: <code>{params['tp_atr_mult']:.1f}×</code>\n\n"
            f"Regime: <code>{_regime_emoji(regime.current_regime)} {regime.current_regime}</code>\n"
            f"BTC Dom: <code>{news.btc_dominance:.1f}%</code>\n"
            f"Rotation: <code>{news.get_dominance_strategy()}</code>"
        )

    async def cmd_heat(args):
        """Portfolio heat — total % at risk across all open positions."""
        heat = risk_manager.get_open_risk_pct(
            portfolio.portfolio_value, portfolio.open_positions
        )
        max_heat = Settings.risk.MAX_PORTFOLIO_HEAT
        heat_bar = "█" * int(heat / max_heat * 10) + "░" * (10 - int(heat / max_heat * 10))
        heat_bar = heat_bar[:10]
        emoji = "🟢" if heat < max_heat * 0.5 else "🟡" if heat < max_heat * 0.8 else "🔴"

        pos_risks = []
        for p in portfolio.open_positions:
            entry = p.get("entry_price", 0)
            stop = p.get("stop_loss", 0)
            qty = p.get("remaining_quantity", p.get("quantity", 0))
            if entry > 0 and stop > 0:
                risk_usd = (entry - stop) * qty
                risk_pct = risk_usd / portfolio.portfolio_value * 100 if portfolio.portfolio_value > 0 else 0
                pos_risks.append(f"  {p['symbol']}: <code>{risk_pct:.2f}%</code> at risk")

        pos_text = "\n".join(pos_risks) if pos_risks else "  No open positions"

        return (
            f"<b>{emoji} PORTFOLIO HEAT</b>\n\n"
            f"Total Risk: <code>{heat:.1%}</code> / <code>{max_heat:.0%}</code> max\n"
            f"Heat Bar: <code>[{heat_bar}]</code>\n\n"
            f"<b>Position Risk:</b>\n{pos_text}\n\n"
            f"<i>Heat = total $ at risk / portfolio value. Cap: {max_heat:.0%}</i>"
        )

    async def cmd_kelly(args):
        """Kelly fraction and anti-martingale status."""
        rs = risk_manager.get_status_summary()
        scale = rs["scale_factor"]
        scale_dir = "📈" if scale > 1.0 else "📉" if scale < 1.0 else "➡️"

        return (
            f"<b>📐 KELLY & SIZING</b>\n\n"
            f"Win Rate: <code>{rs['win_rate']:.1%}</code>\n"
            f"Avg Win: <code>{rs['avg_win']:.2%}</code>\n"
            f"Avg Loss: <code>{rs['avg_loss']:.2%}</code>\n"
            f"Kelly Fraction: <code>{rs['kelly_fraction']:.3f}</code> (quarter Kelly)\n\n"
            f"<b>Anti-Martingale</b>\n"
            f"{scale_dir} Scale Factor: <code>{scale:.2f}×</code>\n"
            f"Consecutive Wins: <code>{rs['consecutive_wins']}</code>\n"
            f"Consecutive Losses: <code>{rs['consecutive_losses']}</code>\n\n"
            f"Position Modifier: <code>{rs['position_size_modifier']:.2f}×</code>\n\n"
            f"<i>Scale up +10% after each win, down -10% after each loss.</i>"
        )

    async def cmd_circuit(args):
        """Circuit breaker status."""
        dd = risk_manager.check_drawdown(portfolio.portfolio_value, portfolio.peak_value)
        drawdown = dd.get("drawdown_pct", 0)
        action = dd.get("action", "NONE")

        cb1 = Settings.risk.CIRCUIT_BREAKER_1
        cb2 = Settings.risk.CIRCUIT_BREAKER_2
        cb3 = Settings.risk.CIRCUIT_BREAKER_3
        max_dd = Settings.risk.MAX_PORTFOLIO_DRAWDOWN

        status_line = f"Current: <code>{drawdown:.1%}</code>\n"

        return (
            f"<b>⚡ CIRCUIT BREAKERS</b>\n\n"
            f"{status_line}"
            f"Action: <code>{action}</code>\n\n"
            f"{'🟢' if drawdown < cb1 else '🟡'} {cb1:.0%} → Reduce sizes 50%\n"
            f"{'🟢' if drawdown < cb2 else '🟡' if drawdown < cb2*1.5 else '🔴'} {cb2:.0%} → Close 50% positions\n"
            f"{'🟢' if drawdown < cb3 else '🔴'} {cb3:.0%} → Capital preservation mode\n"
            f"{'🟢' if drawdown < max_dd else '🚨'} {max_dd:.0%} → Kill switch\n\n"
            f"Kill Switch: <code>{'🚨 ACTIVE' if risk_manager.kill_switch_active else '✅ Off'}</code>\n"
            f"Paused: <code>{risk_manager.is_paused}</code>"
        )

    async def cmd_dominance(args):
        """BTC dominance and altcoin season analysis."""
        strategy = news.get_dominance_strategy()
        alt_idx = news.altcoin_season_index
        btc_dom = news.btc_dominance

        alt_status = (
            "🔥 ALT SEASON ACTIVE" if alt_idx >= 75
            else "🟢 Alts outperforming" if alt_idx >= 50
            else "⚪ Mixed market" if alt_idx >= 25
            else "🟠 BTC Season"
        )

        rotation_advice = {
            "BTC_FOCUS": "Focus 80%+ capital on BTC/ETH. Avoid alts.",
            "ALTCOIN_SEASON": "Rotate into quality alts (L1s → DeFi → AI tokens).",
            "NEUTRAL": "Balanced: 60% BTC/ETH + 30% select alts.",
        }.get(strategy, "Balanced approach.")

        return (
            f"<b>🔄 SECTOR ROTATION</b>\n\n"
            f"BTC Dominance: <code>{btc_dom:.1f}%</code>\n"
            f"Altcoin Season Index: <code>{alt_idx}/100</code>\n"
            f"Status: <b>{alt_status}</b>\n\n"
            f"Strategy: <i>{strategy}</i>\n"
            f"Advice: {rotation_advice}\n\n"
            f"<b>Rotation Cycle (typical):</b>\n"
            f"  1️⃣ BTC leads\n"
            f"  2️⃣ ETH/SOL follow\n"
            f"  3️⃣ DeFi tokens pump\n"
            f"  4️⃣ AI/Gaming tokens\n"
            f"  5️⃣ Meme coins = late signal 🔔\n\n"
            f"Fear & Greed: <code>{news.fear_greed_value} ({news.fear_greed_label})</code>"
        )

    async def cmd_compound(args):
        """Compound growth projection at current strategy parameters."""
        rs = risk_manager.get_status_summary()
        wr = rs["win_rate"]
        avg_win = rs["avg_win"]
        avg_loss = rs["avg_loss"]
        portfolio_val = portfolio.portfolio_value

        # Expected value per trade
        ev = wr * avg_win - (1 - wr) * avg_loss
        rr = avg_win / avg_loss if avg_loss > 0 else 2.0

        # Project compound growth
        trades_per_day = 3  # assumption
        daily_ev = ev * trades_per_day
        weekly_val = portfolio_val * ((1 + ev) ** (trades_per_day * 7))
        monthly_val = portfolio_val * ((1 + ev) ** (trades_per_day * 30))

        return (
            f"<b>📈 COMPOUND GROWTH PROJECTION</b>\n\n"
            f"Win Rate: <code>{wr:.1%}</code>\n"
            f"Avg Win: <code>{avg_win:.2%}</code>\n"
            f"Avg Loss: <code>{avg_loss:.2%}</code>\n"
            f"R:R Ratio: <code>{rr:.2f}:1</code>\n"
            f"Expected Value/Trade: <code>{ev:+.3%}</code>\n\n"
            f"<b>Projections (current: ${portfolio_val:.2f})</b>\n"
            f"  1 Week:  <code>${weekly_val:.2f}</code>\n"
            f"  1 Month: <code>${monthly_val:.2f}</code>\n\n"
            f"<i>Based on {trades_per_day} trades/day assumption.\n"
            f"Actual results vary with market conditions.</i>"
        )

    async def cmd_signals(args):
        """Recent signals with confluence scores."""
        try:
            recent = portfolio.db.get_recent_signals(10)
            if not recent:
                return "No recent signals yet."
            lines = []
            for s in recent[:10]:
                score = s.get("confluence_score", 0)
                symbol = s.get("symbol", "?")
                action = s.get("action_taken", "?")
                ts = s.get("timestamp", "")
                ts_str = str(ts)[:16] if ts else "?"
                bar = "█" * (score // 10) + "░" * (10 - score // 10)
                lines.append(f"  {symbol}: <code>{score}/100</code> [{bar}] → {action}")
            return (
                f"<b>📡 RECENT SIGNALS</b>\n\n" +
                "\n".join(lines) +
                f"\n\n<i>Threshold: {Settings.strategy.CONFLUENCE_SCORE_THRESHOLD}/100</i>"
            )
        except Exception as e:
            return f"Signals error: {e}"

    async def cmd_perf(args):
        """Performance metrics."""
        stats = portfolio.db.get_trade_stats(50)
        daily_pnl = portfolio.db.get_daily_pnl()
        status = portfolio.get_status()

        profit_factor = stats.get("profit_factor", 0)
        pf_emoji = "🟢" if profit_factor > 1.5 else "🟡" if profit_factor > 1.0 else "🔴"

        return (
            f"<b>🏆 PERFORMANCE METRICS</b>\n\n"
            f"Trades (last 50): <code>{stats.get('total_trades', 0)}</code>\n"
            f"Win Rate: <code>{stats.get('win_rate', 0):.1%}</code>\n"
            f"Avg Win: <code>{stats.get('avg_win', 0):.2%}</code>\n"
            f"Avg Loss: <code>{stats.get('avg_loss', 0):.2%}</code>\n"
            f"R:R Ratio: <code>{stats.get('avg_rr', 0):.2f}:1</code>\n"
            f"{pf_emoji} Profit Factor: <code>{profit_factor:.2f}</code>\n"
            f"Stop Hit Rate: <code>{stats.get('stop_loss_hit_rate', 0):.1%}</code>\n\n"
            f"Daily P&L: <code>{_fmt_pnl(daily_pnl)}</code>\n"
            f"Portfolio: <code>${status['portfolio_value']:.2f}</code>\n"
            f"Peak: <code>${portfolio.peak_value:.2f}</code>\n"
            f"Drawdown: <code>{status['drawdown']:.1%}</code>"
        )

    async def cmd_market(args):
        """Full market overview."""
        reg_e = _regime_emoji(regime.current_regime)
        mode_e = _mode_emoji(regime.current_mode)
        return (
            f"<b>🌍 MARKET OVERVIEW</b>\n\n"
            f"{reg_e} Regime: <code>{regime.current_regime}</code>\n"
            f"{mode_e} Mode: <code>{regime.current_mode}</code>\n"
            f"Confidence: <code>{regime.regime_confidence:.0%}</code>\n\n"
            f"😱 Fear & Greed: <code>{news.fear_greed_value} ({news.fear_greed_label})</code>\n"
            f"₿ BTC Dominance: <code>{news.btc_dominance:.1f}%</code>\n"
            f"🔄 AltSeason Index: <code>{news.altcoin_season_index}/100</code>\n"
            f"📊 Strategy: <code>{news.get_dominance_strategy()}</code>\n"
            f"📰 News Sentiment: <code>{news.get_sentiment_score():+.2f}</code>\n\n"
            f"BTC: {'✅ Above 200SMA' if regime.btc_indicators.get('above_200sma') else '❌ Below 200SMA'}\n"
            f"ADX: <code>{regime.btc_indicators.get('adx', 0):.0f}</code> "
            f"({'Trending' if regime.btc_indicators.get('adx', 0) > 25 else 'Ranging'})"
        )

    async def cmd_trades(args):
        pos_lines = []
        for p in portfolio.open_positions[:10]:
            try:
                price = await portfolio.exchange.get_price(p["symbol"])
                pnl_pct = ((price / p["entry_price"]) - 1) * 100
                sl_dist = ((price / p["stop_loss"]) - 1) * 100 if p.get("stop_loss") else 0
                arrow = "▲" if pnl_pct >= 0 else "▼"
                pos_lines.append(
                    f"  {arrow} <b>{p['symbol']}</b>: {pnl_pct:+.2f}%\n"
                    f"     Entry: ${p['entry_price']:.4f} | SL dist: {sl_dist:.1f}%"
                )
            except Exception:
                pos_lines.append(f"  {p['symbol']}: (unavailable)")
        if not pos_lines:
            return "No open positions."
        return "<b>📋 OPEN POSITIONS</b>\n\n" + "\n".join(pos_lines)

    async def cmd_pnl(args):
        daily = portfolio.db.get_daily_pnl()
        trades_today = portfolio.db.get_trades_for_period(1)
        wins = [t for t in trades_today if (t.get("pnl") or 0) > 0]
        losses = [t for t in trades_today if (t.get("pnl") or 0) <= 0]
        return (
            f"<b>💰 TODAY'S P&L</b>\n\n"
            f"Realized P&L: <code>{_fmt_pnl(daily)}</code>\n"
            f"Trades: <code>{len(trades_today)}</code>\n"
            f"Wins: <code>{len(wins)}</code> | Losses: <code>{len(losses)}</code>"
        )

    async def cmd_regime(args):
        return await cmd_market(args)

    async def cmd_news(args):
        headlines = news.get_top_headlines(8)
        if not headlines:
            return "No recent news available."
        lines = []
        for n in headlines[:8]:
            sentiment = n.get("sentiment", 0)
            emoji = "🟢" if sentiment > 0.2 else "🔴" if sentiment < -0.2 else "⚪"
            source = n.get("source", "")
            title = n.get("title", "")[:60]
            lines.append(f"{emoji} [{source}] {title}")
        return "<b>📰 LATEST NEWS</b>\n\n" + "\n".join(lines) + f"\n\n{news.get_market_summary()}"

    async def cmd_ai(args):
        status = ai.get_status()
        return (
            f"<b>🤖 AI ADVISOR STATUS</b>\n\n"
            f"Enabled: <code>{status['enabled']}</code>\n"
            f"Fast Model: <code>{status['fast_model']}</code>\n"
            f"Strong Model: <code>{status['strong_model']}</code>\n"
            f"Veto Power: <code>{status['veto_power']}</code>\n"
            f"Daily Calls: <code>{status['daily_calls']}/{status['max_daily_calls']}</code>\n"
            f"Cost Today: <code>${status['daily_cost_usd']:.4f}</code>\n\n"
            f"Last Portfolio Analysis:\n<i>{status['last_portfolio_analysis']}</i>"
        )

    async def cmd_risk(args):
        rs = risk_manager.get_status_summary()
        dd = risk_manager.check_drawdown(portfolio.portfolio_value, portfolio.peak_value)
        return (
            f"<b>⚠️ RISK STATUS</b>\n\n"
            f"Kill Switch: <code>{'🚨 ACTIVE' if rs['kill_switch_active'] else '✅ Off'}</code>\n"
            f"Paused: <code>{rs['is_paused']}</code>\n"
            f"Drawdown: <code>{dd.get('drawdown_pct', 0):.1%}</code> → {dd.get('action', 'NONE')}\n\n"
            f"Win Rate: <code>{rs['win_rate']:.1%}</code>\n"
            f"Kelly: <code>{rs['kelly_fraction']:.3f}</code>\n"
            f"Scale: <code>{rs['scale_factor']:.2f}×</code>\n"
            f"Modifier: <code>{rs['position_size_modifier']:.2f}×</code>"
        )

    async def cmd_health(args):
        health = watchdog.get_status()
        components = health.get("components", {})
        comp_lines = "\n".join(
            f"  {'✅' if v else '❌'} {k}"
            for k, v in components.items()
        )
        return (
            f"<b>💊 SYSTEM HEALTH</b>\n\n"
            f"Healthy: <code>{'✅' if health['is_healthy'] else '❌'}</code>\n"
            f"Memory: <code>{health['memory_mb']:.0f} MB</code>\n"
            f"CPU: <code>{health['cpu_percent']:.1f}%</code>\n"
            f"Restarts: <code>{health['restart_count']}</code>\n\n"
            f"<b>Components:</b>\n{comp_lines}"
        )

    async def cmd_pause(args):
        risk_manager.is_paused = True
        return "⏸ <b>Trading PAUSED.</b> Use /resume to continue."

    async def cmd_resume(args):
        risk_manager.is_paused = False
        risk_manager.kill_switch_active = False
        risk_manager.kill_switch_activated_at = None
        risk_manager.position_size_modifier = 1.0
        dd = risk_manager.check_drawdown(portfolio.portfolio_value, portfolio.peak_value)
        return (
            "✅ <b>Trading RESUMED</b>\n"
            f"Current drawdown: <code>{dd.get('drawdown_pct', 0):.1%}</code>\n"
            f"Portfolio: <code>${portfolio.portfolio_value:.2f}</code>\n\n"
            "<i>Note: Kill switch auto-resumes when drawdown recovers below 10%.\n"
            "Manual /resume only needed to force-override.</i>"
        )

    async def cmd_sell(args):
        if not args:
            return "Usage: /sell BTCUSDT"
        symbol = args.strip().upper()
        result = await portfolio.force_sell(symbol)
        if result:
            return (f"✅ Sold {symbol}\n"
                    f"P&L: {_fmt_pnl(result.get('pnl', 0))} "
                    f"({_fmt_pct(result.get('pnl_pct', 0))})")
        return f"❌ Could not sell {symbol} — not in open positions."

    async def cmd_sellall(args):
        results = await portfolio.force_sell_all()
        if not results:
            return "No open positions to sell."
        total_pnl = sum(r.get("pnl", 0) for r in results)
        return f"✅ Sold {len(results)} positions\nTotal P&L: {_fmt_pnl(total_pnl)}"

    async def cmd_calibrate(args):
        calibrator.check_and_calibrate()
        return "🎯 Calibration triggered."

    async def cmd_report(args):
        return await cmd_perf(args)

    async def cmd_weekly(args):
        status = portfolio.get_status()
        stats = portfolio.db.get_trade_stats(50)
        return (
            f"<b>📊 WEEKLY SUMMARY</b>\n\n"
            f"Portfolio: <code>${status['portfolio_value']:.2f}</code>\n"
            f"Trades: <code>{stats.get('total_trades', 0)}</code>\n"
            f"Win Rate: <code>{stats.get('win_rate', 0):.1%}</code>\n"
            f"Profit Factor: <code>{stats.get('profit_factor', 0):.2f}</code>"
        )

    async def cmd_help(args):
        return (
            "<b>📖 COMMANDS</b>\n\n"
            "<b>Portfolio</b>\n"
            "/status — Portfolio overview\n"
            "/trades — Open positions with P&L\n"
            "/pnl — Today's realized P&L\n"
            "/perf — Full performance metrics\n"
            "/compound — Growth projection\n\n"
            "<b>Market Intelligence</b>\n"
            "/market — Market overview\n"
            "/regime — Regime & mode status\n"
            "/mode — Operating mode details\n"
            "/dominance — BTC dominance & altseason\n"
            "/news — Latest headlines\n\n"
            "<b>Risk & Sizing</b>\n"
            "/risk — Risk manager status\n"
            "/heat — Portfolio heat (total % at risk)\n"
            "/kelly — Kelly fraction & anti-martingale\n"
            "/circuit — Circuit breaker levels\n\n"
            "<b>Signals & AI</b>\n"
            "/signals — Recent signal scores\n"
            "/ai — AI advisor status\n\n"
            "<b>Controls</b>\n"
            "/pause — Pause trading\n"
            "/resume — Resume + reset kill switch\n"
            "/sell SYMBOL — Force sell position\n"
            "/sellall — Force sell all\n"
            "/calibrate — Recalibrate parameters\n"
            "/health — System health\n"
            "/weekly — Weekly report\n"
            "/report — Quick report\n\n"
            "<b>Alpha Hunter</b>\n"
            "/alpha — Early-mover opportunities (squeeze, OBV, RVOL)\n"
            "/squeeze — Active TTM squeeze setups"
        )

    async def cmd_alpha(args):
        """Alpha hunter opportunities."""
        if alpha_hunter:
            return alpha_hunter.get_status_text()
        return "Alpha Hunter not initialized."

    async def cmd_squeeze(args):
        """Show current TTM squeeze candidates."""
        if not alpha_hunter:
            return "Alpha Hunter not initialized."
        opps = alpha_hunter.get_opportunities()
        squeeze_opps = [o for o in opps
                        if o.signals.get("squeeze_on") or o.signals.get("squeeze_fired")]
        if not squeeze_opps:
            return "No active squeeze setups found."
        lines = []
        for op in squeeze_opps[:8]:
            bars = op.signals.get("squeeze_bars", 0)
            fired = op.signals.get("squeeze_fired", False)
            status = "🚨 FIRED" if fired else f"🔴 ON ({bars} bars)"
            lines.append(f"  <code>{op.symbol}</code>: {status} | score={op.score}")
        return "<b>🗜️ SQUEEZE SETUPS</b>\n\n" + "\n".join(lines)

    return {
        "/status": cmd_status,
        "/mode": cmd_mode,
        "/heat": cmd_heat,
        "/kelly": cmd_kelly,
        "/circuit": cmd_circuit,
        "/dominance": cmd_dominance,
        "/compound": cmd_compound,
        "/signals": cmd_signals,
        "/perf": cmd_perf,
        "/market": cmd_market,
        "/trades": cmd_trades,
        "/pnl": cmd_pnl,
        "/regime": cmd_regime,
        "/news": cmd_news,
        "/ai": cmd_ai,
        "/risk": cmd_risk,
        "/health": cmd_health,
        "/pause": cmd_pause,
        "/resume": cmd_resume,
        "/sell": cmd_sell,
        "/sellall": cmd_sellall,
        "/calibrate": cmd_calibrate,
        "/report": cmd_report,
        "/weekly": cmd_weekly,
        "/help": cmd_help,
        "/alpha": cmd_alpha,
        "/squeeze": cmd_squeeze,
    }
