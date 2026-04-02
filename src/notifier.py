"""
Telegram Notifier — Notifications, alerts, commands, daily/weekly reports.

Commands:
  /status     — Portfolio overview
  /trades     — Open positions with live P&L
  /pnl        — Today's P&L
  /regime     — Market regime info
  /news       — Latest news sentiment
  /ai         — AI advisor status
  /risk       — Risk manager status
  /pause      — Pause trading
  /resume     — Resume trading
  /sell SYM   — Force sell a position
  /sellall    — Force sell all positions
  /calibrate  — Force recalibration
  /health     — System health
  /report     — On-demand daily report
  /weekly     — On-demand weekly report
  /help       — List commands
"""

import asyncio
import json
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


class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self.enabled = bool(bot_token and chat_id)
        self.command_handlers: dict = {}
        self._polling = False
        self._last_update_id = 0

    # ------------------------------------------------------------------
    # Core Messaging
    # ------------------------------------------------------------------

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
            logger.warning("Telegram notification error: %s", e)

    async def send_alert(self, text: str):
        await self.send_message(f"🚨 <b>ALERT</b>\n{text}")

    # ------------------------------------------------------------------
    # Trade Notifications
    # ------------------------------------------------------------------

    async def notify_entry(self, trade: dict):
        ai_conf = trade.get("ai_confidence", None)
        ai_str = f"{ai_conf:.0%}" if isinstance(ai_conf, (int, float)) else "N/A"
        usdt_val = trade.get("usdt_value", trade["entry_price"] * trade["quantity"])
        msg = (
            f"📈 <b>NEW ENTRY</b>\n"
            f"Symbol: <code>{trade['symbol']}</code>\n"
            f"Price: <code>${trade['entry_price']:.4f}</code>\n"
            f"Size: <code>${usdt_val:.2f}</code>\n"
            f"Qty: <code>{trade['quantity']:.6f}</code>\n"
            f"Stop-Loss: <code>${trade['stop_loss']:.4f}</code> ({Settings.risk.STOP_LOSS_PCT*100:.0f}%)\n"
            f"Take-Profit: <code>${trade['take_profit']:.4f}</code> ({Settings.risk.TAKE_PROFIT_PCT*100:.0f}%)\n"
            f"Score: <code>{trade.get('confluence_score', 0)}/100</code>\n"
            f"AI Confidence: <code>{ai_str}</code>\n"
            f"Regime: <code>{trade.get('regime_at_entry', 'N/A')}</code>\n"
            f"Sector: <code>{trade.get('sector', 'N/A')}</code>"
        )
        await self.send_message(msg)

    async def notify_exit(self, result: dict):
        pnl = result.get("pnl", result.get("pnl_usdt", 0))
        pnl_pct = result.get("pnl_pct", 0)
        emoji = "✅" if pnl >= 0 else "❌"
        is_full = result.get("is_full_exit", True)
        exit_type = "FULL EXIT" if is_full else "PARTIAL EXIT (50%)"
        msg = (
            f"{emoji} <b>{exit_type}</b>\n"
            f"Symbol: <code>{result['symbol']}</code>\n"
            f"Price: <code>${result.get('exit_price', 0):.4f}</code>\n"
            f"PnL: <code>{_fmt_pnl(pnl)}</code> ({_fmt_pct(pnl_pct)})\n"
            f"Reason: <code>{result.get('reason', 'Unknown')}</code>"
        )
        await self.send_message(msg)

    async def notify_regime_change(self, old_regime: str, new_regime: str):
        emoji_map = {"BULL": "🟢", "BEAR": "🔴", "SIDEWAYS": "🟡", "HIGH_VOLATILITY": "🟠"}
        msg = (
            f"🔄 <b>REGIME CHANGE</b>\n"
            f"From: {emoji_map.get(old_regime, '⚪')} <code>{old_regime}</code>\n"
            f"To: {emoji_map.get(new_regime, '⚪')} <code>{new_regime}</code>"
        )
        await self.send_message(msg)

    async def notify_calibration(self, cal_data: dict):
        msg = (
            f"⚙️ <b>CALIBRATION</b>\n"
            f"Parameter: <code>{cal_data.get('parameter_name', 'N/A')}</code>\n"
            f"Old: <code>{cal_data.get('old_value', 0):.4f}</code>\n"
            f"New: <code>{cal_data.get('new_value', 0):.4f}</code>\n"
            f"Reason: {cal_data.get('reason', 'N/A')}"
        )
        await self.send_message(msg)

    async def notify_drawdown_warning(self, dd: dict):
        pct = dd.get("drawdown_pct", 0)
        action = dd.get("action", "UNKNOWN")
        msg = (
            f"⚠️ <b>DRAWDOWN WARNING</b>\n"
            f"Drawdown: <code>{pct*100:.1f}%</code>\n"
            f"Action: <code>{action}</code>"
        )
        await self.send_message(msg)

    async def notify_news_alert(self, alerts: list):
        for alert in alerts[:3]:
            headline = alert.get("headline", alert.get("title", "N/A"))
            symbol = alert.get("symbol", "Market")
            sentiment = alert.get("sentiment", 0)
            if isinstance(sentiment, (int, float)):
                sent_str = f"{sentiment:.2f}"
            else:
                sent_str = str(sentiment)
            source = alert.get("source", "Unknown")
            msg = (
                f"📰 <b>NEWS ALERT</b>\n"
                f"Symbol: <code>{symbol}</code>\n"
                f"Headline: {headline}\n"
                f"Source: <code>{source}</code>\n"
                f"Sentiment: <code>{sent_str}</code>"
            )
            await self.send_message(msg)

    # ------------------------------------------------------------------
    # Daily Report
    # ------------------------------------------------------------------

    async def send_daily_report(self, portfolio, regime, news, risk):
        status = portfolio.get_status()
        daily_pnl = portfolio.db.get_daily_pnl()
        trades_today = portfolio.db.get_trades_for_period(1)
        wins = sum(1 for t in trades_today if t.get("pnl_usdt", 0) > 0)
        losses = sum(1 for t in trades_today if t.get("pnl_usdt", 0) < 0)
        stats = portfolio.db.get_trade_stats(100)

        msg = (
            f"📊 <b>DAILY REPORT — {utc_now().strftime('%Y-%m-%d')}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"💰 Portfolio: <code>${status.get('portfolio_value', 0):,.2f}</code>\n"
            f"💵 Cash: <code>${status.get('cash_available', 0):,.2f}</code>\n"
            f"📈 Daily P&L: <code>{_fmt_pnl(daily_pnl)}</code>\n"
            f"📊 Open Positions: <code>{status.get('open_positions', 0)}</code>\n"
            f"🔄 Trades Today: <code>{len(trades_today)}</code> (W:{wins} L:{losses})\n"
            f"📉 Drawdown: <code>{status.get('drawdown', 0)*100:.1f}%</code>\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"🌡️ Regime: <code>{regime.current_regime}</code>\n"
            f"😰 Fear/Greed: <code>{news.fear_greed_value}</code> ({news.fear_greed_label})\n"
            f"📰 Sentiment: <code>{news.get_sentiment_score():.2f}</code>\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"Win Rate: <code>{stats.get('win_rate', 0)*100:.1f}%</code>\n"
            f"Profit Factor: <code>{stats.get('profit_factor', 0):.2f}</code>\n"
            f"🛡️ Kill Switch: <code>{'ACTIVE' if risk.kill_switch_active else 'Off'}</code>"
        )
        await self.send_message(msg)

    # ------------------------------------------------------------------
    # Weekly Report
    # ------------------------------------------------------------------

    async def send_weekly_report(self, portfolio, regime, news, risk,
                                  calibrator, ai):
        status = portfolio.get_status()
        stats = portfolio.db.get_trade_stats(100)
        weekly_trades = portfolio.db.get_trades_for_period(7)
        weekly_pnl = sum(t.get("pnl_usdt", 0) for t in weekly_trades)
        wins = sum(1 for t in weekly_trades if t.get("pnl_usdt", 0) > 0)
        losses = sum(1 for t in weekly_trades if t.get("pnl_usdt", 0) < 0)
        total = wins + losses

        best = max(weekly_trades, key=lambda t: t.get("pnl_usdt", 0), default=None)
        worst = min(weekly_trades, key=lambda t: t.get("pnl_usdt", 0), default=None)

        sector_pnl = {}
        for t in weekly_trades:
            sector = Settings.get_sector_for_asset(t.get("symbol", ""))
            sector_pnl[sector] = sector_pnl.get(sector, 0) + t.get("pnl_usdt", 0)
        sector_text = "\n".join(
            f"  {s}: {_fmt_pnl(p)}"
            for s, p in sorted(sector_pnl.items(), key=lambda x: x[1], reverse=True)
        ) or "  No trades"

        ai_status = ai.get_status() if ai else {}

        wr_str = f"{wins/total*100:.0f}%" if total > 0 else "N/A"

        msg = (
            f"📊 <b>WEEKLY REPORT — {utc_now().strftime('%Y-%m-%d')}</b>\n\n"
            f"━━━ <b>Portfolio</b> ━━━\n"
            f"Value: <code>${status.get('portfolio_value', 0):,.2f}</code>\n"
            f"Cash: <code>${status.get('cash_available', 0):,.2f}</code>\n"
            f"Invested: <code>${status.get('invested_value', 0):,.2f}</code>\n"
            f"Drawdown: <code>{status.get('drawdown', 0)*100:.1f}%</code>\n\n"
            f"━━━ <b>Weekly Performance</b> ━━━\n"
            f"P&L: <code>{_fmt_pnl(weekly_pnl)}</code>\n"
            f"Trades: <code>{total}</code> (W:{wins} L:{losses})\n"
            f"Win Rate: <code>{wr_str}</code>\n"
        )
        if best:
            msg += f"Best: <code>{best.get('symbol', '?')}</code> {_fmt_pnl(best.get('pnl_usdt', 0))}\n"
        if worst:
            msg += f"Worst: <code>{worst.get('symbol', '?')}</code> {_fmt_pnl(worst.get('pnl_usdt', 0))}\n"

        msg += (
            f"\n━━━ <b>Sector P&L</b> ━━━\n{sector_text}\n\n"
            f"━━━ <b>Market</b> ━━━\n"
            f"Regime: <code>{regime.current_regime}</code>\n"
            f"Fear/Greed: <code>{news.fear_greed_value}</code> ({news.fear_greed_label})\n\n"
            f"━━━ <b>AI Advisor</b> ━━━\n"
            f"Calls: <code>{ai_status.get('recent_calls', 0)}</code>\n"
            f"Model: <code>{ai_status.get('model', 'N/A')}</code>\n\n"
            f"━━━ <b>Risk</b> ━━━\n"
            f"Win Rate (all): <code>{stats.get('win_rate', 0)*100:.1f}%</code>\n"
            f"Avg Win: <code>{stats.get('avg_win', 0)*100:.2f}%</code>\n"
            f"Avg Loss: <code>{stats.get('avg_loss', 0)*100:.2f}%</code>\n"
            f"Kelly: <code>{risk.kelly_fraction*100:.2f}%</code>\n"
            f"Consecutive Losses: <code>{portfolio.db.get_consecutive_losses()}</code>"
        )
        await self.send_message(msg[:4096])

    # ------------------------------------------------------------------
    # Command Polling
    # ------------------------------------------------------------------

    def register_commands(self, handlers: dict):
        self.command_handlers.update(handlers)

    async def start_polling(self):
        if not self.enabled:
            return
        self._polling = True
        logger.info("Telegram command polling started")
        while self._polling:
            try:
                await self._poll_updates()
            except Exception as e:
                logger.warning("Telegram polling error: %s", e)
            await asyncio.sleep(2)

    async def stop_polling(self):
        self._polling = False

    async def _poll_updates(self):
        try:
            async with aiohttp.ClientSession() as session:
                params = {"offset": self._last_update_id + 1, "timeout": 5}
                async with session.get(
                    f"{self.base_url}/getUpdates", params=params,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for update in data.get("result", []):
                            self._last_update_id = update["update_id"]
                            message = update.get("message", {})
                            text = message.get("text", "")
                            chat_id = str(message.get("chat", {}).get("id", ""))
                            if chat_id == self.chat_id and text.startswith("/"):
                                await self._handle_command(text)
        except Exception:
            pass

    async def _handle_command(self, text: str):
        parts = text.strip().split()
        command = parts[0].split("@")[0].lower()
        args = parts[1:]

        handler = self.command_handlers.get(command)
        if handler:
            try:
                result = await handler(args) if asyncio.iscoroutinefunction(handler) else handler(args)
                if isinstance(result, str):
                    await self.send_message(result)
            except Exception as e:
                await self.send_message(f"Command error: {e}")
        else:
            available = ", ".join(sorted(self.command_handlers.keys()))
            await self.send_message(f"Unknown command. Available: {available}")


# ------------------------------------------------------------------
# Command Handler Factory
# ------------------------------------------------------------------

def build_command_handlers(portfolio, risk_manager, calibrator, watchdog,
                           regime, news, ai) -> dict:

    async def cmd_status(args):
        status = portfolio.get_status()
        return (
            f"📊 <b>STATUS</b>\n"
            f"Portfolio: <code>${status.get('portfolio_value', 0):,.2f}</code>\n"
            f"Cash: <code>${status.get('cash_available', 0):,.2f}</code>\n"
            f"Invested: <code>${status.get('invested_value', 0):,.2f}</code>\n"
            f"Positions: <code>{status.get('open_positions', 0)}</code>\n"
            f"Drawdown: <code>{status.get('drawdown', 0)*100:.1f}%</code>\n"
            f"Regime: <code>{regime.current_regime}</code>\n"
            f"F&G: <code>{news.fear_greed_value}</code> ({news.fear_greed_label})"
        )

    async def cmd_trades(args):
        positions = portfolio.open_positions
        if not positions:
            return "No open positions."
        lines = ["<b>OPEN POSITIONS</b>"]
        for p in positions:
            try:
                price = await portfolio.exchange.get_price(p["symbol"])
            except Exception:
                price = p.get("entry_price", 0)
            pnl_pct = ((price / p["entry_price"]) - 1) * 100 if p["entry_price"] > 0 else 0
            emoji = "🟢" if pnl_pct >= 0 else "🔴"
            exits = p.get("tranche_exits", [])
            if isinstance(exits, str):
                exits = json.loads(exits)
            partial = " (50% sold)" if len(exits) > 0 else ""
            lines.append(
                f"{emoji} <code>{p['symbol']}</code> @ ${p['entry_price']:.4f} "
                f"→ ${price:.4f} ({_fmt_pct(pnl_pct)}){partial}\n"
                f"   SL=${p['stop_loss']:.4f}"
            )
        return "\n".join(lines)

    async def cmd_pnl(args):
        daily = portfolio.db.get_daily_pnl()
        trades = portfolio.db.get_trades_for_period(1)
        wins = sum(1 for t in trades if t.get("pnl_usdt", 0) > 0)
        losses = sum(1 for t in trades if t.get("pnl_usdt", 0) < 0)
        return (
            f"💰 <b>TODAY'S P&L</b>\n"
            f"Realized: <code>{_fmt_pnl(daily)}</code>\n"
            f"Trades: <code>{len(trades)}</code> (W:{wins} L:{losses})"
        )

    async def cmd_regime(args):
        params = regime.get_regime_params()
        return (
            f"🌡️ <b>MARKET REGIME</b>\n"
            f"Current: <code>{regime.current_regime}</code>\n"
            f"Entries: <code>{'Allowed' if params.get('entries_allowed') else 'Blocked'}</code>\n"
            f"Max Exposure: <code>{params.get('max_exposure', 0)*100:.0f}%</code>\n"
            f"Position Mult: <code>{params.get('position_multiplier', 1):.1f}x</code>"
        )

    async def cmd_news(args):
        headlines = news.get_top_headlines(5)
        if not headlines:
            return f"📰 No recent news. F&G: {news.fear_greed_value}"
        lines = [f"📰 <b>TOP NEWS</b> (F&G: {news.fear_greed_value} — {news.fear_greed_label})"]
        for h in headlines:
            sent = h.get("sentiment", 0)
            if isinstance(sent, str):
                emoji = "🟢" if sent == "positive" else "🔴" if sent == "negative" else "⚪"
            else:
                emoji = "🟢" if sent > 0 else "🔴" if sent < 0 else "⚪"
            lines.append(f"{emoji} {h.get('title', 'N/A')[:80]}")
        return "\n".join(lines)

    async def cmd_ai(args):
        s = ai.get_status()
        return (
            f"🤖 <b>AI ADVISOR</b>\n"
            f"Enabled: <code>{s.get('enabled')}</code>\n"
            f"Model: <code>{s.get('model')}</code>\n"
            f"Daily Calls: <code>{s.get('daily_calls')}/{s.get('max_daily_calls')}</code>\n"
            f"Veto Power: <code>{s.get('veto_power')}</code>\n"
            f"Last Analysis: {s.get('last_portfolio_analysis', 'None')[:100]}"
        )

    async def cmd_risk(args):
        stats = portfolio.db.get_trade_stats(100)
        return (
            f"🛡️ <b>RISK MANAGER</b>\n"
            f"Kill Switch: <code>{'ACTIVE' if risk_manager.kill_switch_active else 'Off'}</code>\n"
            f"Paused: <code>{risk_manager.is_paused}</code>\n"
            f"Kelly: <code>{risk_manager.kelly_fraction*100:.2f}%</code>\n"
            f"Win Rate: <code>{stats.get('win_rate', 0)*100:.1f}%</code>\n"
            f"Avg Win: <code>{stats.get('avg_win', 0)*100:.2f}%</code>\n"
            f"Avg Loss: <code>{stats.get('avg_loss', 0)*100:.2f}%</code>\n"
            f"Consecutive Losses: <code>{portfolio.db.get_consecutive_losses()}</code>\n"
            f"Position Modifier: <code>{risk_manager.position_size_modifier:.1f}x</code>"
        )

    async def cmd_pause(args):
        risk_manager.is_paused = True
        return "⏸️ Trading <b>PAUSED</b>. Use /resume to continue."

    async def cmd_resume(args):
        risk_manager.is_paused = False
        risk_manager.kill_switch_active = False
        risk_manager.position_size_modifier = 1.0
        return "▶️ Trading <b>RESUMED</b>. Kill switch reset."

    async def cmd_sell(args):
        if not args:
            return "Usage: /sell SYMBOL (e.g. /sell SOL)"
        symbol = args[0].upper()
        if not symbol.endswith("USDT"):
            symbol += "USDT"
        result = await portfolio.force_sell(symbol)
        if result:
            return f"Sold {symbol}: PnL={_fmt_pnl(result.get('pnl', result.get('pnl_usdt', 0)))}"
        return f"No open position for {symbol}"

    async def cmd_sellall(args):
        results = await portfolio.force_sell_all()
        return f"Sold {len(results)} positions"

    async def cmd_calibrate(args):
        calibrator.force_calibrate()
        status = calibrator.get_status()
        return (
            f"⚙️ <b>CALIBRATION FORCED</b>\n"
            f"Kelly: <code>{risk_manager.kelly_fraction:.4f}</code>\n"
            f"Win Rate: <code>{risk_manager.win_rate*100:.1f}%</code>\n"
            f"Trades Analyzed: <code>{status.get('trades_analyzed', 0)}</code>"
        )

    async def cmd_health(args):
        status = watchdog.get_status()
        components = "\n".join(
            f"{'✅' if v else '❌'} {k}" for k, v in status.get("components", {}).items()
        )
        return (
            f"🏥 <b>HEALTH</b>\n"
            f"Healthy: {'✅' if status.get('is_healthy') else '❌'}\n"
            f"Memory: <code>{status.get('memory_mb', 0)}MB</code>\n"
            f"CPU: <code>{status.get('cpu_percent', 0)}%</code>\n"
            f"Restarts: <code>{status.get('restart_count', 0)}</code>\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"{components}"
        )

    async def cmd_report(args):
        status = portfolio.get_status()
        daily = portfolio.db.get_daily_pnl()
        return (
            f"📋 <b>QUICK REPORT</b>\n"
            f"Portfolio: <code>${status.get('portfolio_value', 0):,.2f}</code>\n"
            f"Daily P&L: <code>{_fmt_pnl(daily)}</code>\n"
            f"Positions: <code>{status.get('open_positions', 0)}</code>\n"
            f"Regime: <code>{regime.current_regime}</code>\n"
            f"F&G: <code>{news.fear_greed_value}</code>"
        )

    async def cmd_weekly(args):
        weekly_trades = portfolio.db.get_trades_for_period(7)
        weekly_pnl = sum(t.get("pnl_usdt", 0) for t in weekly_trades)
        wins = sum(1 for t in weekly_trades if t.get("pnl_usdt", 0) > 0)
        total = len(weekly_trades)
        wr = f"{wins/total*100:.0f}%" if total > 0 else "N/A"
        return (
            f"📊 <b>WEEKLY SUMMARY</b>\n"
            f"P&L: <code>{_fmt_pnl(weekly_pnl)}</code>\n"
            f"Trades: <code>{total}</code> (Wins: {wins})\n"
            f"Win Rate: <code>{wr}</code>"
        )

    async def cmd_help(args):
        return (
            "<b>COMMANDS</b>\n"
            "/status — Portfolio overview\n"
            "/trades — Open positions with live P&L\n"
            "/pnl — Today's P&L\n"
            "/regime — Market regime info\n"
            "/news — Latest news & sentiment\n"
            "/ai — AI advisor status\n"
            "/risk — Risk manager status\n"
            "/pause — Pause trading\n"
            "/resume — Resume trading\n"
            "/sell SYMBOL — Force sell position\n"
            "/sellall — Force sell all\n"
            "/calibrate — Force recalibration\n"
            "/health — System health\n"
            "/report — Quick daily report\n"
            "/weekly — Weekly performance\n"
            "/help — This message"
        )

    return {
        "/status": cmd_status,
        "/trades": cmd_trades,
        "/positions": cmd_trades,  # alias
        "/pnl": cmd_pnl,
        "/regime": cmd_regime,
        "/news": cmd_news,
        "/ai": cmd_ai,
        "/risk": cmd_risk,
        "/pause": cmd_pause,
        "/resume": cmd_resume,
        "/sell": cmd_sell,
        "/sellall": cmd_sellall,
        "/calibrate": cmd_calibrate,
        "/health": cmd_health,
        "/report": cmd_report,
        "/weekly": cmd_weekly,
        "/help": cmd_help,
    }
