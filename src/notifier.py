"""
Telegram Notifier — Rich visual notifications with charts and dashboards.

Commands:
  /status     — Portfolio dashboard (visual chart)
  /trades     — Open positions with live P&L
  /pnl        — Today's P&L with trade bars chart
  /market     — Market overview dashboard (visual)
  /regime     — Market regime info
  /news       — Latest news sentiment
  /ai         — AI advisor status
  /risk       — Risk manager status
  /chart      — Portfolio P&L history chart
  /pause      — Pause trading
  /resume     — Resume trading
  /sell SYM   — Force sell a position
  /sellall    — Force sell all positions
  /calibrate  — Force recalibration
  /health     — System health
  /report     — Full daily report (visual)
  /weekly     — Weekly report (visual chart)
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

    async def send_photo(self, photo_bytes: bytes, caption: str = ""):
        """Send a photo (PNG bytes) to Telegram chat."""
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
                        logger.warning("Telegram photo send failed (%d): %s", resp.status, body[:200])
        except Exception as e:
            logger.warning("Telegram photo error: %s", e)

    async def send_alert(self, text: str):
        await self.send_message(f"<b>ALERT</b>\n{text}")

    # ------------------------------------------------------------------
    # Trade Notifications (with visual cards)
    # ------------------------------------------------------------------

    async def notify_entry(self, trade: dict):
        """Send trade entry notification with visual card."""
        ai_conf = trade.get("ai_confidence", None)
        ai_str = f"{ai_conf:.0%}" if isinstance(ai_conf, (int, float)) else "N/A"
        usdt_val = trade.get("usdt_value", trade["entry_price"] * trade["quantity"])

        # Send visual card
        try:
            from src.charts import generate_trade_card
            card_data = {
                "symbol": trade["symbol"],
                "price": trade["entry_price"],
                "confluence_score": trade.get("confluence_score", 0),
                "stop_loss": trade.get("stop_loss", 0),
                "take_profit": trade.get("take_profit", 0),
                "position_size_usd": usdt_val,
                "regime": trade.get("regime_at_entry", "N/A"),
                "ai_verdict": ai_str,
            }
            chart_bytes = generate_trade_card(card_data, action="ENTRY")
            caption = (
                f"<b>NEW ENTRY</b> | {trade['symbol']}\n"
                f"${trade['entry_price']:.4f} | Size: ${usdt_val:.2f} | Score: {trade.get('confluence_score', 0)}/100"
            )
            await self.send_photo(chart_bytes, caption)
        except Exception as e:
            logger.warning("Trade card generation failed: %s", e)
            # Fallback to text
            msg = (
                f"<b>NEW ENTRY</b>\n"
                f"Symbol: <code>{trade['symbol']}</code>\n"
                f"Price: <code>${trade['entry_price']:.4f}</code>\n"
                f"Size: <code>${usdt_val:.2f}</code>\n"
                f"SL: <code>${trade['stop_loss']:.4f}</code> | TP: <code>${trade['take_profit']:.4f}</code>\n"
                f"Score: <code>{trade.get('confluence_score', 0)}/100</code> | AI: <code>{ai_str}</code>"
            )
            await self.send_message(msg)

    async def notify_exit(self, result: dict):
        """Send trade exit notification with visual card."""
        pnl = result.get("pnl", result.get("pnl_usdt", 0))
        pnl_pct = result.get("pnl_pct", 0)
        is_full = result.get("is_full_exit", True)
        exit_type = "FULL EXIT" if is_full else "PARTIAL EXIT (50%)"

        try:
            from src.charts import generate_trade_card
            card_data = {
                "symbol": result["symbol"],
                "price": result.get("exit_price", 0),
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "hold_time": result.get("hold_time", "N/A"),
                "exit_reason": result.get("reason", "Unknown"),
            }
            chart_bytes = generate_trade_card(card_data, action="EXIT")
            emoji = "+" if pnl >= 0 else ""
            caption = (
                f"<b>{exit_type}</b> | {result['symbol']}\n"
                f"P&L: {emoji}${pnl:.2f} ({emoji}{pnl_pct:.2f}%) | {result.get('reason', '')}"
            )
            await self.send_photo(chart_bytes, caption)
        except Exception as e:
            logger.warning("Exit card generation failed: %s", e)
            emoji = "+" if pnl >= 0 else ""
            msg = (
                f"<b>{exit_type}</b>\n"
                f"Symbol: <code>{result['symbol']}</code>\n"
                f"Price: <code>${result.get('exit_price', 0):.4f}</code>\n"
                f"PnL: <code>{_fmt_pnl(pnl)}</code> ({_fmt_pct(pnl_pct)})\n"
                f"Reason: <code>{result.get('reason', 'Unknown')}</code>"
            )
            await self.send_message(msg)

    async def notify_regime_change(self, old_regime: str, new_regime: str):
        emoji_map = {"BULL": "GREEN", "BEAR": "RED", "SIDEWAYS": "YELLOW", "HIGH_VOLATILITY": "ORANGE"}
        msg = (
            f"<b>REGIME CHANGE</b>\n"
            f"From: <code>{old_regime}</code>\n"
            f"To: <code>{new_regime}</code>"
        )
        await self.send_message(msg)

    async def notify_calibration(self, cal_data: dict):
        msg = (
            f"<b>CALIBRATION</b>\n"
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
            f"<b>DRAWDOWN WARNING</b>\n"
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
                f"<b>NEWS ALERT</b>\n"
                f"Symbol: <code>{symbol}</code>\n"
                f"Headline: {headline}\n"
                f"Source: <code>{source}</code>\n"
                f"Sentiment: <code>{sent_str}</code>"
            )
            await self.send_message(msg)

    # ------------------------------------------------------------------
    # Visual Daily Report
    # ------------------------------------------------------------------

    async def send_daily_report(self, portfolio, regime, news, risk):
        """Send daily report with portfolio dashboard chart."""
        status = portfolio.get_status()
        daily_pnl = portfolio.db.get_daily_pnl()
        trades_today = portfolio.db.get_trades_for_period(1)
        wins = sum(1 for t in trades_today if t.get("pnl_usdt", 0) > 0)
        losses = sum(1 for t in trades_today if t.get("pnl_usdt", 0) < 0)

        # Build positions data for chart
        positions_data = []
        for p in portfolio.open_positions:
            try:
                price = await portfolio.exchange.get_price(p["symbol"])
            except Exception:
                price = p.get("entry_price", 0)
            value = price * p.get("quantity", 0)
            pnl_pct = ((price / p["entry_price"]) - 1) * 100 if p.get("entry_price", 0) > 0 else 0
            positions_data.append({
                "symbol": p["symbol"],
                "value": value,
                "unrealized_pnl_pct": pnl_pct,
            })

        try:
            from src.charts import generate_portfolio_dashboard
            chart_data = {
                "portfolio_value": status.get("portfolio_value", 0),
                "cash_available": status.get("cash_available", 0),
                "daily_pnl": daily_pnl,
                "daily_pnl_pct": (daily_pnl / max(status.get("portfolio_value", 1), 1)) * 100,
                "positions": positions_data,
                "regime": regime.current_regime,
                "fear_greed": news.fear_greed_value,
                "current_drawdown": status.get("drawdown", 0),
                "max_drawdown_limit": Settings.risk.MAX_PORTFOLIO_DRAWDOWN,
                "daily_loss": abs(daily_pnl) if daily_pnl < 0 else 0,
                "max_daily_loss": Settings.risk.MAX_DAILY_LOSS,
                "max_positions": Settings.risk.MAX_OPEN_POSITIONS,
            }
            chart_bytes = generate_portfolio_dashboard(chart_data)
            caption = (
                f"<b>DAILY REPORT</b> | {utc_now().strftime('%Y-%m-%d')}\n"
                f"Value: ${status.get('portfolio_value', 0):,.2f} | "
                f"P&L: {_fmt_pnl(daily_pnl)} | "
                f"Trades: {len(trades_today)} (W:{wins} L:{losses})"
            )
            await self.send_photo(chart_bytes, caption)
        except Exception as e:
            logger.warning("Dashboard chart failed: %s", e)
            # Fallback to text report
            msg = (
                f"<b>DAILY REPORT</b> | {utc_now().strftime('%Y-%m-%d')}\n"
                f"Portfolio: <code>${status.get('portfolio_value', 0):,.2f}</code>\n"
                f"Cash: <code>${status.get('cash_available', 0):,.2f}</code>\n"
                f"Daily P&L: <code>{_fmt_pnl(daily_pnl)}</code>\n"
                f"Trades: <code>{len(trades_today)}</code> (W:{wins} L:{losses})\n"
                f"Regime: <code>{regime.current_regime}</code>\n"
                f"F&G: <code>{news.fear_greed_value}</code>"
            )
            await self.send_message(msg)

    # ------------------------------------------------------------------
    # Visual Weekly Report
    # ------------------------------------------------------------------

    async def send_weekly_report(self, portfolio, regime, news, risk,
                                  calibrator, ai):
        """Send weekly report with visual performance chart."""
        status = portfolio.get_status()
        weekly_trades = portfolio.db.get_trades_for_period(7)
        weekly_pnl = sum(t.get("pnl_usdt", 0) for t in weekly_trades)
        wins = sum(1 for t in weekly_trades if t.get("pnl_usdt", 0) > 0)
        losses = sum(1 for t in weekly_trades if t.get("pnl_usdt", 0) < 0)
        total = wins + losses

        best = max(weekly_trades, key=lambda t: t.get("pnl_usdt", 0), default=None)
        worst = min(weekly_trades, key=lambda t: t.get("pnl_usdt", 0), default=None)

        # Sector P&L
        sector_pnl = {}
        for t in weekly_trades:
            sector = Settings.get_sector_for_asset(t.get("symbol", ""))
            sector_pnl[sector] = sector_pnl.get(sector, 0) + t.get("pnl_usdt", 0)

        # Daily P&L breakdown
        daily_pnls = []
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        from datetime import timedelta
        now = utc_now()
        for i in range(6, -1, -1):
            day = now - timedelta(days=i)
            day_trades = [
                t for t in weekly_trades
                if str(t.get("close_time", "")).startswith(day.strftime("%Y-%m-%d"))
            ]
            day_pnl = sum(t.get("pnl_usdt", 0) for t in day_trades)
            daily_pnls.append({
                "day": day.strftime("%a"),
                "pnl": day_pnl,
            })

        try:
            from src.charts import generate_weekly_report
            weekly_data = {
                "total_pnl": weekly_pnl,
                "total_pnl_pct": (weekly_pnl / max(status.get("portfolio_value", 1), 1)) * 100,
                "wins": wins,
                "losses": losses,
                "best_trade": {
                    "symbol": best.get("symbol", "?") if best else "?",
                    "pnl_pct": best.get("pnl_pct", 0) if best else 0,
                } if best else {},
                "worst_trade": {
                    "symbol": worst.get("symbol", "?") if worst else "?",
                    "pnl_pct": worst.get("pnl_pct", 0) if worst else 0,
                } if worst else {},
                "daily_pnls": daily_pnls,
                "sector_pnl": sector_pnl,
            }
            chart_bytes = generate_weekly_report(weekly_data)
            wr = f"{wins/total*100:.0f}%" if total > 0 else "N/A"
            caption = (
                f"<b>WEEKLY REPORT</b> | {utc_now().strftime('%Y-%m-%d')}\n"
                f"P&L: {_fmt_pnl(weekly_pnl)} | Trades: {total} | Win Rate: {wr}"
            )
            await self.send_photo(chart_bytes, caption)
        except Exception as e:
            logger.warning("Weekly chart failed: %s", e)
            wr = f"{wins/total*100:.0f}%" if total > 0 else "N/A"
            msg = (
                f"<b>WEEKLY REPORT</b>\n"
                f"P&L: <code>{_fmt_pnl(weekly_pnl)}</code>\n"
                f"Trades: <code>{total}</code> (W:{wins} L:{losses})\n"
                f"Win Rate: <code>{wr}</code>"
            )
            await self.send_message(msg)

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
                elif isinstance(result, bytes):
                    # Handler returned chart bytes
                    await self.send_photo(result)
                elif isinstance(result, tuple) and len(result) == 2:
                    # Handler returned (bytes, caption)
                    await self.send_photo(result[0], result[1])
            except Exception as e:
                await self.send_message(f"Command error: {e}")
        else:
            available = ", ".join(sorted(self.command_handlers.keys()))
            await self.send_message(f"Unknown command. Available: {available}")


# ------------------------------------------------------------------
# Command Handler Factory — with Visual Charts
# ------------------------------------------------------------------

def build_command_handlers(portfolio, risk_manager, calibrator, watchdog,
                           regime, news, ai) -> dict:

    async def cmd_status(args):
        """Visual portfolio dashboard."""
        status = portfolio.get_status()
        positions_data = []
        for p in portfolio.open_positions:
            try:
                price = await portfolio.exchange.get_price(p["symbol"])
            except Exception:
                price = p.get("entry_price", 0)
            value = price * p.get("quantity", 0)
            pnl_pct = ((price / p["entry_price"]) - 1) * 100 if p.get("entry_price", 0) > 0 else 0
            positions_data.append({
                "symbol": p["symbol"],
                "value": value,
                "unrealized_pnl_pct": pnl_pct,
            })

        try:
            from src.charts import generate_portfolio_dashboard
            daily_pnl = portfolio.db.get_daily_pnl()
            chart_data = {
                "portfolio_value": status.get("portfolio_value", 0),
                "cash_available": status.get("cash_available", 0),
                "daily_pnl": daily_pnl,
                "daily_pnl_pct": (daily_pnl / max(status.get("portfolio_value", 1), 1)) * 100,
                "positions": positions_data,
                "regime": regime.current_regime,
                "fear_greed": news.fear_greed_value,
                "current_drawdown": status.get("drawdown", 0),
                "max_drawdown_limit": Settings.risk.MAX_PORTFOLIO_DRAWDOWN,
                "daily_loss": abs(daily_pnl) if daily_pnl < 0 else 0,
                "max_daily_loss": Settings.risk.MAX_DAILY_LOSS,
                "max_positions": Settings.risk.MAX_OPEN_POSITIONS,
            }
            chart_bytes = generate_portfolio_dashboard(chart_data)
            caption = (
                f"<b>PORTFOLIO DASHBOARD</b>\n"
                f"${status.get('portfolio_value', 0):,.2f} | "
                f"{regime.current_regime} | F&G: {news.fear_greed_value}"
            )
            return (chart_bytes, caption)
        except Exception as e:
            logger.warning("Dashboard chart failed in /status: %s", e)
            return (
                f"<b>STATUS</b>\n"
                f"Portfolio: <code>${status.get('portfolio_value', 0):,.2f}</code>\n"
                f"Cash: <code>${status.get('cash_available', 0):,.2f}</code>\n"
                f"Positions: <code>{status.get('open_positions', 0)}</code>\n"
                f"Drawdown: <code>{status.get('drawdown', 0)*100:.1f}%</code>\n"
                f"Regime: <code>{regime.current_regime}</code>\n"
                f"F&G: <code>{news.fear_greed_value}</code>"
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
            emoji = "+" if pnl_pct >= 0 else ""
            exits = p.get("tranche_exits", [])
            if isinstance(exits, str):
                exits = json.loads(exits)
            partial = " (50% sold)" if len(exits) > 0 else ""
            lines.append(
                f"<code>{p['symbol']}</code> @ ${p['entry_price']:.4f} "
                f"-> ${price:.4f} ({emoji}{pnl_pct:.2f}%){partial}\n"
                f"   SL=${p['stop_loss']:.4f}"
            )
        return "\n".join(lines)

    async def cmd_pnl(args):
        """Today's P&L with trade bars chart."""
        daily = portfolio.db.get_daily_pnl()
        trades = portfolio.db.get_trades_for_period(1)
        wins = sum(1 for t in trades if t.get("pnl_usdt", 0) > 0)
        losses = sum(1 for t in trades if t.get("pnl_usdt", 0) < 0)

        if trades:
            try:
                from src.charts import generate_trade_bars
                chart_bytes = generate_trade_bars(trades, last_n=20)
                caption = (
                    f"<b>TODAY'S P&L</b>\n"
                    f"Realized: {_fmt_pnl(daily)} | Trades: {len(trades)} (W:{wins} L:{losses})"
                )
                return (chart_bytes, caption)
            except Exception:
                pass

        return (
            f"<b>TODAY'S P&L</b>\n"
            f"Realized: <code>{_fmt_pnl(daily)}</code>\n"
            f"Trades: <code>{len(trades)}</code> (W:{wins} L:{losses})"
        )

    async def cmd_market(args):
        """Visual market overview dashboard."""
        try:
            from src.charts import generate_market_overview
            # Get top movers from exchange if available
            top_movers = []
            try:
                tickers = await portfolio.exchange.get_all_tickers()
                if tickers:
                    sorted_tickers = sorted(tickers, key=lambda x: float(x.get("priceChangePercent", 0)), reverse=True)
                    for t in sorted_tickers[:5]:
                        top_movers.append({
                            "symbol": t.get("symbol", ""),
                            "change_pct": float(t.get("priceChangePercent", 0)),
                        })
                    for t in sorted_tickers[-5:]:
                        top_movers.append({
                            "symbol": t.get("symbol", ""),
                            "change_pct": float(t.get("priceChangePercent", 0)),
                        })
            except Exception:
                pass

            chart_bytes = generate_market_overview(
                regime=regime.current_regime,
                fear_greed=news.fear_greed_value,
                btc_dominance=getattr(news, "btc_dominance", 0),
                sentiment_score=news.get_sentiment_score(),
                top_movers=top_movers,
            )
            caption = (
                f"<b>MARKET OVERVIEW</b>\n"
                f"Regime: {regime.current_regime} | F&G: {news.fear_greed_value}"
            )
            return (chart_bytes, caption)
        except Exception as e:
            logger.warning("Market chart failed: %s", e)
            return (
                f"<b>MARKET</b>\n"
                f"Regime: <code>{regime.current_regime}</code>\n"
                f"F&G: <code>{news.fear_greed_value}</code>\n"
                f"Sentiment: <code>{news.get_sentiment_score():.2f}</code>"
            )

    async def cmd_chart(args):
        """P&L history line chart."""
        try:
            from src.charts import generate_pnl_chart
            all_trades = portfolio.db.get_trades_for_period(30)
            chart_bytes = generate_pnl_chart(all_trades, days=30)
            return (chart_bytes, "<b>P&L HISTORY</b> (Last 30 Days)")
        except Exception as e:
            logger.warning("P&L chart failed: %s", e)
            return "Chart generation failed. Try /pnl for text summary."

    async def cmd_regime(args):
        params = regime.get_regime_params()
        return (
            f"<b>MARKET REGIME</b>\n"
            f"Current: <code>{regime.current_regime}</code>\n"
            f"Entries: <code>{'Allowed' if params.get('entries_allowed') else 'Blocked'}</code>\n"
            f"Max Exposure: <code>{params.get('max_exposure', 0)*100:.0f}%</code>\n"
            f"Position Mult: <code>{params.get('position_multiplier', 1):.1f}x</code>"
        )

    async def cmd_news(args):
        headlines = news.get_top_headlines(5)
        if not headlines:
            return f"No recent news. F&G: {news.fear_greed_value}"
        lines = [f"<b>TOP NEWS</b> (F&G: {news.fear_greed_value} - {news.fear_greed_label})"]
        for h in headlines:
            sent = h.get("sentiment", 0)
            if isinstance(sent, str):
                indicator = "[+]" if sent == "positive" else "[-]" if sent == "negative" else "[=]"
            else:
                indicator = "[+]" if sent > 0 else "[-]" if sent < 0 else "[=]"
            lines.append(f"{indicator} {h.get('title', 'N/A')[:80]}")
        return "\n".join(lines)

    async def cmd_ai(args):
        s = ai.get_status()
        stats = s.get('model_stats', {})
        fast_stats = stats.get('fast', {})
        strong_stats = stats.get('strong', {})
        return (
            f"<b>AI ADVISOR</b>\n\n"
            f"Enabled: <code>{s.get('enabled')}</code>\n"
            f"Veto Power: <code>{s.get('veto_power')}</code>\n\n"
            f"<b>Models</b>\n"
            f"Fast: <code>{s.get('fast_model', 'N/A')}</code>\n"
            f"  Calls: <code>{fast_stats.get('calls', 0)}</code> | Errors: <code>{fast_stats.get('errors', 0)}</code>\n"
            f"Strong: <code>{s.get('strong_model', 'N/A')}</code>\n"
            f"  Calls: <code>{strong_stats.get('calls', 0)}</code> | Errors: <code>{strong_stats.get('errors', 0)}</code>\n\n"
            f"<b>Usage</b>\n"
            f"Daily Calls: <code>{s.get('daily_calls', 0)}/{s.get('max_daily_calls', 500)}</code>\n"
            f"Cost: <code>${s.get('daily_cost_usd', 0):.4f}</code> (Groq = FREE)\n"
            f"Recent Queue: <code>{s.get('recent_calls', 0)}</code>\n\n"
            f"<b>Last Analysis</b>\n"
            f"{s.get('last_portfolio_analysis', 'None')[:200]}"
        )

    async def cmd_risk(args):
        stats = portfolio.db.get_trade_stats(100)
        return (
            f"<b>RISK MANAGER</b>\n"
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
        return "Trading <b>PAUSED</b>. Use /resume to continue."

    async def cmd_resume(args):
        risk_manager.is_paused = False
        risk_manager.kill_switch_active = False
        risk_manager.position_size_modifier = 1.0
        return "Trading <b>RESUMED</b>. Kill switch reset."

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
            f"<b>CALIBRATION FORCED</b>\n"
            f"Kelly: <code>{risk_manager.kelly_fraction:.4f}</code>\n"
            f"Win Rate: <code>{risk_manager.win_rate*100:.1f}%</code>\n"
            f"Trades Analyzed: <code>{status.get('trades_analyzed', 0)}</code>"
        )

    async def cmd_health(args):
        status = watchdog.get_status()
        components = "\n".join(
            f"{'OK' if v else 'FAIL'} {k}" for k, v in status.get("components", {}).items()
        )
        return (
            f"<b>HEALTH</b>\n"
            f"Healthy: {'OK' if status.get('is_healthy') else 'FAIL'}\n"
            f"Memory: <code>{status.get('memory_mb', 0)}MB</code>\n"
            f"CPU: <code>{status.get('cpu_percent', 0)}%</code>\n"
            f"Restarts: <code>{status.get('restart_count', 0)}</code>\n"
            f"---\n"
            f"{components}"
        )

    async def cmd_report(args):
        """Full daily report with visual dashboard."""
        status = portfolio.get_status()
        daily = portfolio.db.get_daily_pnl()
        trades = portfolio.db.get_trades_for_period(1)
        wins = sum(1 for t in trades if t.get("pnl_usdt", 0) > 0)
        losses = sum(1 for t in trades if t.get("pnl_usdt", 0) < 0)

        positions_data = []
        for p in portfolio.open_positions:
            try:
                price = await portfolio.exchange.get_price(p["symbol"])
            except Exception:
                price = p.get("entry_price", 0)
            value = price * p.get("quantity", 0)
            pnl_pct = ((price / p["entry_price"]) - 1) * 100 if p.get("entry_price", 0) > 0 else 0
            positions_data.append({
                "symbol": p["symbol"],
                "value": value,
                "unrealized_pnl_pct": pnl_pct,
            })

        try:
            from src.charts import generate_portfolio_dashboard
            chart_data = {
                "portfolio_value": status.get("portfolio_value", 0),
                "cash_available": status.get("cash_available", 0),
                "daily_pnl": daily,
                "daily_pnl_pct": (daily / max(status.get("portfolio_value", 1), 1)) * 100,
                "positions": positions_data,
                "regime": regime.current_regime,
                "fear_greed": news.fear_greed_value,
                "current_drawdown": status.get("drawdown", 0),
                "max_drawdown_limit": Settings.risk.MAX_PORTFOLIO_DRAWDOWN,
                "daily_loss": abs(daily) if daily < 0 else 0,
                "max_daily_loss": Settings.risk.MAX_DAILY_LOSS,
                "max_positions": Settings.risk.MAX_OPEN_POSITIONS,
            }
            chart_bytes = generate_portfolio_dashboard(chart_data)
            caption = (
                f"<b>DAILY REPORT</b> | {utc_now().strftime('%Y-%m-%d')}\n"
                f"${status.get('portfolio_value', 0):,.2f} | P&L: {_fmt_pnl(daily)} | "
                f"Trades: {len(trades)} (W:{wins} L:{losses})"
            )
            return (chart_bytes, caption)
        except Exception:
            return (
                f"<b>DAILY REPORT</b>\n"
                f"Portfolio: <code>${status.get('portfolio_value', 0):,.2f}</code>\n"
                f"P&L: <code>{_fmt_pnl(daily)}</code>\n"
                f"Trades: <code>{len(trades)}</code> (W:{wins} L:{losses})"
            )

    async def cmd_weekly(args):
        """Visual weekly report."""
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

        from datetime import timedelta
        daily_pnls = []
        now = utc_now()
        for i in range(6, -1, -1):
            day = now - timedelta(days=i)
            day_trades = [
                t for t in weekly_trades
                if str(t.get("close_time", "")).startswith(day.strftime("%Y-%m-%d"))
            ]
            day_pnl = sum(t.get("pnl_usdt", 0) for t in day_trades)
            daily_pnls.append({"day": day.strftime("%a"), "pnl": day_pnl})

        try:
            from src.charts import generate_weekly_report
            weekly_data = {
                "total_pnl": weekly_pnl,
                "total_pnl_pct": (weekly_pnl / max(portfolio.get_status().get("portfolio_value", 1), 1)) * 100,
                "wins": wins,
                "losses": losses,
                "best_trade": {"symbol": best.get("symbol", "?"), "pnl_pct": best.get("pnl_pct", 0)} if best else {},
                "worst_trade": {"symbol": worst.get("symbol", "?"), "pnl_pct": worst.get("pnl_pct", 0)} if worst else {},
                "daily_pnls": daily_pnls,
                "sector_pnl": sector_pnl,
            }
            chart_bytes = generate_weekly_report(weekly_data)
            wr = f"{wins/total*100:.0f}%" if total > 0 else "N/A"
            caption = (
                f"<b>WEEKLY REPORT</b> | {utc_now().strftime('%Y-%m-%d')}\n"
                f"P&L: {_fmt_pnl(weekly_pnl)} | Trades: {total} | Win Rate: {wr}"
            )
            return (chart_bytes, caption)
        except Exception:
            wr = f"{wins/total*100:.0f}%" if total > 0 else "N/A"
            return (
                f"<b>WEEKLY REPORT</b>\n"
                f"P&L: <code>{_fmt_pnl(weekly_pnl)}</code>\n"
                f"Trades: <code>{total}</code> (W:{wins} L:{losses})\n"
                f"Win Rate: <code>{wr}</code>"
            )

    async def cmd_help(args):
        return (
            "<b>COMMANDS</b>\n\n"
            "<b>Dashboards (Visual)</b>\n"
            "/status - Portfolio dashboard\n"
            "/market - Market overview\n"
            "/chart - P&L history chart\n"
            "/report - Full daily report\n"
            "/weekly - Weekly performance\n\n"
            "<b>Information</b>\n"
            "/trades - Open positions\n"
            "/pnl - Today's P&L\n"
            "/regime - Market regime\n"
            "/news - Latest news\n"
            "/ai - AI advisor status\n"
            "/risk - Risk metrics\n"
            "/health - System health\n\n"
            "<b>Actions</b>\n"
            "/pause - Pause trading\n"
            "/resume - Resume trading\n"
            "/sell SYMBOL - Force sell\n"
            "/sellall - Sell everything\n"
            "/calibrate - Force recalibration"
        )

    return {
        "/status": cmd_status,
        "/trades": cmd_trades,
        "/positions": cmd_trades,
        "/pnl": cmd_pnl,
        "/market": cmd_market,
        "/chart": cmd_chart,
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
