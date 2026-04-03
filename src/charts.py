"""
Charts Module — Generate visual dashboards and charts for Telegram.

Creates portfolio pie charts, P&L curves, trade history bars,
market regime gauges, and position summary cards using matplotlib.
All charts are saved as PNG and sent via Telegram photo API.
"""

import io
import os
import math
from datetime import datetime, timedelta, timezone
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np

from src.utils import setup_logging

logger = setup_logging()

# --- Dark theme for all charts ---
DARK_BG = "#0d1117"
CARD_BG = "#161b22"
GREEN = "#00d26a"
RED = "#ff4757"
BLUE = "#58a6ff"
YELLOW = "#f0c929"
ORANGE = "#ff9f43"
PURPLE = "#b388ff"
CYAN = "#00e5ff"
WHITE = "#e6edf3"
GRAY = "#8b949e"
GRID_COLOR = "#21262d"

SECTOR_COLORS = {
    "Layer1": "#58a6ff",
    "Layer2": "#b388ff",
    "DeFi": "#00d26a",
    "Meme": "#ff4757",
    "AI": "#00e5ff",
    "Gaming": "#ff9f43",
    "Storage": "#f0c929",
    "Oracle": "#e040fb",
    "Privacy": "#7c4dff",
    "Exchange": "#ff6e40",
    "Stablecoin": "#69f0ae",
    "Other": "#8b949e",
    "Cash": "#455a64",
}


def _apply_dark_theme(fig, ax_or_axes):
    """Apply consistent dark theme to figure and axes."""
    fig.patch.set_facecolor(DARK_BG)
    axes = ax_or_axes if hasattr(ax_or_axes, "__iter__") else [ax_or_axes]
    for ax in axes:
        ax.set_facecolor(CARD_BG)
        ax.tick_params(colors=WHITE, labelsize=9)
        ax.xaxis.label.set_color(WHITE)
        ax.yaxis.label.set_color(WHITE)
        ax.title.set_color(WHITE)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)


def _fig_to_bytes(fig) -> bytes:
    """Convert matplotlib figure to PNG bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# 1. Portfolio Dashboard
# ---------------------------------------------------------------------------

def generate_portfolio_dashboard(portfolio_data: dict) -> bytes:
    """
    Full portfolio dashboard with:
    - Portfolio value & daily P&L header
    - Allocation pie chart
    - Open positions table
    - Risk metrics bar
    """
    fig = plt.figure(figsize=(12, 8))
    fig.patch.set_facecolor(DARK_BG)
    gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

    portfolio_value = portfolio_data.get("portfolio_value", 0)
    cash = portfolio_data.get("cash_available", 0)
    daily_pnl = portfolio_data.get("daily_pnl", 0)
    daily_pnl_pct = portfolio_data.get("daily_pnl_pct", 0)
    positions = portfolio_data.get("positions", [])
    regime = portfolio_data.get("regime", "UNKNOWN")
    fear_greed = portfolio_data.get("fear_greed", 50)

    # --- Header ---
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.set_facecolor(DARK_BG)
    ax_header.axis("off")

    pnl_color = GREEN if daily_pnl >= 0 else RED
    pnl_sign = "+" if daily_pnl >= 0 else ""

    ax_header.text(0.02, 0.85, "PORTFOLIO DASHBOARD", fontsize=18, fontweight="bold",
                   color=CYAN, transform=ax_header.transAxes, va="top")
    ax_header.text(0.02, 0.55, f"${portfolio_value:,.2f}", fontsize=28, fontweight="bold",
                   color=WHITE, transform=ax_header.transAxes, va="top")
    ax_header.text(0.35, 0.55, f"{pnl_sign}${daily_pnl:,.2f} ({pnl_sign}{daily_pnl_pct:.2f}%)",
                   fontsize=16, color=pnl_color, transform=ax_header.transAxes, va="top")

    # Regime badge
    regime_colors = {"BULL": GREEN, "BEAR": RED, "SIDEWAYS": YELLOW, "HIGH_VOLATILITY": ORANGE}
    ax_header.text(0.75, 0.85, f"Regime: {regime}", fontsize=12, fontweight="bold",
                   color=regime_colors.get(regime, GRAY), transform=ax_header.transAxes, va="top",
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=CARD_BG, edgecolor=regime_colors.get(regime, GRAY)))

    # Fear & Greed
    fg_color = RED if fear_greed < 25 else ORANGE if fear_greed < 45 else YELLOW if fear_greed < 55 else GREEN if fear_greed < 75 else RED
    ax_header.text(0.75, 0.45, f"Fear & Greed: {fear_greed}", fontsize=11,
                   color=fg_color, transform=ax_header.transAxes, va="top")

    # --- Allocation Pie Chart ---
    ax_pie = fig.add_subplot(gs[1, 0])
    ax_pie.set_facecolor(DARK_BG)

    if positions:
        labels = [p.get("symbol", "?").replace("USDT", "") for p in positions]
        sizes = [p.get("value", 0) for p in positions]
        if cash > 0:
            labels.append("CASH")
            sizes.append(cash)

        colors = []
        for lbl in labels:
            sector = _get_sector(lbl)
            colors.append(SECTOR_COLORS.get(sector, BLUE))

        wedges, texts, autotexts = ax_pie.pie(
            sizes, labels=None, autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
            colors=colors, startangle=90, pctdistance=0.8,
            textprops={"color": WHITE, "fontsize": 8},
        )
        ax_pie.legend(labels, loc="center left", bbox_to_anchor=(-0.15, 0.5),
                      fontsize=7, facecolor=CARD_BG, edgecolor=GRID_COLOR,
                      labelcolor=WHITE)
    else:
        ax_pie.pie([1], labels=["CASH"], colors=[SECTOR_COLORS["Cash"]],
                    textprops={"color": WHITE})

    ax_pie.set_title("Allocation", fontsize=12, color=WHITE, fontweight="bold")

    # --- Positions Table ---
    ax_table = fig.add_subplot(gs[1, 1])
    ax_table.set_facecolor(CARD_BG)
    ax_table.axis("off")
    ax_table.set_title("Open Positions", fontsize=12, color=WHITE, fontweight="bold")

    if positions:
        table_data = []
        for p in positions[:8]:
            sym = p.get("symbol", "?").replace("USDT", "")
            pnl = p.get("unrealized_pnl_pct", 0)
            pnl_str = f"{'+' if pnl >= 0 else ''}{pnl:.2f}%"
            value = f"${p.get('value', 0):,.0f}"
            table_data.append([sym, value, pnl_str])

        table = ax_table.table(
            cellText=table_data,
            colLabels=["Asset", "Value", "P&L"],
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)

        for (row, col), cell in table.get_celld().items():
            cell.set_facecolor(CARD_BG)
            cell.set_edgecolor(GRID_COLOR)
            if row == 0:
                cell.set_text_props(color=CYAN, fontweight="bold")
            else:
                text = cell.get_text().get_text()
                if "%" in text:
                    color = GREEN if text.startswith("+") else RED
                    cell.set_text_props(color=color)
                else:
                    cell.set_text_props(color=WHITE)
    else:
        ax_table.text(0.5, 0.5, "No open positions", fontsize=12, color=GRAY,
                      ha="center", va="center", transform=ax_table.transAxes)

    # --- Risk Metrics Bar ---
    ax_risk = fig.add_subplot(gs[2, :])
    ax_risk.set_facecolor(CARD_BG)
    ax_risk.axis("off")
    ax_risk.set_title("Risk Metrics", fontsize=12, color=WHITE, fontweight="bold")

    drawdown = portfolio_data.get("current_drawdown", 0)
    max_dd = portfolio_data.get("max_drawdown_limit", 0.25)
    daily_loss = abs(portfolio_data.get("daily_loss", 0))
    max_daily = portfolio_data.get("max_daily_loss", 0.05)
    open_count = len(positions)
    max_positions = portfolio_data.get("max_positions", 15)

    metrics = [
        ("Drawdown", drawdown, max_dd),
        ("Daily Loss", daily_loss, max_daily),
        ("Positions", open_count / max(max_positions, 1), 1.0),
    ]

    bar_width = 0.25
    for i, (label, current, maximum) in enumerate(metrics):
        x_start = 0.05 + i * 0.33
        fill_pct = min(current / max(maximum, 0.001), 1.0) if isinstance(current, float) else current

        bar_color = GREEN if fill_pct < 0.5 else YELLOW if fill_pct < 0.75 else RED

        ax_risk.barh(0, fill_pct * 0.28, left=x_start, height=0.4,
                     color=bar_color, alpha=0.8)
        ax_risk.barh(0, 0.28, left=x_start, height=0.4,
                     color=GRID_COLOR, alpha=0.3)

        if label == "Positions":
            display = f"{open_count}/{max_positions}"
        else:
            display = f"{current:.1%}"

        ax_risk.text(x_start + 0.14, 0.6, label, fontsize=10, color=WHITE,
                     ha="center", va="center", fontweight="bold")
        ax_risk.text(x_start + 0.14, -0.5, display, fontsize=10, color=bar_color,
                     ha="center", va="center")

    ax_risk.set_xlim(0, 1)
    ax_risk.set_ylim(-1, 1)

    return _fig_to_bytes(fig)


# ---------------------------------------------------------------------------
# 2. P&L History Chart
# ---------------------------------------------------------------------------

def generate_pnl_chart(trades: list, days: int = 30) -> bytes:
    """Line chart showing cumulative P&L over time."""
    fig, ax = plt.subplots(figsize=(10, 5))
    _apply_dark_theme(fig, ax)

    if not trades:
        ax.text(0.5, 0.5, "No trade history yet", fontsize=14, color=GRAY,
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Cumulative P&L", fontsize=14, fontweight="bold")
        return _fig_to_bytes(fig)

    dates = []
    cum_pnl = []
    running = 0.0
    for t in sorted(trades, key=lambda x: x.get("close_time", x.get("open_time", ""))):
        pnl = t.get("pnl_usdt", t.get("realized_pnl", 0))
        running += pnl
        ts = t.get("close_time", t.get("open_time", ""))
        try:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            dt = datetime.now(timezone.utc)
        dates.append(dt)
        cum_pnl.append(running)

    # Fill color based on positive/negative
    ax.plot(dates, cum_pnl, color=CYAN, linewidth=2, zorder=3)
    ax.fill_between(dates, cum_pnl, 0,
                    where=[p >= 0 for p in cum_pnl], color=GREEN, alpha=0.15)
    ax.fill_between(dates, cum_pnl, 0,
                    where=[p < 0 for p in cum_pnl], color=RED, alpha=0.15)

    ax.axhline(y=0, color=GRAY, linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title(f"Cumulative P&L (Last {days} Days)", fontsize=14, fontweight="bold")
    ax.set_ylabel("P&L ($)", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.grid(True, color=GRID_COLOR, alpha=0.3)

    # Annotate final value
    if cum_pnl:
        final = cum_pnl[-1]
        color = GREEN if final >= 0 else RED
        ax.annotate(f"${final:+,.2f}", xy=(dates[-1], final),
                    fontsize=12, fontweight="bold", color=color,
                    xytext=(10, 10), textcoords="offset points")

    return _fig_to_bytes(fig)


# ---------------------------------------------------------------------------
# 3. Trade History Bar Chart
# ---------------------------------------------------------------------------

def generate_trade_bars(trades: list, last_n: int = 20) -> bytes:
    """Bar chart showing individual trade P&L for the last N trades."""
    fig, ax = plt.subplots(figsize=(10, 5))
    _apply_dark_theme(fig, ax)

    recent = sorted(trades, key=lambda x: x.get("close_time", ""))[-last_n:]

    if not recent:
        ax.text(0.5, 0.5, "No completed trades yet", fontsize=14, color=GRAY,
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Recent Trades", fontsize=14, fontweight="bold")
        return _fig_to_bytes(fig)

    symbols = [t.get("symbol", "?").replace("USDT", "")[:6] for t in recent]
    pnls = [t.get("pnl_pct", t.get("pnl_usdt", 0)) for t in recent]
    colors = [GREEN if p >= 0 else RED for p in pnls]

    bars = ax.bar(range(len(symbols)), pnls, color=colors, alpha=0.85, edgecolor=GRID_COLOR)
    ax.set_xticks(range(len(symbols)))
    ax.set_xticklabels(symbols, rotation=45, ha="right", fontsize=8)
    ax.axhline(y=0, color=GRAY, linestyle="--", linewidth=0.8)
    ax.set_title(f"Last {len(recent)} Trades — P&L %", fontsize=14, fontweight="bold")
    ax.set_ylabel("P&L (%)", fontsize=11)
    ax.grid(True, axis="y", color=GRID_COLOR, alpha=0.3)

    # Win rate annotation
    wins = sum(1 for p in pnls if p > 0)
    total = len(pnls)
    wr = (wins / total * 100) if total > 0 else 0
    avg_pnl = sum(pnls) / total if total else 0
    ax.text(0.98, 0.95, f"Win Rate: {wr:.0f}% | Avg: {avg_pnl:+.2f}%",
            fontsize=11, color=CYAN, ha="right", va="top",
            transform=ax.transAxes, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=CARD_BG, edgecolor=CYAN, alpha=0.8))

    return _fig_to_bytes(fig)


# ---------------------------------------------------------------------------
# 4. Market Overview Chart
# ---------------------------------------------------------------------------

def generate_market_overview(regime: str, fear_greed: int, btc_dominance: float,
                             sentiment_score: float, top_movers: list = None) -> bytes:
    """Visual market overview with regime, fear/greed gauge, and top movers."""
    fig = plt.figure(figsize=(10, 6))
    fig.patch.set_facecolor(DARK_BG)
    gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

    # --- Fear & Greed Gauge ---
    ax_gauge = fig.add_subplot(gs[0, 0], projection="polar")
    ax_gauge.set_facecolor(DARK_BG)

    theta = np.linspace(0, np.pi, 100)
    r = np.ones(100)

    # Color gradient: red -> orange -> yellow -> green
    colors_gradient = plt.cm.RdYlGn(np.linspace(0, 1, 100))
    for i in range(len(theta) - 1):
        ax_gauge.fill_between([theta[i], theta[i + 1]], 0, 1,
                              color=colors_gradient[i], alpha=0.6)

    # Needle
    needle_angle = np.pi * (1 - fear_greed / 100)
    ax_gauge.plot([needle_angle, needle_angle], [0, 0.85], color=WHITE, linewidth=3)
    ax_gauge.plot(needle_angle, 0.85, "o", color=WHITE, markersize=6)

    ax_gauge.set_ylim(0, 1.1)
    ax_gauge.set_thetamin(0)
    ax_gauge.set_thetamax(180)
    ax_gauge.set_yticklabels([])
    ax_gauge.set_xticklabels([])
    ax_gauge.grid(False)
    ax_gauge.spines["polar"].set_visible(False)

    fg_label = (
        "Extreme Fear" if fear_greed < 25
        else "Fear" if fear_greed < 45
        else "Neutral" if fear_greed < 55
        else "Greed" if fear_greed < 75
        else "Extreme Greed"
    )
    ax_gauge.set_title(f"Fear & Greed: {fear_greed}\n{fg_label}",
                       fontsize=13, color=WHITE, fontweight="bold", pad=10)

    # --- Regime + BTC Dominance ---
    ax_info = fig.add_subplot(gs[0, 1])
    ax_info.set_facecolor(CARD_BG)
    ax_info.axis("off")

    regime_colors = {"BULL": GREEN, "BEAR": RED, "SIDEWAYS": YELLOW, "HIGH_VOLATILITY": ORANGE}
    rc = regime_colors.get(regime, GRAY)

    ax_info.text(0.5, 0.85, "Market Regime", fontsize=11, color=GRAY,
                 ha="center", va="top", transform=ax_info.transAxes)
    ax_info.text(0.5, 0.6, regime, fontsize=24, color=rc, fontweight="bold",
                 ha="center", va="top", transform=ax_info.transAxes)
    ax_info.text(0.5, 0.35, f"BTC Dominance: {btc_dominance:.1f}%", fontsize=12,
                 color=BLUE, ha="center", va="top", transform=ax_info.transAxes)
    ax_info.text(0.5, 0.15, f"Sentiment Score: {sentiment_score:+.2f}", fontsize=12,
                 color=GREEN if sentiment_score > 0 else RED,
                 ha="center", va="top", transform=ax_info.transAxes)

    # --- Top Movers ---
    ax_movers = fig.add_subplot(gs[1, :])
    ax_movers.set_facecolor(CARD_BG)
    ax_movers.axis("off")
    ax_movers.set_title("Top Movers (24h)", fontsize=12, color=WHITE, fontweight="bold")

    if top_movers:
        for i, mover in enumerate(top_movers[:10]):
            x = 0.05 + (i % 5) * 0.19
            y = 0.6 if i < 5 else 0.15
            sym = mover.get("symbol", "?").replace("USDT", "")
            change = mover.get("change_pct", 0)
            color = GREEN if change >= 0 else RED
            ax_movers.text(x, y, f"{sym}", fontsize=10, color=WHITE,
                           transform=ax_movers.transAxes, fontweight="bold")
            ax_movers.text(x, y - 0.15, f"{change:+.1f}%", fontsize=10, color=color,
                           transform=ax_movers.transAxes)
    else:
        ax_movers.text(0.5, 0.5, "Loading market data...", fontsize=12, color=GRAY,
                       ha="center", va="center", transform=ax_movers.transAxes)

    return _fig_to_bytes(fig)


# ---------------------------------------------------------------------------
# 5. Weekly Report Chart
# ---------------------------------------------------------------------------

def generate_weekly_report(weekly_data: dict) -> bytes:
    """Comprehensive weekly report visual."""
    fig = plt.figure(figsize=(12, 8))
    fig.patch.set_facecolor(DARK_BG)
    gs = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

    total_pnl = weekly_data.get("total_pnl", 0)
    total_pnl_pct = weekly_data.get("total_pnl_pct", 0)
    wins = weekly_data.get("wins", 0)
    losses = weekly_data.get("losses", 0)
    total_trades = wins + losses
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    best_trade = weekly_data.get("best_trade", {})
    worst_trade = weekly_data.get("worst_trade", {})
    daily_pnls = weekly_data.get("daily_pnls", [])
    sector_pnl = weekly_data.get("sector_pnl", {})

    # --- Header ---
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.set_facecolor(DARK_BG)
    ax_header.axis("off")

    pnl_color = GREEN if total_pnl >= 0 else RED
    ax_header.text(0.02, 0.9, "WEEKLY PERFORMANCE REPORT", fontsize=18,
                   fontweight="bold", color=CYAN, transform=ax_header.transAxes, va="top")
    ax_header.text(0.02, 0.55, f"P&L: ${total_pnl:+,.2f} ({total_pnl_pct:+.2f}%)",
                   fontsize=22, fontweight="bold", color=pnl_color,
                   transform=ax_header.transAxes, va="top")

    stats_text = (
        f"Trades: {total_trades}  |  Wins: {wins}  |  Losses: {losses}  |  "
        f"Win Rate: {win_rate:.0f}%"
    )
    ax_header.text(0.02, 0.2, stats_text, fontsize=12, color=WHITE,
                   transform=ax_header.transAxes, va="top")

    if best_trade:
        ax_header.text(0.7, 0.55, f"Best: {best_trade.get('symbol', '?')} +{best_trade.get('pnl_pct', 0):.2f}%",
                       fontsize=11, color=GREEN, transform=ax_header.transAxes, va="top")
    if worst_trade:
        ax_header.text(0.7, 0.3, f"Worst: {worst_trade.get('symbol', '?')} {worst_trade.get('pnl_pct', 0):.2f}%",
                       fontsize=11, color=RED, transform=ax_header.transAxes, va="top")

    # --- Daily P&L Bars ---
    ax_daily = fig.add_subplot(gs[1, 0])
    _apply_dark_theme(fig, ax_daily)

    if daily_pnls:
        days = [d.get("day", f"D{i}") for i, d in enumerate(daily_pnls)]
        values = [d.get("pnl", 0) for d in daily_pnls]
        colors = [GREEN if v >= 0 else RED for v in values]
        ax_daily.bar(days, values, color=colors, alpha=0.85, edgecolor=GRID_COLOR)
        ax_daily.axhline(y=0, color=GRAY, linestyle="--", linewidth=0.8)
    ax_daily.set_title("Daily P&L", fontsize=12, fontweight="bold")
    ax_daily.grid(True, axis="y", color=GRID_COLOR, alpha=0.3)

    # --- Win/Loss Donut ---
    ax_wl = fig.add_subplot(gs[1, 1])
    ax_wl.set_facecolor(DARK_BG)

    if total_trades > 0:
        sizes = [wins, losses]
        colors_wl = [GREEN, RED]
        wedges, texts, autotexts = ax_wl.pie(
            sizes, labels=["Wins", "Losses"], autopct="%1.0f%%",
            colors=colors_wl, startangle=90, pctdistance=0.75,
            textprops={"color": WHITE, "fontsize": 10},
            wedgeprops=dict(width=0.4),
        )
        ax_wl.text(0, 0, f"{win_rate:.0f}%", fontsize=18, fontweight="bold",
                   color=WHITE, ha="center", va="center")
    else:
        ax_wl.text(0.5, 0.5, "No trades", fontsize=14, color=GRAY,
                   ha="center", va="center", transform=ax_wl.transAxes)
    ax_wl.set_title("Win Rate", fontsize=12, color=WHITE, fontweight="bold")

    # --- Sector Performance ---
    ax_sector = fig.add_subplot(gs[2, :])
    _apply_dark_theme(fig, ax_sector)

    if sector_pnl:
        sectors = list(sector_pnl.keys())[:10]
        values = [sector_pnl[s] for s in sectors]
        colors_s = [SECTOR_COLORS.get(s, BLUE) for s in sectors]
        ax_sector.barh(sectors, values, color=colors_s, alpha=0.85, edgecolor=GRID_COLOR)
        ax_sector.axvline(x=0, color=GRAY, linestyle="--", linewidth=0.8)
    ax_sector.set_title("Sector Performance ($)", fontsize=12, fontweight="bold")
    ax_sector.grid(True, axis="x", color=GRID_COLOR, alpha=0.3)

    return _fig_to_bytes(fig)


# ---------------------------------------------------------------------------
# 6. Trade Entry/Exit Card
# ---------------------------------------------------------------------------

def generate_trade_card(trade_data: dict, action: str = "ENTRY") -> bytes:
    """Visual card for trade entry or exit notification."""
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)
    ax.axis("off")

    symbol = trade_data.get("symbol", "UNKNOWN")
    price = trade_data.get("price", 0)
    is_entry = action == "ENTRY"

    # Header
    header_color = GREEN if is_entry else (GREEN if trade_data.get("pnl", 0) >= 0 else RED)
    header_text = f"{'BUY' if is_entry else 'SELL'} — {symbol}"
    ax.text(0.05, 0.9, header_text, fontsize=20, fontweight="bold",
            color=header_color, transform=ax.transAxes, va="top")

    # Price
    ax.text(0.05, 0.7, f"Price: ${price:,.6f}", fontsize=14, color=WHITE,
            transform=ax.transAxes, va="top")

    if is_entry:
        score = trade_data.get("confluence_score", 0)
        sl = trade_data.get("stop_loss", 0)
        tp = trade_data.get("take_profit", 0)
        size = trade_data.get("position_size_usd", 0)

        ax.text(0.05, 0.5, f"Score: {score}/100", fontsize=12, color=CYAN,
                transform=ax.transAxes, va="top")
        ax.text(0.05, 0.35, f"Size: ${size:,.2f}", fontsize=12, color=WHITE,
                transform=ax.transAxes, va="top")
        ax.text(0.55, 0.5, f"SL: ${sl:,.6f}", fontsize=12, color=RED,
                transform=ax.transAxes, va="top")
        ax.text(0.55, 0.35, f"TP: ${tp:,.6f}", fontsize=12, color=GREEN,
                transform=ax.transAxes, va="top")

        # Regime & AI
        regime = trade_data.get("regime", "?")
        ai_verdict = trade_data.get("ai_verdict", "N/A")
        ax.text(0.05, 0.15, f"Regime: {regime} | AI: {ai_verdict}", fontsize=10,
                color=GRAY, transform=ax.transAxes, va="top")
    else:
        pnl = trade_data.get("pnl", 0)
        pnl_pct = trade_data.get("pnl_pct", 0)
        hold_time = trade_data.get("hold_time", "?")
        exit_reason = trade_data.get("exit_reason", "?")

        pnl_color = GREEN if pnl >= 0 else RED
        ax.text(0.05, 0.5, f"P&L: ${pnl:+,.2f} ({pnl_pct:+.2f}%)", fontsize=14,
                fontweight="bold", color=pnl_color, transform=ax.transAxes, va="top")
        ax.text(0.05, 0.3, f"Hold Time: {hold_time}", fontsize=11, color=WHITE,
                transform=ax.transAxes, va="top")
        ax.text(0.05, 0.15, f"Exit: {exit_reason}", fontsize=11, color=ORANGE,
                transform=ax.transAxes, va="top")

    # Timestamp
    ax.text(0.95, 0.05, datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            fontsize=8, color=GRAY, ha="right", transform=ax.transAxes)

    return _fig_to_bytes(fig)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_sector(symbol: str) -> str:
    """Quick sector lookup for coloring."""
    sector_map = {
        "BTC": "Layer1", "ETH": "Layer1", "SOL": "Layer1", "ADA": "Layer1",
        "AVAX": "Layer1", "DOT": "Layer1", "ATOM": "Layer1", "NEAR": "Layer1",
        "MATIC": "Layer2", "ARB": "Layer2", "OP": "Layer2", "IMX": "Layer2",
        "UNI": "DeFi", "AAVE": "DeFi", "MKR": "DeFi", "CRV": "DeFi",
        "DOGE": "Meme", "SHIB": "Meme", "PEPE": "Meme", "FLOKI": "Meme",
        "FET": "AI", "RNDR": "AI", "AGIX": "AI", "TAO": "AI",
        "AXS": "Gaming", "SAND": "Gaming", "MANA": "Gaming", "GALA": "Gaming",
        "FIL": "Storage", "AR": "Storage",
        "LINK": "Oracle", "BAND": "Oracle",
        "XMR": "Privacy", "ZEC": "Privacy",
        "BNB": "Exchange", "CRO": "Exchange",
        "CASH": "Cash",
    }
    clean = symbol.replace("USDT", "").upper()
    return sector_map.get(clean, "Other")
