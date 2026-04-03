"""
Risk Manager — Implements the SOL bot's proven risk management:
  - 10% wallet rule (max position size)
  - 3% fixed stop-loss
  - 6% fixed take-profit
  - Trailing stop (activates at 3% gain, trails by 1.5%)
  - 50/50 partial take-profit (sell 50% at 6%, ride runner with trailing stop)
  - Daily loss limit (3%)
  - Drawdown circuit breakers (5%/10%/15%/20%)
  - Kelly criterion position sizing (auto-calibrated)
  - Consecutive loss detection (halves size after 3 losses)
"""

import json
from typing import Optional

from config.settings import Settings
from src.utils import setup_logging, utc_now

logger = setup_logging()


class RiskManager:
    def __init__(self, database, regime_detector):
        self.db = database
        self.regime = regime_detector
        self.cfg = Settings.risk

        # Kelly criterion parameters (auto-calibrated)
        self.win_rate: float = 0.55
        self.avg_win: float = 0.06
        self.avg_loss: float = 0.03
        self.kelly_fraction: float = 0.05

        # Calibratable parameters
        self.atr_stop_multiplier: float = Settings.strategy.ATR_STOP_MULTIPLIER
        self.confluence_threshold: int = Settings.strategy.CONFLUENCE_SCORE_THRESHOLD
        self.position_size_modifier: float = 1.0

        # State
        self.is_paused: bool = False
        self.kill_switch_active: bool = False
        self._news_intel = None

    def initialize_from_history(self):
        """Load risk parameters from trade history."""
        stats = self.db.get_trade_stats(Settings.calibration.LOOKBACK_TRADES)
        if stats["total_trades"] >= 20:
            self.win_rate = stats["win_rate"]
            self.avg_win = stats["avg_win"]
            self.avg_loss = stats["avg_loss"]
            self._recalculate_kelly()
            logger.info(
                "Risk params from history: win_rate=%.2f, avg_win=%.3f, avg_loss=%.3f, kelly=%.4f",
                self.win_rate, self.avg_win, self.avg_loss, self.kelly_fraction,
            )
        else:
            self._recalculate_kelly()
            logger.info("Using default risk params (insufficient trade history)")

        consecutive = self.db.get_consecutive_losses()
        if consecutive >= 3:
            self.position_size_modifier = 0.5
            logger.warning("Detected %d consecutive losses, reducing position sizes by 50%%", consecutive)

    def set_news_intel(self, news_intel):
        self._news_intel = news_intel

    def _recalculate_kelly(self):
        if self.avg_loss == 0:
            self.kelly_fraction = 0.02
            return
        b = self.avg_win / self.avg_loss
        p = self.win_rate
        q = 1 - p
        kelly = (p * b - q) / b if b > 0 else 0
        # Quarter-Kelly for safety, clamped to 0.5%–10%
        self.kelly_fraction = max(min(kelly * 0.25, 0.10), 0.005)

    # ------------------------------------------------------------------
    # Position Sizing — 10% wallet rule
    # ------------------------------------------------------------------

    def calculate_position_size(self, portfolio_value: float, atr: float,
                                price: float) -> float:
        """Calculate position size respecting the 10% wallet rule."""
        if self.kill_switch_active or self.is_paused:
            return 0.0

        regime_params = self.regime.get_regime_params()
        regime_mult = regime_params.get("position_multiplier", 1.0)

        # Method 1: Kelly criterion
        kelly_size = portfolio_value * self.kelly_fraction

        # Method 2: Fixed risk per trade (risk 2% of portfolio)
        risk_per_trade = portfolio_value * self.cfg.MAX_RISK_PER_TRADE
        stop_pct = self.cfg.STOP_LOSS_PCT
        if stop_pct > 0:
            risk_based_size = risk_per_trade / stop_pct
        else:
            risk_based_size = kelly_size

        # Method 3: 10% wallet rule (hard cap)
        max_pos = portfolio_value * self.cfg.MAX_POSITION_SIZE

        # Take the minimum of all methods
        base_size = min(kelly_size, risk_based_size, max_pos)

        # Apply regime and performance modifiers
        adjusted_size = base_size * regime_mult * self.position_size_modifier

        # Consecutive loss reduction
        if self.db.get_consecutive_losses() >= 3:
            adjusted_size *= 0.5

        # Extreme fear bonus (buy more when others are fearful)
        if self._news_intel and self._news_intel.is_extreme_fear():
            adjusted_size *= 1.25

        # Floor and cap
        adjusted_size = max(adjusted_size, 20.0)
        adjusted_size = min(adjusted_size, max_pos)

        return round(adjusted_size, 2)

    # ------------------------------------------------------------------
    # Stop-Loss — Fixed 3% with ATR validation
    # ------------------------------------------------------------------

    def calculate_stop_loss(self, entry_price: float, atr: float) -> float:
        """Calculate stop-loss: fixed 3% or ATR-based, whichever is tighter."""
        # Fixed 3% stop
        fixed_stop = entry_price * (1 - self.cfg.STOP_LOSS_PCT)

        # ATR-based stop (regime-adjusted)
        regime_params = self.regime.get_regime_params()
        mult = regime_params.get("stop_atr_mult", self.atr_stop_multiplier)
        atr_stop = entry_price - (atr * mult)

        # Use the higher (tighter) stop
        return max(fixed_stop, atr_stop)

    # ------------------------------------------------------------------
    # Take-Profit — Fixed 6% with 50/50 partial exit
    # ------------------------------------------------------------------

    def calculate_take_profit(self, entry_price: float) -> float:
        """Fixed 6% take-profit target."""
        return entry_price * (1 + self.cfg.TAKE_PROFIT_PCT)

    def get_partial_tp_plan(self, entry_price: float, quantity: float) -> list:
        """50/50 partial take-profit plan:
        - Tranche 1: Sell 50% at 6% gain
        - Tranche 2: Ride remaining 50% with trailing stop
        """
        tp_price = entry_price * (1 + self.cfg.PARTIAL_TP_TRIGGER)
        return [
            {
                "level": 1,
                "price": tp_price,
                "pct": self.cfg.PARTIAL_TP_PCT,
                "quantity": quantity * self.cfg.PARTIAL_TP_PCT,
                "type": "FIXED_TP",
                "description": f"Sell 50% at {self.cfg.PARTIAL_TP_TRIGGER*100:.0f}% gain",
            },
            {
                "level": 2,
                "price": 0,  # trailing stop, no fixed price
                "pct": 1.0 - self.cfg.PARTIAL_TP_PCT,
                "quantity": quantity * (1.0 - self.cfg.PARTIAL_TP_PCT),
                "type": "TRAILING_RUNNER",
                "description": f"Ride remaining {(1-self.cfg.PARTIAL_TP_PCT)*100:.0f}% with trailing stop",
            },
        ]

    # ------------------------------------------------------------------
    # Trailing Stop — Activates at 3% gain, trails by 1.5%
    # ------------------------------------------------------------------

    def update_trailing_stop(self, trade: dict, current_price: float,
                             current_atr: float) -> dict:
        """Update trailing stop for a position."""
        entry_price = trade["entry_price"]
        current_stop = trade["stop_loss"]
        highest = max(trade.get("highest_price", entry_price), current_price)

        gain_pct = (current_price / entry_price) - 1
        exits_done = trade.get("tranche_exits", [])
        if isinstance(exits_done, str):
            exits_done = json.loads(exits_done)
        has_taken_partial = len(exits_done) > 0

        # Trailing stop activation
        activation_pct = self.cfg.TRAILING_STOP_ACTIVATION_PCT
        trail_distance = self.cfg.TRAILING_STOP_DISTANCE_PCT

        if gain_pct >= activation_pct:
            # For the runner (after partial TP), use tighter trail
            if has_taken_partial:
                trail_distance = self.cfg.RUNNER_TRAILING_DISTANCE

            trailing_stop = highest * (1 - trail_distance)

            # Never lower the stop
            new_stop = max(trailing_stop, current_stop)

            # After 3% gain, stop must be at least at breakeven
            if gain_pct >= 0.03:
                breakeven_plus = entry_price * 1.002  # breakeven + 0.2% for fees
                new_stop = max(new_stop, breakeven_plus)

            # After 6% gain, lock in at least 3% profit
            if gain_pct >= 0.06:
                lock_profit = entry_price * 1.03
                new_stop = max(new_stop, lock_profit)

            return {
                "stop_loss": new_stop,
                "highest_price": highest,
            }

        return {
            "stop_loss": current_stop,
            "highest_price": highest,
        }

    # ------------------------------------------------------------------
    # Partial Take-Profit Check — 50/50 split
    # ------------------------------------------------------------------

    def check_take_profit_tranches(self, trade: dict, current_price: float,
                                   current_atr: float) -> list:
        """Check if any take-profit tranches should execute (50/50 system)."""
        entry_price = trade["entry_price"]
        remaining = trade.get("remaining_quantity", trade["quantity"])
        exits_done = trade.get("tranche_exits", [])
        if isinstance(exits_done, str):
            exits_done = json.loads(exits_done)

        exits_to_execute = []
        done_levels = {e.get("level") for e in exits_done} if exits_done else set()

        # Tranche 1: Sell 50% at 6% gain
        tp_trigger = self.cfg.PARTIAL_TP_TRIGGER
        tp_price = entry_price * (1 + tp_trigger)

        if 1 not in done_levels and current_price >= tp_price:
            sell_qty = trade["quantity"] * self.cfg.PARTIAL_TP_PCT
            sell_qty = min(sell_qty, remaining)
            if sell_qty > 0:
                exits_to_execute.append({
                    "level": 1,
                    "price": tp_price,
                    "quantity": sell_qty,
                    "reason": f"TP1: Sell {self.cfg.PARTIAL_TP_PCT*100:.0f}% at {tp_trigger*100:.0f}% gain",
                })

        # Tranche 2 (runner) is managed by trailing stop, not a fixed level
        # It exits when trailing stop is hit, handled in _monitor_positions

        return exits_to_execute

    # ------------------------------------------------------------------
    # Drawdown Circuit Breakers
    # ------------------------------------------------------------------

    def check_drawdown(self, current_value: float) -> dict:
        peak = self.db.get_peak_portfolio_value()
        if peak <= 0:
            return {"drawdown_pct": 0, "action": "NONE"}

        drawdown = (peak - current_value) / peak

        if drawdown > self.cfg.MAX_PORTFOLIO_DRAWDOWN:
            self.kill_switch_active = True
            return {"drawdown_pct": drawdown, "action": "KILL_SWITCH"}
        elif drawdown > 0.15:
            return {"drawdown_pct": drawdown, "action": "EMERGENCY_LIQUIDATE_50"}
        elif drawdown > 0.10:
            return {"drawdown_pct": drawdown, "action": "DEFENSIVE"}
        elif drawdown > 0.05:
            return {"drawdown_pct": drawdown, "action": "WARNING"}
        return {"drawdown_pct": drawdown, "action": "NONE"}

    # ------------------------------------------------------------------
    # Daily Loss Limit — 3%
    # ------------------------------------------------------------------

    def check_daily_loss_limit(self) -> bool:
        daily_pnl = self.db.get_daily_pnl()
        snapshots = self.db.get_snapshots_for_period(1)
        if snapshots:
            start_value = snapshots[0]["total_value_usdt"]
            if start_value > 0:
                daily_loss_pct = abs(daily_pnl) / start_value
                if daily_pnl < 0 and daily_loss_pct >= self.cfg.MAX_DAILY_LOSS:
                    logger.warning("Daily loss limit hit: %.2f%%", daily_loss_pct * 100)
                    return True
        return False

    # ------------------------------------------------------------------
    # Entry Admission Control
    # ------------------------------------------------------------------

    def can_open_position(self, portfolio_value: float, cash_available: float,
                          open_positions: int, sector: str,
                          sector_exposure: dict) -> dict:
        if self.kill_switch_active:
            return {"allowed": False, "reason": "Kill switch active"}
        if self.is_paused:
            return {"allowed": False, "reason": "Trading paused"}

        regime_params = self.regime.get_regime_params()
        # Regime never blocks entries — only adjusts position sizing

        if open_positions >= self.cfg.MAX_OPEN_POSITIONS:
            return {"allowed": False, "reason": f"Max positions ({self.cfg.MAX_OPEN_POSITIONS}) reached"}

        min_cash = portfolio_value * self.cfg.MIN_CASH_RESERVE
        if cash_available <= min_cash:
            # In aggressive mode, only block if truly out of cash
            if cash_available < 15:
                return {"allowed": False, "reason": "Insufficient cash (< $15)"}

        max_exposure = regime_params.get("max_exposure", 0.80)
        invested = portfolio_value - cash_available
        if portfolio_value > 0 and invested / portfolio_value >= max_exposure:
            return {"allowed": False, "reason": f"Max exposure ({max_exposure*100:.0f}%) reached"}

        # Sector allocation check — relaxed for aggressive mode (50% cap)
        sector_data = Settings.sectors().get("sectors", {}).get(sector, {})
        max_sector = max(sector_data.get("max_allocation", 0.30), 0.50)
        current_sector = sector_exposure.get(sector, 0)
        if portfolio_value > 0 and current_sector / portfolio_value >= max_sector:
            return {"allowed": False, "reason": f"Sector {sector} at max allocation ({max_sector*100:.0f}%)"}

        if self.check_daily_loss_limit():
            return {"allowed": False, "reason": "Daily loss limit reached"}

        dd = self.check_drawdown(portfolio_value)
        if dd["action"] in ("KILL_SWITCH", "DEFENSIVE", "EMERGENCY_LIQUIDATE_50"):
            return {"allowed": False, "reason": f"Drawdown control: {dd['action']}"}
        if dd["action"] == "WARNING":
            self.position_size_modifier = 0.5

        return {"allowed": True, "reason": "All checks passed"}
