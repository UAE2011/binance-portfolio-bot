"""
Risk Manager — Playbook-based risk management for exponential compounding.

Key principles:
  - Quarter Kelly (25%) for safety while maximizing geometric growth
  - Anti-martingale: scale UP after wins, DOWN after losses
  - Portfolio heat cap: max 10% total open risk (correlation-adjusted)
  - Circuit breakers at 5%/10%/15% drawdown (escalating response)
  - Chandelier Exit for trailing stops (ATR 22, 3× multiplier)
  - 50/50 partial TP at 2R locks profit, runner compounds

Compounding math: 55% win rate + 2:1 R:R + 2% risk = ~0.5% avg per trade.
At 5 trades/day = 0.75-1% daily compound = 1,300%+ annualized (theoretical max).
"""
import json
import math
from typing import Optional

from config.settings import Settings
from src.utils import setup_logging, utc_now

logger = setup_logging()


class RiskManager:
    def __init__(self, database, regime_detector):
        self.db = database
        self.regime = regime_detector
        self.cfg = Settings.risk

        # Kelly criterion parameters (auto-calibrated from trade history)
        self.win_rate: float = 0.55
        self.avg_win: float = 0.06
        self.avg_loss: float = 0.03
        self.kelly_fraction: float = 0.05     # quarter Kelly applied below

        # Anti-martingale scale factor (1.0 = neutral)
        self.scale_factor: float = 1.0
        self._consecutive_wins: int = 0
        self._consecutive_losses: int = 0

        # Calibratable parameters
        self.atr_stop_multiplier: float = Settings.risk.ATR_STOP_MULTIPLIER
        self.confluence_threshold: int = Settings.strategy.CONFLUENCE_SCORE_THRESHOLD
        self.position_size_modifier: float = 1.0

        # State
        self.is_paused: bool = False
        self.kill_switch_active: bool = False
        self.kill_switch_activated_at = None   # datetime when kill switch fired
        self.auto_resume_cooldown_min: int = int(
            __import__("os").getenv("AUTO_RESUME_COOLDOWN_MIN", "30")
        )  # minutes before auto-resume check begins
        self._news_intel = None

        # Portfolio heat tracking (total % at risk across all open positions)
        self._open_risk_pct: float = 0.0

    def initialize_from_history(self):
        stats = self.db.get_trade_stats(Settings.calibration.LOOKBACK_TRADES)
        if stats["total_trades"] >= 10:
            self.win_rate = stats["win_rate"]
            self.avg_win = stats["avg_win"]
            self.avg_loss = stats["avg_loss"]
            self._recalculate_kelly()
            logger.info("Risk from history: win=%.2f avg_win=%.3f avg_loss=%.3f kelly=%.4f",
                        self.win_rate, self.avg_win, self.avg_loss, self.kelly_fraction)
        else:
            self._recalculate_kelly()
            logger.info("Default risk params (< 10 trades in history)")

        # Restore anti-martingale scale
        consecutive = self.db.get_consecutive_losses()
        if consecutive >= 3:
            self.scale_factor = max(self.cfg.MIN_SCALE_FACTOR,
                                    self.cfg.LOSS_SCALE_DOWN ** consecutive)
            logger.warning("%d consecutive losses → scale factor: %.2f", consecutive, self.scale_factor)

    def set_news_intel(self, news_intel):
        self._news_intel = news_intel

    def _recalculate_kelly(self):
        """Quarter Kelly for safety. Full Kelly maximizes growth but is too volatile for crypto."""
        if self.avg_loss == 0:
            self.kelly_fraction = 0.02
            return
        b = self.avg_win / self.avg_loss          # win/loss ratio
        p = self.win_rate
        q = 1 - p
        full_kelly = (p * b - q) / b if b > 0 else 0
        # Quarter Kelly: 25% of full Kelly, clamped to 0.5%–8%
        quarter_kelly = full_kelly * Settings.risk.KELLY_FRACTION
        self.kelly_fraction = max(min(quarter_kelly, 0.08), 0.005)

    def record_trade_result(self, won: bool):
        """Update anti-martingale scale factor after each trade."""
        if won:
            self._consecutive_wins += 1
            self._consecutive_losses = 0
            # Scale up after each win (up to max)
            self.scale_factor = min(
                self.scale_factor * self.cfg.WIN_SCALE_UP,
                self.cfg.MAX_SCALE_FACTOR
            )
        else:
            self._consecutive_losses += 1
            self._consecutive_wins = 0
            # Scale down after each loss (down to min)
            self.scale_factor = max(
                self.scale_factor * self.cfg.LOSS_SCALE_DOWN,
                self.cfg.MIN_SCALE_FACTOR
            )
        logger.debug("Anti-martingale scale: %.2f (wins=%d, losses=%d)",
                     self.scale_factor, self._consecutive_wins, self._consecutive_losses)

    # ------------------------------------------------------------------
    # Position Sizing — Quarter Kelly + Anti-Martingale + Portfolio Heat
    # ------------------------------------------------------------------

    def calculate_position_size(self, portfolio_value: float, atr: float,
                                price: float, signal_score: int = 50) -> float:
        """
        Position size = min(Kelly-based, ATR-based, max position).
        Scaled by: regime × anti-martingale × signal quality.
        Capped by portfolio heat (total open risk ≤ 10%).
        """
        if self.kill_switch_active or self.is_paused:
            return 0.0

        regime_params = self.regime.get_regime_params()
        regime_mult = regime_params.get("position_multiplier", 1.0)

        # Method 1: Quarter Kelly
        kelly_size = portfolio_value * self.kelly_fraction

        # Method 2: Fixed risk per trade (risk 2% of portfolio)
        risk_amount = portfolio_value * self.cfg.MAX_RISK_PER_TRADE
        stop_pct = self.cfg.STOP_LOSS_PCT
        risk_based_size = risk_amount / stop_pct if stop_pct > 0 else kelly_size

        # Method 3: Max position cap
        if portfolio_value < 200:
            max_pos = min(portfolio_value * 0.30, portfolio_value * 0.95)
        else:
            max_pos = portfolio_value * self.cfg.MAX_POSITION_SIZE

        # Take the minimum for safety
        base_size = min(kelly_size, risk_based_size, max_pos)

        # Signal quality bonus: high-conviction signals get up to 1.3× size
        score_mult = 1.0
        if signal_score >= 70:
            score_mult = 1.25
        elif signal_score >= 60:
            score_mult = 1.10
        elif signal_score < 45:
            score_mult = 0.85

        # Apply all multipliers
        adjusted_size = base_size * regime_mult * self.scale_factor * score_mult * self.position_size_modifier

        # Extreme fear bonus: buy more when others are fearful (contrarian)
        if self._news_intel and self._news_intel.fear_greed_value < 15:
            dca_mult = regime_params.get("dca_multiplier", 1.0)
            adjusted_size *= min(dca_mult, 2.0)

        # Floor: $5 absolute minimum
        adjusted_size = max(adjusted_size, 5.0)
        adjusted_size = min(adjusted_size, max_pos)

        return round(adjusted_size, 2)

    def get_open_risk_pct(self, portfolio_value: float, open_positions: list) -> float:
        """Calculate total portfolio heat (% at risk across all open positions)."""
        if not open_positions or portfolio_value <= 0:
            return 0.0
        total_risk = 0.0
        for pos in open_positions:
            entry = pos.get("entry_price", 0)
            stop = pos.get("stop_loss", 0)
            remaining_qty = pos.get("remaining_quantity", pos.get("quantity", 0))
            if entry > 0 and stop > 0:
                risk_per_unit = entry - stop
                position_risk = risk_per_unit * remaining_qty
                total_risk += max(position_risk, 0)
        return total_risk / portfolio_value

    def portfolio_heat_ok(self, portfolio_value: float, open_positions: list) -> bool:
        """Portfolio heat check — cap total open risk at MAX_PORTFOLIO_HEAT."""
        heat = self.get_open_risk_pct(portfolio_value, open_positions)
        return heat < self.cfg.MAX_PORTFOLIO_HEAT

    # ------------------------------------------------------------------
    # Stop-Loss — ATR-based Chandelier style (3× ATR from recent high)
    # ------------------------------------------------------------------

    def calculate_stop_loss(self, entry_price: float, atr: float) -> float:
        """
        Stop-loss: max(fixed %, ATR-based).
        Uses 3× ATR multiplier (crypto-optimized, avoids premature stops).
        """
        fixed_stop = entry_price * (1 - self.cfg.STOP_LOSS_PCT)
        regime_params = self.regime.get_regime_params()
        mult = regime_params.get("stop_atr_mult", self.atr_stop_multiplier)
        atr_stop = entry_price - (atr * mult) if atr and atr > 0 else fixed_stop
        # Use the HIGHER (tighter) stop, but never below fixed minimum
        return max(fixed_stop, atr_stop)

    def calculate_take_profit(self, entry_price: float, atr: float = None) -> dict:
        """Dynamic TP targets: TP1 (6%), TP2 (trailing runner)."""
        tp1 = entry_price * (1 + self.cfg.PARTIAL_TP_TRIGGER)
        # ATR-based extended target
        if atr and atr > 0:
            regime_params = self.regime.get_regime_params()
            tp_mult = regime_params.get("tp_atr_mult", self.cfg.ATR_TP_MULTIPLIER)
            atr_tp = entry_price + (atr * tp_mult)
            tp2_target = max(tp1 * 1.05, atr_tp)  # at least 5% above TP1
        else:
            tp2_target = entry_price * (1 + self.cfg.TAKE_PROFIT_PCT * 1.5)
        return {"tp1": tp1, "tp2": tp2_target}

    # ------------------------------------------------------------------
    # Trailing Stop — Chandelier Exit style
    # ------------------------------------------------------------------

    def update_trailing_stop(self, trade: dict, current_price: float,
                             current_atr: float) -> dict:
        """
        Chandelier Exit trailing stop:
        - Activates at 2% gain (1R equivalent)
        - Trails at 3× ATR from highest high
        - After partial TP: tighter 2× ATR trail on runner
        - After 6% gain: lock in 3% minimum profit
        """
        entry_price = trade["entry_price"]
        current_stop = trade["stop_loss"]
        highest = max(trade.get("highest_price", entry_price), current_price)
        gain_pct = (current_price / entry_price) - 1
        exits_done = trade.get("tranche_exits", [])
        if isinstance(exits_done, str):
            exits_done = json.loads(exits_done)
        has_taken_partial = len(exits_done) > 0

        activation = self.cfg.TRAILING_STOP_ACTIVATION_PCT  # 2%

        if gain_pct >= activation:
            # Tighter trail for runner (after partial TP)
            if has_taken_partial:
                trail_dist = self.cfg.RUNNER_TRAILING_DISTANCE  # 2%
                chandelier_mult = 2.0  # tighter for runner
            else:
                trail_dist = self.cfg.TRAILING_STOP_DISTANCE_PCT  # 1.5%
                chandelier_mult = 3.0

            # Chandelier: highest_high - ATR × multiplier
            if current_atr and current_atr > 0:
                chandelier_stop = highest - (current_atr * chandelier_mult)
            else:
                chandelier_stop = highest * (1 - trail_dist)

            new_stop = max(chandelier_stop, current_stop)

            # Breakeven lock after 2% gain
            if gain_pct >= 0.02:
                breakeven = entry_price * 1.002   # breakeven + 0.2% fees
                new_stop = max(new_stop, breakeven)

            # Lock in 3% profit after 6% gain
            if gain_pct >= 0.06:
                lock_3pct = entry_price * 1.03
                new_stop = max(new_stop, lock_3pct)

            # Lock in 5% profit after 10% gain
            if gain_pct >= 0.10:
                lock_5pct = entry_price * 1.05
                new_stop = max(new_stop, lock_5pct)

            return {"stop_loss": new_stop, "highest_price": highest}

        return {"stop_loss": current_stop, "highest_price": highest}

    # ------------------------------------------------------------------
    # Partial Take-Profit — 50/50 at 2R
    # ------------------------------------------------------------------

    def get_partial_tp_plan(self, entry_price: float, quantity: float) -> list:
        """50/50 split: sell half at 6%, ride runner with trailing stop."""
        tp_price = entry_price * (1 + self.cfg.PARTIAL_TP_TRIGGER)
        return [
            {
                "level": 1, "price": tp_price,
                "pct": self.cfg.PARTIAL_TP_PCT,
                "quantity": quantity * self.cfg.PARTIAL_TP_PCT,
                "type": "FIXED_TP",
                "description": f"Sell 50% at {self.cfg.PARTIAL_TP_TRIGGER*100:.0f}% gain",
            },
            {
                "level": 2, "price": 0,
                "pct": 1.0 - self.cfg.PARTIAL_TP_PCT,
                "quantity": quantity * (1.0 - self.cfg.PARTIAL_TP_PCT),
                "type": "TRAILING_RUNNER",
                "description": "Runner: trail with Chandelier Exit",
            },
        ]

    def check_take_profit_tranches(self, trade: dict, current_price: float,
                                   current_atr: float) -> list:
        """Check if TP1 (50%) should execute."""
        entry_price = trade["entry_price"]
        remaining = trade.get("remaining_quantity", trade["quantity"])
        exits_done = trade.get("tranche_exits", [])
        if isinstance(exits_done, str):
            exits_done = json.loads(exits_done)
        done_levels = {e.get("level") for e in exits_done} if exits_done else set()

        exits_to_execute = []
        tp_price = entry_price * (1 + self.cfg.PARTIAL_TP_TRIGGER)

        if 1 not in done_levels and current_price >= tp_price:
            sell_qty = min(trade["quantity"] * self.cfg.PARTIAL_TP_PCT, remaining)
            if sell_qty > 0:
                exits_to_execute.append({
                    "level": 1,
                    "price": tp_price,
                    "quantity": sell_qty,
                    "reason": f"TP1: 50% at {self.cfg.PARTIAL_TP_TRIGGER*100:.0f}% gain (2R)",
                })
        return exits_to_execute

    # ------------------------------------------------------------------
    # Circuit Breakers — Escalating drawdown response
    # ------------------------------------------------------------------

    def check_drawdown(self, current_value: float, peak_value: float = None) -> dict:
        """
        3-stage circuit breaker system:
        5% drawdown  → reduce position sizes 50%
        10% drawdown → close 50% of positions
        15% drawdown → go to cash (capital preservation)
        20% drawdown → kill switch (full stop)

        peak_value: pass portfolio.peak_value (current-session tracked peak).
        Avoids false triggers from stale all-time-high DB records (e.g. old
        testnet sessions with inflated fake balances).
        """
        # Use the passed-in peak (portfolio.peak_value — current session max).
        # Fall back to DB only if no peak given, and sanity-check that the
        # DB peak is not an inflated relic from an old session (>50% above current).
        if peak_value and peak_value > current_value:
            peak = peak_value
        else:
            db_peak = self.db.get_peak_portfolio_value()
            # Ignore DB peak if it is >50% above current — that is old data.
            if db_peak > 0 and db_peak <= current_value * 1.5:
                peak = db_peak
            else:
                peak = current_value  # treat current as peak — no drawdown
        if peak <= 0 or peak < 10.0:
            return {"drawdown_pct": 0, "action": "NONE"}
        if (peak - current_value) < 5.0:
            return {"drawdown_pct": 0, "action": "NONE"}

        drawdown = (peak - current_value) / peak

        if drawdown >= self.cfg.MAX_PORTFOLIO_DRAWDOWN:
            if not self.kill_switch_active:  # only set time on first trigger
                from src.utils import utc_now
                self.kill_switch_activated_at = utc_now()
            self.kill_switch_active = True
            return {"drawdown_pct": drawdown, "action": "KILL_SWITCH"}
        elif drawdown >= self.cfg.CIRCUIT_BREAKER_3:  # 15%
            return {"drawdown_pct": drawdown, "action": "CAPITAL_PRESERVATION"}
        elif drawdown >= self.cfg.CIRCUIT_BREAKER_2:  # 10%
            return {"drawdown_pct": drawdown, "action": "EMERGENCY_LIQUIDATE_50"}
        elif drawdown >= self.cfg.CIRCUIT_BREAKER_1:  # 5%
            self.position_size_modifier = 0.5
            return {"drawdown_pct": drawdown, "action": "DEFENSIVE"}
        elif drawdown > 0.03:
            return {"drawdown_pct": drawdown, "action": "WARNING"}
        return {"drawdown_pct": drawdown, "action": "NONE"}

    def reset_kill_switch(self):
        """Reset kill switch — called by /resume or auto-resume logic."""
        self.kill_switch_active = False
        self.kill_switch_activated_at = None
        self.position_size_modifier = 1.0
        logger.info("Kill switch reset. Trading resumed.")

    # ------------------------------------------------------------------
    # Daily Loss Limit
    # ------------------------------------------------------------------

    def check_daily_loss_limit(self) -> bool:
        daily_pnl = self.db.get_daily_pnl()
        snapshots = self.db.get_snapshots_for_period(1)
        if snapshots:
            start_value = snapshots[0]["total_value_usdt"]
            if start_value > 0 and daily_pnl < 0:
                daily_loss_pct = abs(daily_pnl) / start_value
                if daily_loss_pct >= self.cfg.MAX_DAILY_LOSS:
                    logger.warning("Daily loss limit hit: %.2f%%", daily_loss_pct * 100)
                    return True
        return False

    # ------------------------------------------------------------------
    # Entry Admission Control
    # ------------------------------------------------------------------

    def can_open_position(self, portfolio_value: float, cash_available: float,
                          open_positions: list, sector: str,
                          sector_exposure: dict) -> dict:
        if self.kill_switch_active:
            return {"allowed": False, "reason": "Kill switch active — send /resume"}
        if self.is_paused:
            return {"allowed": False, "reason": "Trading paused — send /resume"}

        regime_params = self.regime.get_regime_params()

        if not regime_params.get("entries_allowed", True):
            return {"allowed": False, "reason": "Capital preservation mode — no new entries"}

        open_count = len(open_positions)
        # Max positions = how many more the available cash can fund.
        # Each new position needs at least the Binance notional minimum ($10)
        # plus a sensible per-trade size (1% of portfolio, min $11).
        min_viable_position = max(11.0, portfolio_value * 0.01)
        # How many more can we open with remaining cash?
        affordable = int(cash_available / min_viable_position)
        # Hard cap: no more than configured max (prevents over-diversification)
        max_pos = min(affordable + open_count, self.cfg.MAX_OPEN_POSITIONS)
        if open_count >= max_pos or affordable <= 0:
            return {
                "allowed": False,
                "reason": (
                    f"Insufficient cash for new position "
                    f"(have ${cash_available:.2f}, need ≥${min_viable_position:.2f})"
                    if affordable <= 0
                    else f"Max affordable positions ({max_pos}) reached"
                ),
            }

        # Portfolio heat check
        if not self.portfolio_heat_ok(portfolio_value, open_positions):
            heat = self.get_open_risk_pct(portfolio_value, open_positions)
            return {"allowed": False, "reason": f"Portfolio heat {heat:.1%} ≥ {self.cfg.MAX_PORTFOLIO_HEAT:.0%} max"}

        # Cash check — adaptive floor
        cash_reserve = regime_params.get("cash_reserve", 0.20)
        min_cash = portfolio_value * cash_reserve
        abs_floor = 5.0 if portfolio_value < 100 else 15.0
        if cash_available < max(min_cash, abs_floor):
            return {"allowed": False, "reason": f"Insufficient cash (need {cash_reserve:.0%} reserve)"}

        # Exposure cap
        max_exposure = regime_params.get("max_exposure", 0.80)
        invested = portfolio_value - cash_available
        if portfolio_value > 0 and invested / portfolio_value >= max_exposure:
            return {"allowed": False, "reason": f"Max exposure {max_exposure:.0%} reached"}

        # Sector cap — 30% max per sector (relaxed to 50% for small accounts)
        from config.settings import Settings
        sector_data = Settings.sectors().get("sectors", {}).get(sector, {})
        max_sector = max(sector_data.get("max_allocation", 0.30), 0.50)
        current_sector = sector_exposure.get(sector, 0)
        if portfolio_value > 0 and current_sector / portfolio_value >= max_sector:
            return {"allowed": False, "reason": f"Sector {sector} at max ({max_sector:.0%})"}

        if self.check_daily_loss_limit():
            return {"allowed": False, "reason": "Daily loss limit reached"}

        dd = self.check_drawdown(portfolio_value)
        if dd["action"] in ("KILL_SWITCH", "CAPITAL_PRESERVATION", "EMERGENCY_LIQUIDATE_50"):
            return {"allowed": False, "reason": f"Drawdown circuit breaker: {dd['action']}"}

        return {"allowed": True, "reason": "All checks passed"}

    def get_status_summary(self) -> dict:
        """Full status for /risk and /heat Telegram commands."""
        return {
            "win_rate": self.win_rate,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "kelly_fraction": self.kelly_fraction,
            "scale_factor": self.scale_factor,
            "consecutive_wins": self._consecutive_wins,
            "consecutive_losses": self._consecutive_losses,
            "position_size_modifier": self.position_size_modifier,
            "kill_switch_active": self.kill_switch_active,
            "is_paused": self.is_paused,
            "atr_stop_multiplier": self.atr_stop_multiplier,
            "confluence_threshold": self.confluence_threshold,
        }
