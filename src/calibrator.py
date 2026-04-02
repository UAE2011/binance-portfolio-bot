import asyncio
from typing import Optional
from datetime import datetime, timezone

from config.settings import Settings
from src.utils import setup_logging, utc_now

logger = setup_logging()


class SelfCalibrator:
    def __init__(self, database, risk_manager, regime_detector):
        self.db = database
        self.risk = risk_manager
        self.regime = regime_detector
        self.calibration_interval = Settings.calibration.CALIBRATION_INTERVAL_TRADES
        self.lookback = Settings.calibration.LOOKBACK_TRADES
        self.last_calibration_count: int = 0
        self.cold_start_complete: bool = False

    def initialize(self):
        total = self.db.get_total_trade_count()
        self.last_calibration_count = total
        if total >= 20:
            self.cold_start_complete = True
            self._run_calibration()
        else:
            logger.info(
                "Cold start mode: %d/%d trades before first calibration",
                total, 20,
            )

    def check_and_calibrate(self) -> bool:
        total = self.db.get_total_trade_count()
        trades_since = total - self.last_calibration_count

        if not self.cold_start_complete:
            if total >= 20:
                self.cold_start_complete = True
                logger.info("Cold start complete at %d trades, running first calibration", total)
                self._run_calibration()
                self.last_calibration_count = total
                return True
            return False

        if trades_since >= self.calibration_interval:
            self._run_calibration()
            self.last_calibration_count = total
            return True
        return False

    def _run_calibration(self):
        logger.info("Running self-calibration...")
        stats = self.db.get_trade_stats(self.lookback)

        if stats["total_trades"] < 10:
            logger.info("Insufficient trades for calibration (%d)", stats["total_trades"])
            return

        self._calibrate_kelly(stats)
        self._calibrate_stop_loss(stats)
        self._calibrate_confluence_threshold(stats)
        self._calibrate_position_modifier(stats)

        logger.info(
            "Calibration complete: kelly=%.4f, atr_mult=%.2f, threshold=%d, modifier=%.2f",
            self.risk.kelly_fraction, self.risk.atr_stop_multiplier,
            self.risk.confluence_threshold, self.risk.position_size_modifier,
        )

    def _calibrate_kelly(self, stats: dict):
        old_kelly = self.risk.kelly_fraction
        self.risk.win_rate = stats["win_rate"]
        self.risk.avg_win = stats["avg_win"]
        self.risk.avg_loss = stats["avg_loss"]
        self.risk._recalculate_kelly()
        new_kelly = self.risk.kelly_fraction

        if abs(new_kelly - old_kelly) > 0.001:
            self.db.save_calibration({
                "timestamp": utc_now(),
                "parameter_name": "kelly_fraction",
                "old_value": old_kelly,
                "new_value": new_kelly,
                "reason": f"win_rate={stats['win_rate']:.3f}, avg_win={stats['avg_win']:.4f}, avg_loss={stats['avg_loss']:.4f}",
                "trade_count_at_calibration": stats["total_trades"],
            })
            logger.info("Kelly fraction: %.4f -> %.4f", old_kelly, new_kelly)

    def _calibrate_stop_loss(self, stats: dict):
        old_mult = self.risk.atr_stop_multiplier
        stop_rate = stats["stop_loss_hit_rate"]

        if stop_rate > 0.5:
            new_mult = min(old_mult + 0.25, 4.0)
            reason = f"Stop hit rate too high ({stop_rate:.2f}), widening stops"
        elif stop_rate < 0.15 and stats["avg_rr"] < 1.2:
            new_mult = max(old_mult - 0.25, 1.0)
            reason = f"Stop hit rate low ({stop_rate:.2f}) but R:R poor ({stats['avg_rr']:.2f}), tightening"
        else:
            new_mult = old_mult
            reason = "No adjustment needed"

        if new_mult != old_mult:
            self.risk.atr_stop_multiplier = new_mult
            self.db.save_calibration({
                "timestamp": utc_now(),
                "parameter_name": "atr_stop_multiplier",
                "old_value": old_mult,
                "new_value": new_mult,
                "reason": reason,
                "trade_count_at_calibration": stats["total_trades"],
            })
            logger.info("ATR stop multiplier: %.2f -> %.2f (%s)", old_mult, new_mult, reason)

    def _calibrate_confluence_threshold(self, stats: dict):
        old_threshold = self.risk.confluence_threshold
        win_rate = stats["win_rate"]
        profit_factor = stats["profit_factor"]

        if win_rate < 0.45:
            new_threshold = min(old_threshold + 5, 95)
            reason = f"Win rate low ({win_rate:.2f}), raising threshold for higher quality entries"
        elif win_rate > 0.65 and profit_factor > 2.0:
            new_threshold = max(old_threshold - 5, 55)
            reason = f"Win rate high ({win_rate:.2f}), PF={profit_factor:.1f}, lowering threshold for more entries"
        else:
            new_threshold = old_threshold
            reason = "No adjustment needed"

        if new_threshold != old_threshold:
            self.risk.confluence_threshold = new_threshold
            self.db.save_calibration({
                "timestamp": utc_now(),
                "parameter_name": "confluence_threshold",
                "old_value": float(old_threshold),
                "new_value": float(new_threshold),
                "reason": reason,
                "trade_count_at_calibration": stats["total_trades"],
            })
            logger.info("Confluence threshold: %d -> %d (%s)", old_threshold, new_threshold, reason)

    def _calibrate_position_modifier(self, stats: dict):
        old_mod = self.risk.position_size_modifier
        consecutive = self.db.get_consecutive_losses()

        if consecutive >= 5:
            new_mod = 0.25
            reason = f"{consecutive} consecutive losses, severe reduction"
        elif consecutive >= 3:
            new_mod = 0.5
            reason = f"{consecutive} consecutive losses, moderate reduction"
        elif stats["win_rate"] > 0.55 and stats["profit_factor"] > 1.5:
            new_mod = min(old_mod + 0.1, 1.0)
            reason = "Strong performance, increasing position sizes"
        else:
            new_mod = 1.0
            reason = "Reset to normal"

        if abs(new_mod - old_mod) > 0.05:
            self.risk.position_size_modifier = new_mod
            self.db.save_calibration({
                "timestamp": utc_now(),
                "parameter_name": "position_size_modifier",
                "old_value": old_mod,
                "new_value": new_mod,
                "reason": reason,
                "trade_count_at_calibration": stats["total_trades"],
            })
            logger.info("Position modifier: %.2f -> %.2f (%s)", old_mod, new_mod, reason)

    def get_calibration_status(self) -> dict:
        total = self.db.get_total_trade_count()
        trades_until_next = max(
            0, self.calibration_interval - (total - self.last_calibration_count)
        )
        recent = self.db.get_recent_calibrations(10)
        return {
            "cold_start_complete": self.cold_start_complete,
            "total_trades": total,
            "trades_until_next_calibration": trades_until_next,
            "current_params": {
                "kelly_fraction": self.risk.kelly_fraction,
                "atr_stop_multiplier": self.risk.atr_stop_multiplier,
                "confluence_threshold": self.risk.confluence_threshold,
                "position_modifier": self.risk.position_size_modifier,
                "win_rate": self.risk.win_rate,
                "avg_win": self.risk.avg_win,
                "avg_loss": self.risk.avg_loss,
            },
            "recent_calibrations": recent,
        }

    def force_calibrate(self):
        self._run_calibration()
        self.last_calibration_count = self.db.get_total_trade_count()
