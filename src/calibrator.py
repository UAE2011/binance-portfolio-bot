"""
Calibrator — Walk-forward optimization of key parameters based on trade history.
Runs every N trades or on /calibrate command.
"""
from config.settings import Settings
from src.utils import setup_logging, utc_now

logger = setup_logging()


class Calibrator:
    def __init__(self, database, risk_manager, regime_detector, notifier):
        self.db = database
        self.risk = risk_manager
        self.regime = regime_detector
        self.notifier = notifier
        self.cfg = Settings.calibration
        self._trades_since_calibration = 0

    def on_trade_closed(self):
        self._trades_since_calibration += 1
        if (self.cfg.ENABLED
                and self._trades_since_calibration >= self.cfg.CALIBRATION_INTERVAL_TRADES):
            self.check_and_calibrate()
            self._trades_since_calibration = 0

    def check_and_calibrate(self):
        stats = self.db.get_trade_stats(self.cfg.LOOKBACK_TRADES)
        total = stats.get("total_trades", 0)
        if total < self.cfg.MIN_TRADES:
            logger.info("Calibration skipped — only %d trades (need %d)", total, self.cfg.MIN_TRADES)
            return

        logger.info("Calibrating parameters on %d trades...", total)
        changes = []

        # 1. Update win rate and Kelly fraction
        old_kelly = self.risk.kelly_fraction
        old_win_rate = self.risk.win_rate
        self.risk.win_rate = stats["win_rate"]
        self.risk.avg_win = max(stats["avg_win"], 0.005)
        self.risk.avg_loss = max(stats["avg_loss"], 0.002)
        self.risk._recalculate_kelly()
        new_kelly = self.risk.kelly_fraction

        if abs(new_kelly - old_kelly) > 0.002:
            changes.append({
                "parameter_name": "kelly_fraction",
                "old_value": old_kelly, "new_value": new_kelly,
                "reason": f"Win rate changed {old_win_rate:.1%}→{stats['win_rate']:.1%}",
                "trades_analyzed": total,
            })

        # 2. Confluence threshold: raise if win rate is low, lower if profitable
        old_threshold = self.risk.confluence_threshold
        profit_factor = stats.get("profit_factor", 1.0)
        sl_hit_rate = stats.get("stop_loss_hit_rate", 0)
        new_threshold = old_threshold

        if profit_factor < 1.0 or sl_hit_rate > 0.6:
            # Many stops hit → raise bar for entries
            new_threshold = min(old_threshold + 3, 70)
        elif profit_factor > 2.0 and stats["win_rate"] > 0.60:
            # High PF + win rate → can lower threshold for more trades
            new_threshold = max(old_threshold - 3, 35)

        if new_threshold != old_threshold:
            self.risk.confluence_threshold = new_threshold
            changes.append({
                "parameter_name": "confluence_threshold",
                "old_value": old_threshold, "new_value": new_threshold,
                "reason": f"PF={profit_factor:.2f} SL_hit={sl_hit_rate:.1%}",
                "trades_analyzed": total,
            })

        # 3. ATR stop multiplier: tighten if low SL hits, widen if many stop-outs
        old_atr_mult = self.risk.atr_stop_multiplier
        new_atr_mult = old_atr_mult

        if sl_hit_rate > 0.50:
            new_atr_mult = min(old_atr_mult + 0.25, 5.0)  # wider stops
        elif sl_hit_rate < 0.15 and profit_factor > 1.5:
            new_atr_mult = max(old_atr_mult - 0.25, 2.0)  # tighter stops = better R:R

        if abs(new_atr_mult - old_atr_mult) >= 0.25:
            self.risk.atr_stop_multiplier = new_atr_mult
            changes.append({
                "parameter_name": "atr_stop_multiplier",
                "old_value": old_atr_mult, "new_value": new_atr_mult,
                "reason": f"SL hit rate {sl_hit_rate:.1%}",
                "trades_analyzed": total,
            })

        for change in changes:
            self.db.save_calibration(change)
            import asyncio
            asyncio.create_task(self.notifier.notify_calibration(change))
            logger.info("Calibrated %s: %.4f → %.4f (%s)",
                        change["parameter_name"], change["old_value"],
                        change["new_value"], change["reason"])

        if not changes:
            logger.info("Calibration: no adjustments needed")
