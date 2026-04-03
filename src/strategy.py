"""
Strategy Module — AGGRESSIVE short-term confluence scoring.

Loosened gates for fast scalping/swing trades. Fewer confirmation layers,
lower thresholds, AI advisory (non-blocking), and rapid signal generation.
"""

import json
from typing import Optional

from config.settings import Settings
from src.utils import setup_logging, utc_now
from src.indicators import detect_swing_points, cluster_levels, detect_rsi_divergence

logger = setup_logging()


# ---------------------------------------------------------------------------
# Support / Resistance Engine
# ---------------------------------------------------------------------------

class SupportResistanceEngine:
    """Detects and tracks support/resistance levels from price history."""

    def __init__(self):
        self.levels_cache: dict = {}

    def compute_levels(self, symbol: str, kline_history: list) -> dict:
        closes = [k["close"] for k in kline_history]
        highs = [k["high"] for k in kline_history]
        lows = [k["low"] for k in kline_history]
        if len(closes) < 20:
            return {"support": [], "resistance": []}

        swings = detect_swing_points(closes, lookback=3)
        swing_highs = swings["swing_highs"]
        swing_lows = swings["swing_lows"]

        wick_swings_h = detect_swing_points(highs, lookback=3)
        wick_swings_l = detect_swing_points(lows, lookback=3)
        swing_highs += wick_swings_h["swing_highs"]
        swing_lows += wick_swings_l["swing_lows"]

        resistance_zones = cluster_levels(swing_highs, threshold_pct=0.005)
        support_zones = cluster_levels(swing_lows, threshold_pct=0.005)

        current_price = closes[-1]

        support = sorted(
            [z for z in support_zones if z["level"] < current_price],
            key=lambda x: x["level"], reverse=True,
        )
        resistance = sorted(
            [z for z in resistance_zones if z["level"] > current_price],
            key=lambda x: x["level"],
        )

        result = {
            "support": support[:10],
            "resistance": resistance[:10],
        }
        self.levels_cache[symbol] = result
        return result

    def get_nearest_support(self, symbol: str, current_price: float) -> Optional[dict]:
        levels = self.levels_cache.get(symbol, {}).get("support", [])
        for level in levels:
            if level["level"] < current_price:
                return level
        return None

    def get_nearest_resistance(self, symbol: str, current_price: float) -> Optional[dict]:
        levels = self.levels_cache.get(symbol, {}).get("resistance", [])
        for level in levels:
            if level["level"] > current_price:
                return level
        return None

    def check_support_break(self, symbol: str, current_price: float) -> Optional[dict]:
        nearest = self.get_nearest_support(symbol, current_price * 1.02)
        if nearest is None:
            return None
        support_price = nearest["level"]
        threshold = Settings.risk.SUPPORT_BREAK_THRESHOLD
        if current_price < support_price * (1 - threshold):
            return {
                "broken": True,
                "support_level": support_price,
                "break_pct": ((current_price / support_price) - 1) * 100,
                "touches": nearest.get("touches", 1),
            }
        return None

    def is_near_support(self, symbol: str, current_price: float) -> bool:
        nearest = self.get_nearest_support(symbol, current_price)
        if nearest is None:
            return False
        distance_pct = abs(current_price - nearest["level"]) / nearest["level"]
        return distance_pct <= Settings.strategy.SR_PROXIMITY_PCT

    def get_support_prices(self, symbol: str) -> list:
        return [z["level"] for z in self.levels_cache.get(symbol, {}).get("support", [])]

    def get_resistance_prices(self, symbol: str) -> list:
        return [z["level"] for z in self.levels_cache.get(symbol, {}).get("resistance", [])]


# ---------------------------------------------------------------------------
# Volume Spike Detector
# ---------------------------------------------------------------------------

class VolumeSpikeDetector:
    """Detects volume spikes that confirm breakouts or signal reversals."""

    @staticmethod
    def detect_spike(volume: float, volume_sma: float) -> dict:
        if volume_sma <= 0:
            return {"is_spike": False, "ratio": 0}
        ratio = volume / volume_sma
        multiplier = Settings.strategy.VOLUME_SPIKE_MULTIPLIER
        return {
            "is_spike": ratio >= multiplier,
            "ratio": ratio,
            "strength": (
                "EXTREME" if ratio >= 4.0
                else "STRONG" if ratio >= 3.0
                else "MODERATE" if ratio >= multiplier
                else "NORMAL"
            ),
        }

    @staticmethod
    def detect_volume_trend(volumes: list, lookback: int = 5) -> str:
        if len(volumes) < lookback + 1:
            return "UNKNOWN"
        recent = sum(volumes[-lookback:]) / lookback
        prior = sum(volumes[-lookback * 2:-lookback]) / lookback if len(volumes) >= lookback * 2 else recent
        if prior == 0:
            return "UNKNOWN"
        ratio = recent / prior
        if ratio > 1.5:
            return "INCREASING"
        elif ratio < 0.7:
            return "DECREASING"
        return "STABLE"


# ---------------------------------------------------------------------------
# Confluence Scorer — AGGRESSIVE (loosened)
# ---------------------------------------------------------------------------

class ConfluenceScorer:
    """
    100-point confluence scoring — AGGRESSIVE mode.

    Key changes from conservative mode:
    - Technical score still dominates (50 pts) but easier to achieve
    - Regime score reduced (15 pts) — sideways markets still allowed
    - Sentiment reduced (10 pts) — don't over-filter on news
    - Volume gives points even without spikes (15 pts)
    - Risk/reward still matters (10 pts)
    - Threshold lowered to 45 (from 70) via .env
    """

    def __init__(self, regime_detector, news_intel, database, sr_engine: SupportResistanceEngine):
        self.regime = regime_detector
        self.news = news_intel
        self.db = database
        self.sr = sr_engine

    def score(self, symbol: str, indicators_entry: dict, indicators_primary: dict,
              indicators_trend: dict, kline_history: list,
              regime_params: dict) -> dict:
        breakdown = {}
        total = 0

        # 1. Technical Score (50 points) — easier to achieve
        tech_score, tech_detail = self._technical_score(
            symbol, indicators_primary, indicators_entry, indicators_trend, kline_history,
        )
        breakdown["technical"] = {"score": tech_score, "max": 50, "details": tech_detail}
        total += tech_score

        # 2. Regime Score (15 points) — sideways still gets points
        regime_score = self._regime_score()
        breakdown["regime"] = {"score": regime_score, "max": 15}
        total += regime_score

        # 3. Sentiment Score (10 points) — reduced weight
        sentiment_score = self._sentiment_score()
        breakdown["sentiment"] = {"score": sentiment_score, "max": 10}
        total += sentiment_score

        # 4. Volume Score (15 points) — gives partial credit
        volume_score, vol_detail = self._volume_score(indicators_primary)
        breakdown["volume"] = {"score": volume_score, "max": 15, "details": vol_detail}
        total += volume_score

        # 5. Risk/Reward Score (10 points)
        rr_score, rr_detail = self._risk_reward_score(symbol, indicators_primary)
        breakdown["risk_reward"] = {"score": rr_score, "max": 10, "details": rr_detail}
        total += rr_score

        threshold = regime_params.get("confluence_threshold",
                                       Settings.strategy.CONFLUENCE_SCORE_THRESHOLD)

        self.db.save_signal({
            "timestamp": utc_now(),
            "symbol": symbol,
            "confluence_score": total,
            "score_breakdown": json.dumps(breakdown),
            "action_taken": "PENDING",
            "regime": self.regime.current_regime,
        })

        return {
            "symbol": symbol,
            "total_score": total,
            "breakdown": breakdown,
            "threshold": threshold,
            "passes": total >= threshold,
        }

    def _technical_score(self, symbol: str, ind_primary: dict, ind_entry: dict,
                         ind_trend: dict, history: list) -> tuple:
        score = 0
        details = {}
        price = ind_primary.get("close", 0)

        # --- RSI (15 pts) — wider acceptable range ---
        rsi = ind_primary.get("rsi", 50)
        if 25 <= rsi <= 45:
            rsi_score = 15
            details["rsi"] = f"RSI={rsi:.1f} (oversold recovery — strong buy zone)"
        elif 45 < rsi <= 60:
            rsi_score = 12
            details["rsi"] = f"RSI={rsi:.1f} (momentum zone — good)"
        elif 20 <= rsi < 25:
            rsi_score = 10
            details["rsi"] = f"RSI={rsi:.1f} (deep oversold — reversal play)"
        elif 60 < rsi <= 70:
            rsi_score = 6
            details["rsi"] = f"RSI={rsi:.1f} (strong momentum — still tradeable)"
        elif rsi > 75:
            rsi_score = 0
            details["rsi"] = f"RSI={rsi:.1f} (overbought — skip)"
        else:
            rsi_score = 4
            details["rsi"] = f"RSI={rsi:.1f}"
        score += rsi_score

        # --- MACD (15 pts) — more generous scoring ---
        macd_hist = ind_primary.get("macd_histogram", 0)
        macd_line = ind_primary.get("macd", 0)
        macd_signal = ind_primary.get("macd_signal", 0)
        if macd_line > macd_signal and macd_hist > 0:
            macd_score = 15
            details["macd"] = "MACD bullish crossover + positive histogram"
        elif macd_hist > 0:
            macd_score = 10
            details["macd"] = "MACD histogram positive (momentum building)"
        elif macd_line > macd_signal:
            macd_score = 8
            details["macd"] = "MACD above signal (early bullish)"
        elif macd_hist > ind_primary.get("prev_macd_histogram", macd_hist - 1):
            macd_score = 5
            details["macd"] = "MACD histogram improving (turning bullish)"
        else:
            macd_score = 0
            details["macd"] = "MACD bearish"
        score += macd_score

        # --- EMA alignment + Bollinger (10 pts) ---
        ema_score = 0
        ema9 = ind_primary.get("ema9", 0)
        ema21 = ind_primary.get("ema21", 0)
        bb_lower = ind_primary.get("bb_lower", 0)

        if price > ema9 > ema21 and ema9 > 0:
            ema_score += 5
            details["ema"] = "Price > EMA9 > EMA21 (perfect alignment)"
        elif price > ema21 and ema21 > 0:
            ema_score += 4
            details["ema"] = "Price above EMA21 (trend intact)"
        elif price > ema9 and ema9 > 0:
            ema_score += 3
            details["ema"] = "Price above EMA9 (short-term bullish)"
        else:
            ema_score += 0
            details["ema"] = "Below EMAs"

        # Bollinger bounce — great short-term signal
        if bb_lower > 0 and price <= bb_lower * 1.005:
            ema_score += 5
            details["bollinger"] = "At lower Bollinger band (bounce setup!)"
        elif bb_lower > 0 and price <= bb_lower * 1.015:
            ema_score += 3
            details["bollinger"] = "Near lower Bollinger band"
        score += min(ema_score, 10)

        # --- S/R proximity bonus (5 pts) ---
        sr_score = 0
        if self.sr.is_near_support(symbol, price):
            nearest = self.sr.get_nearest_support(symbol, price)
            if nearest:
                sr_score = 5
                details["sr"] = f"Near support ${nearest['level']:.4f}"
        else:
            # Still give points if there's room to resistance
            res = self.sr.get_nearest_resistance(symbol, price)
            if res:
                dist = (res["level"] - price) / price if price > 0 else 1
                if dist > 0.03:
                    sr_score = 3
                    details["sr"] = f"Room to resistance ({dist:.1%} away)"
                else:
                    sr_score = 1
                    details["sr"] = "Close to resistance"
        score += sr_score

        # --- Entry timeframe momentum bonus (5 pts) ---
        entry_rsi = ind_entry.get("rsi", 50)
        entry_close = ind_entry.get("close", 0)
        entry_ema9 = ind_entry.get("ema9", 0)
        entry_bonus = 0
        if entry_close > entry_ema9 and entry_ema9 > 0:
            entry_bonus += 3
            details["entry_tf"] = "Entry TF price > EMA9 (timing confirmed)"
        if 30 <= entry_rsi <= 60:
            entry_bonus += 2
            details["entry_rsi"] = f"Entry TF RSI={entry_rsi:.0f} (good zone)"
        score += min(entry_bonus, 5)

        # --- RSI divergence bonus (up to +3) ---
        closes = [k["close"] for k in history] if history else []
        if len(closes) >= 20:
            try:
                from src.indicators import IncrementalRSI
                rsi_calc = IncrementalRSI(Settings.strategy.RSI_PERIOD)
                rsi_values = []
                for k in history:
                    rsi_calc.update(k["close"])
                    rsi_values.append(rsi_calc.value)
                if detect_rsi_divergence(closes, rsi_values):
                    score += 3
                    details["divergence"] = "Bullish RSI divergence (+3 bonus)"
            except Exception:
                pass

        return min(score, 50), details

    def _regime_score(self) -> int:
        """Sideways markets still get decent points — we trade in all conditions."""
        regime_map = {
            "BULL": 15,
            "SIDEWAYS": 10,
            "HIGH_VOLATILITY": 7,
            "BEAR": 3,
        }
        return regime_map.get(self.regime.current_regime, 5)

    def _sentiment_score(self) -> int:
        """Reduced weight — don't let sentiment block good technical setups."""
        raw = self.news.get_sentiment_points()
        # Scale from 0-15 range down to 0-10
        return min(int(raw * 10 / 15), 10)

    def _volume_score(self, ind: dict) -> tuple:
        """More generous — give partial credit even without a spike."""
        volume = ind.get("volume", 0)
        vol_sma = ind.get("volume_sma", 1)
        spike = VolumeSpikeDetector.detect_spike(volume, vol_sma)

        if spike["is_spike"]:
            if spike["strength"] == "EXTREME":
                return 15, f"EXTREME volume {spike['ratio']:.1f}x"
            elif spike["strength"] == "STRONG":
                return 13, f"STRONG volume {spike['ratio']:.1f}x"
            else:
                return 10, f"Volume spike {spike['ratio']:.1f}x"
        elif spike["ratio"] > 1.0:
            return 7, f"Volume {spike['ratio']:.1f}x avg (above normal)"
        elif spike["ratio"] > 0.7:
            return 4, f"Volume {spike['ratio']:.1f}x avg (acceptable)"
        else:
            return 2, f"Volume {spike['ratio']:.1f}x avg (low but not blocking)"

    def _risk_reward_score(self, symbol: str, ind: dict) -> tuple:
        price = ind.get("close", 0)
        if price == 0:
            return 0, "Price unavailable"

        sl_pct = Settings.risk.STOP_LOSS_PCT
        tp_pct = Settings.risk.TAKE_PROFIT_PCT
        rr = tp_pct / sl_pct if sl_pct > 0 else 0

        nearest_support = self.sr.get_nearest_support(symbol, price)
        if nearest_support:
            support_dist = (price - nearest_support["level"]) / price
            if 0 < support_dist < sl_pct:
                effective_rr = tp_pct / support_dist
                if effective_rr > rr:
                    rr = effective_rr

        if rr > 2.5:
            return 10, f"R:R = {rr:.1f}:1 (excellent)"
        elif rr > 1.5:
            return 7, f"R:R = {rr:.1f}:1 (good)"
        elif rr > 1.0:
            return 5, f"R:R = {rr:.1f}:1 (acceptable)"
        else:
            return 3, f"R:R = {rr:.1f}:1 (tight but tradeable)"


# ---------------------------------------------------------------------------
# Signal Generator — AGGRESSIVE (fewer gates)
# ---------------------------------------------------------------------------

class SignalGenerator:
    """
    Generates trade signals — AGGRESSIVE mode.

    Only 2 hard gates remain:
    1. Regime must not be deep BEAR (still allows SIDEWAYS, HIGH_VOL)
    2. Confluence score must pass threshold (lowered to 45)

    AI is advisory only (no veto power by default).
    No extreme greed block. No 4h SMA50 gate. No 15m EMA9 gate.
    """

    def __init__(self, scorer: ConfluenceScorer, regime_detector, news_intel,
                 ai_advisor=None):
        self.scorer = scorer
        self.regime = regime_detector
        self.news = news_intel
        self.ai = ai_advisor

    async def evaluate(self, symbol: str, indicators_entry: dict,
                       indicators_primary: dict, indicators_trend: dict,
                       kline_history: list,
                       portfolio_context: dict = None) -> Optional[dict]:
        """Streamlined evaluation: confluence score -> optional AI -> signal."""
        regime_params = self.regime.get_regime_params()

        # Gate 1: Only block in confirmed deep BEAR with no recovery signs
        if self.regime.current_regime == "BEAR":
            bear_rsi = indicators_primary.get("rsi", 50)
            # Even in bear, allow oversold bounces (RSI < 35)
            if bear_rsi > 40 and not regime_params.get("entries_allowed", False):
                return None

        # Score the setup
        result = self.scorer.score(
            symbol, indicators_entry, indicators_primary, indicators_trend,
            kline_history, regime_params,
        )

        if not result["passes"]:
            return None

        price = indicators_primary.get("close", 0)
        atr = indicators_primary.get("atr", 0)

        # Build signal
        signal = {
            "symbol": symbol,
            "price": price,
            "confluence_score": result["total_score"],
            "breakdown": result["breakdown"],
            "stop_loss": price * (1 - Settings.risk.STOP_LOSS_PCT),
            "take_profit": price * (1 + Settings.risk.TAKE_PROFIT_PCT),
            "atr": atr,
            "regime": self.regime.current_regime,
            "fear_greed": self.news.fear_greed_value,
            "news_sentiment": self.news.get_sentiment_score(),
            "volume_spike": VolumeSpikeDetector.detect_spike(
                indicators_primary.get("volume", 0),
                indicators_primary.get("volume_sma", 1),
            ),
            "support_levels": self.scorer.sr.get_support_prices(symbol),
            "resistance_levels": self.scorer.sr.get_resistance_prices(symbol),
        }

        # AI analysis — advisory only, does NOT block trades unless AI_VETO_POWER=true
        if self.ai and Settings.ai.ENABLED:
            try:
                recent_news = self.news.get_top_headlines(5)
                ai_result = await self.ai.analyze_trade(
                    symbol=symbol,
                    indicators=indicators_primary,
                    regime=self.regime.current_regime,
                    news_sentiment=self.news.get_sentiment_score(),
                    fear_greed=self.news.fear_greed_value,
                    support_levels=signal["support_levels"],
                    resistance_levels=signal["resistance_levels"],
                    recent_news=recent_news,
                    portfolio_context=portfolio_context or {},
                )
                approval = self.ai.should_approve_entry(ai_result, result["total_score"])
                signal["ai_analysis"] = ai_result
                signal["ai_approval"] = approval

                # Only block if AI_VETO_POWER is explicitly enabled
                if Settings.ai.VETO_POWER and not approval["approved"]:
                    logger.info("AI vetoed %s: %s", symbol, approval["reason"])
                    self.scorer.db.save_signal({
                        "timestamp": utc_now(), "symbol": symbol,
                        "confluence_score": result["total_score"],
                        "score_breakdown": json.dumps(result["breakdown"]),
                        "action_taken": "AI_VETOED",
                        "regime": self.regime.current_regime,
                    })
                    return None

                # AI can still suggest tighter SL/TP even in advisory mode
                if approval.get("ai_sl"):
                    ai_sl = price * (1 - approval["ai_sl"])
                    signal["stop_loss"] = max(signal["stop_loss"], ai_sl)
                if approval.get("ai_tp"):
                    ai_tp = price * (1 + approval["ai_tp"])
                    signal["take_profit"] = min(signal["take_profit"], ai_tp)

            except Exception as e:
                logger.warning("AI analysis failed for %s (non-blocking): %s", symbol, e)
                signal["ai_analysis"] = None
                signal["ai_approval"] = {"approved": True, "reason": "AI unavailable"}

        return signal
