"""
Strategy Module — SCALPING MODE (1m / 5m / 15m).

Ultra-fast signal generation for short-term opportunities.
Minimal gates, low threshold (35/100), rapid entries.
Designed to capture 0.3-1.5% moves multiple times per day.
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

        resistance_zones = cluster_levels(swing_highs, threshold_pct=0.003)
        support_zones = cluster_levels(swing_lows, threshold_pct=0.003)

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
        return distance_pct <= 0.015  # 1.5% proximity for scalping

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
# Confluence Scorer — SCALPING MODE
# ---------------------------------------------------------------------------

class ConfluenceScorer:
    """
    100-point confluence scoring — SCALPING mode.

    Scoring breakdown:
    - Technical (50 pts): RSI, MACD, EMA, Bollinger, S/R, entry TF
    - Momentum (20 pts): Price action, candle patterns, micro-trend
    - Volume (15 pts): Volume confirmation with generous scoring
    - Regime (10 pts): Soft bonus, never blocks
    - Sentiment (5 pts): Minimal weight — technicals dominate

    Threshold: 35/100 (ultra-low for maximum opportunity capture)
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

        # 1. Technical Score (50 points)
        tech_score, tech_detail = self._technical_score(
            symbol, indicators_primary, indicators_entry, indicators_trend, kline_history,
        )
        breakdown["technical"] = {"score": tech_score, "max": 50, "details": tech_detail}
        total += tech_score

        # 2. Momentum Score (20 points) — NEW: price action micro-patterns
        mom_score, mom_detail = self._momentum_score(indicators_entry, indicators_primary, kline_history)
        breakdown["momentum"] = {"score": mom_score, "max": 20, "details": mom_detail}
        total += mom_score

        # 3. Volume Score (15 points)
        volume_score, vol_detail = self._volume_score(indicators_primary)
        breakdown["volume"] = {"score": volume_score, "max": 15, "details": vol_detail}
        total += volume_score

        # 4. Regime Score (10 points) — soft bonus
        regime_score = self._regime_score()
        breakdown["regime"] = {"score": regime_score, "max": 10}
        total += regime_score

        # 5. Sentiment Score (5 points) — minimal weight
        sentiment_score = self._sentiment_score()
        breakdown["sentiment"] = {"score": sentiment_score, "max": 5}
        total += sentiment_score

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

        # --- RSI (12 pts) — fast RSI, wider acceptable range ---
        rsi = ind_primary.get("rsi", 50)
        if 20 <= rsi <= 40:
            rsi_score = 12
            details["rsi"] = f"RSI={rsi:.1f} (oversold — strong buy)"
        elif 40 < rsi <= 55:
            rsi_score = 10
            details["rsi"] = f"RSI={rsi:.1f} (momentum zone)"
        elif 55 < rsi <= 65:
            rsi_score = 7
            details["rsi"] = f"RSI={rsi:.1f} (strong momentum)"
        elif 65 < rsi <= 75:
            rsi_score = 4
            details["rsi"] = f"RSI={rsi:.1f} (hot but tradeable)"
        elif rsi < 20:
            rsi_score = 8
            details["rsi"] = f"RSI={rsi:.1f} (extreme oversold — risky bounce)"
        else:
            rsi_score = 2
            details["rsi"] = f"RSI={rsi:.1f} (overbought — small position only)"
        score += rsi_score

        # --- MACD (12 pts) — fast MACD (5,13,4) ---
        macd_hist = ind_primary.get("macd_histogram", 0)
        macd_line = ind_primary.get("macd", 0)
        macd_signal = ind_primary.get("macd_signal", 0)
        if macd_line > macd_signal and macd_hist > 0:
            macd_score = 12
            details["macd"] = "MACD bullish crossover + positive histogram"
        elif macd_hist > 0:
            macd_score = 9
            details["macd"] = "MACD histogram positive"
        elif macd_line > macd_signal:
            macd_score = 7
            details["macd"] = "MACD above signal (early bullish)"
        elif macd_hist > ind_primary.get("prev_macd_histogram", macd_hist - 0.0001):
            macd_score = 5
            details["macd"] = "MACD improving (turning bullish)"
        else:
            macd_score = 2
            details["macd"] = "MACD bearish (small base points)"
        score += macd_score

        # --- EMA alignment (8 pts) ---
        ema_score = 0
        ema9 = ind_primary.get("ema9", 0)
        ema21 = ind_primary.get("ema21", 0)

        if price > ema9 > ema21 and ema9 > 0:
            ema_score = 8
            details["ema"] = "Price > EMA9 > EMA21 (perfect alignment)"
        elif price > ema21 and ema21 > 0:
            ema_score = 6
            details["ema"] = "Price above EMA21 (trend intact)"
        elif price > ema9 and ema9 > 0:
            ema_score = 5
            details["ema"] = "Price above EMA9 (short-term bullish)"
        elif ema9 > 0 and abs(price - ema9) / ema9 < 0.003:
            ema_score = 4
            details["ema"] = "Price at EMA9 (potential bounce)"
        else:
            ema_score = 2
            details["ema"] = "Below EMAs (base points)"
        score += ema_score

        # --- Bollinger Bands (8 pts) ---
        bb_lower = ind_primary.get("bb_lower", 0)
        bb_upper = ind_primary.get("bb_upper", 0)
        bb_mid = ind_primary.get("bb_middle", 0)
        bb_score = 0
        if bb_lower > 0 and price <= bb_lower * 1.002:
            bb_score = 8
            details["bollinger"] = "AT lower Bollinger band (bounce!)"
        elif bb_lower > 0 and price <= bb_lower * 1.01:
            bb_score = 6
            details["bollinger"] = "Near lower Bollinger band"
        elif bb_mid > 0 and price < bb_mid:
            bb_score = 4
            details["bollinger"] = "Below BB midline (room to run)"
        elif bb_upper > 0 and price < bb_upper:
            bb_score = 2
            details["bollinger"] = "Between mid and upper BB"
        else:
            bb_score = 1
            details["bollinger"] = "Above upper BB"
        score += bb_score

        # --- S/R proximity (5 pts) ---
        sr_score = 0
        if self.sr.is_near_support(symbol, price):
            nearest = self.sr.get_nearest_support(symbol, price)
            if nearest:
                sr_score = 5
                details["sr"] = f"Near support ${nearest['level']:.4f}"
        else:
            res = self.sr.get_nearest_resistance(symbol, price)
            if res:
                dist = (res["level"] - price) / price if price > 0 else 1
                if dist > 0.02:
                    sr_score = 3
                    details["sr"] = f"Room to resistance ({dist:.1%})"
                else:
                    sr_score = 1
                    details["sr"] = "Close to resistance"
            else:
                sr_score = 2
                details["sr"] = "No clear S/R (neutral)"
        score += sr_score

        # --- Entry timeframe confirmation (5 pts) ---
        entry_rsi = ind_entry.get("rsi", 50)
        entry_close = ind_entry.get("close", 0)
        entry_ema9 = ind_entry.get("ema9", 0)
        entry_bonus = 0
        if entry_close > entry_ema9 and entry_ema9 > 0:
            entry_bonus += 3
            details["entry_tf"] = "Entry TF price > EMA9"
        else:
            entry_bonus += 1
            details["entry_tf"] = "Entry TF neutral"
        if 25 <= entry_rsi <= 60:
            entry_bonus += 2
            details["entry_rsi"] = f"Entry RSI={entry_rsi:.0f} (good)"
        else:
            entry_bonus += 1
            details["entry_rsi"] = f"Entry RSI={entry_rsi:.0f}"
        score += min(entry_bonus, 5)

        return min(score, 50), details

    def _momentum_score(self, ind_entry: dict, ind_primary: dict, history: list) -> tuple:
        """NEW: Price action and micro-momentum scoring for scalping."""
        score = 0
        details = {}

        # --- Green candle streak (6 pts) ---
        if history and len(history) >= 3:
            green_count = 0
            for k in history[-5:]:
                if k.get("close", 0) > k.get("open", 0):
                    green_count += 1
            if green_count >= 4:
                score += 6
                details["candles"] = f"{green_count}/5 green candles (strong momentum)"
            elif green_count >= 3:
                score += 4
                details["candles"] = f"{green_count}/5 green candles (building)"
            elif green_count >= 2:
                score += 2
                details["candles"] = f"{green_count}/5 green candles"
            else:
                score += 1
                details["candles"] = f"{green_count}/5 green candles (base)"

        # --- Price acceleration (7 pts) ---
        if history and len(history) >= 10:
            closes = [k["close"] for k in history]
            recent_5 = closes[-5:]
            prior_5 = closes[-10:-5]
            recent_change = (recent_5[-1] - recent_5[0]) / recent_5[0] if recent_5[0] > 0 else 0
            prior_change = (prior_5[-1] - prior_5[0]) / prior_5[0] if prior_5[0] > 0 else 0

            if recent_change > 0 and recent_change > prior_change:
                score += 7
                details["acceleration"] = f"Price accelerating +{recent_change:.2%}"
            elif recent_change > 0:
                score += 5
                details["acceleration"] = f"Price rising +{recent_change:.2%}"
            elif recent_change > -0.005:
                score += 3
                details["acceleration"] = "Price stable (potential breakout)"
            else:
                score += 1
                details["acceleration"] = f"Price declining {recent_change:.2%} (base)"

        # --- StochRSI oversold bounce (4 pts) ---
        stoch_k = ind_primary.get("stoch_k", 50)
        stoch_d = ind_primary.get("stoch_d", 50)
        if stoch_k < 25:
            score += 4
            details["stochrsi"] = f"StochRSI K={stoch_k:.0f} (oversold — bounce setup)"
        elif stoch_k < 40 and stoch_k > stoch_d:
            score += 3
            details["stochrsi"] = f"StochRSI K={stoch_k:.0f} crossing up"
        elif stoch_k < 60:
            score += 2
            details["stochrsi"] = f"StochRSI K={stoch_k:.0f} (neutral)"
        else:
            score += 1
            details["stochrsi"] = f"StochRSI K={stoch_k:.0f}"

        # --- RSI divergence bonus (3 pts) ---
        if history and len(history) >= 20:
            try:
                closes = [k["close"] for k in history]
                from src.indicators import IncrementalRSI
                rsi_calc = IncrementalRSI(Settings.strategy.RSI_PERIOD)
                rsi_values = []
                for k in history:
                    rsi_calc.update(k["close"])
                    rsi_values.append(rsi_calc.value)
                if detect_rsi_divergence(closes, rsi_values):
                    score += 3
                    details["divergence"] = "Bullish RSI divergence (+3)"
            except Exception:
                pass

        return min(score, 20), details

    def _regime_score(self) -> int:
        """Soft bonus — never blocks. Even BEAR gets base points."""
        regime_map = {
            "BULL": 10,
            "SIDEWAYS": 8,
            "HIGH_VOLATILITY": 6,
            "BEAR": 4,
        }
        return regime_map.get(self.regime.current_regime, 5)

    def _sentiment_score(self) -> int:
        """Minimal weight — technicals dominate in scalping."""
        raw = self.news.get_sentiment_points()
        # Scale from 0-15 range down to 0-5, minimum 2 base points
        scaled = min(int(raw * 5 / 15), 5)
        return max(scaled, 2)

    def _volume_score(self, ind: dict) -> tuple:
        """Generous scoring — give base points even without spike."""
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
            return 8, f"Volume {spike['ratio']:.1f}x avg (above normal)"
        elif spike["ratio"] > 0.7:
            return 5, f"Volume {spike['ratio']:.1f}x avg (acceptable)"
        elif spike["ratio"] > 0.4:
            return 3, f"Volume {spike['ratio']:.1f}x avg (low but not blocking)"
        else:
            return 2, f"Volume {spike['ratio']:.1f}x avg (base points)"


# ---------------------------------------------------------------------------
# Signal Generator — SCALPING MODE (no hard gates)
# ---------------------------------------------------------------------------

class SignalGenerator:
    """
    Generates trade signals — SCALPING mode.

    NO hard gates at all. Every asset gets scored.
    Only the confluence threshold determines if a trade happens.
    AI is advisory only (never blocks).
    Designed for maximum opportunity capture.
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
        """Streamlined evaluation: score everything, no gates, just threshold."""
        regime_params = self.regime.get_regime_params()

        # NO GATES — every asset gets scored regardless of regime, sentiment, etc.

        # Score the setup
        result = self.scorer.score(
            symbol, indicators_entry, indicators_primary, indicators_trend,
            kline_history, regime_params,
        )

        if not result["passes"]:
            return None

        price = indicators_primary.get("close", 0)
        atr = indicators_primary.get("atr", 0)

        # Build signal with scalping-optimized SL/TP
        sl_pct = Settings.risk.STOP_LOSS_PCT      # 0.02 (2%)
        tp_pct = Settings.risk.TAKE_PROFIT_PCT     # 0.03 (3%)

        signal = {
            "symbol": symbol,
            "price": price,
            "confluence_score": result["total_score"],
            "breakdown": result["breakdown"],
            "stop_loss": price * (1 - sl_pct),
            "take_profit": price * (1 + tp_pct),
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

        # AI analysis — ADVISORY ONLY, never blocks
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
                signal["ai_analysis"] = ai_result
                # Enforce veto power if configured
                approval = self.ai.should_approve_entry(ai_result, result["total_score"])
                signal["ai_approval"] = approval

                if Settings.ai.VETO_POWER and not approval.get("approved", True):
                    logger.info(
                        "AI vetoed %s (confidence=%.0f%%): %s",
                        symbol, ai_result.get("confidence", 0) * 100,
                        approval.get("reason", "")[:80],
                    )
                    return None

                # AI can suggest tighter SL/TP
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
