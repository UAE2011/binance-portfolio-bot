"""
Strategy Module — Confluence scoring, signal generation, support break exit,
volume spike detection, and multi-timeframe (15m + 1hr + 4h) analysis.

Integrates all SOL bot features into a diversified portfolio context.
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
        self.levels_cache: dict = {}  # symbol -> {"support": [...], "resistance": [...]}

    def compute_levels(self, symbol: str, kline_history: list) -> dict:
        closes = [k["close"] for k in kline_history]
        highs = [k["high"] for k in kline_history]
        lows = [k["low"] for k in kline_history]
        if len(closes) < 30:
            return {"support": [], "resistance": []}

        swings = detect_swing_points(closes, lookback=5)
        swing_highs = swings["swing_highs"]
        swing_lows = swings["swing_lows"]

        # Also detect from actual high/low wicks
        wick_swings_h = detect_swing_points(highs, lookback=5)
        wick_swings_l = detect_swing_points(lows, lookback=5)
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
        """Check if price has broken below nearest support — triggers exit."""
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
            "strength": "EXTREME" if ratio >= 4.0 else "STRONG" if ratio >= 3.0 else "MODERATE" if ratio >= multiplier else "NORMAL",
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
# Confluence Scorer
# ---------------------------------------------------------------------------

class ConfluenceScorer:
    """100-point confluence scoring system across 5 dimensions."""

    def __init__(self, regime_detector, news_intel, database, sr_engine: SupportResistanceEngine):
        self.regime = regime_detector
        self.news = news_intel
        self.db = database
        self.sr = sr_engine

    def score(self, symbol: str, indicators_15m: dict, indicators_1h: dict,
              indicators_4h: dict, kline_history_1h: list,
              regime_params: dict) -> dict:
        breakdown = {}
        total = 0

        # 1. Technical Score (40 points) — uses 1hr as primary
        tech_score, tech_detail = self._technical_score(
            symbol, indicators_1h, indicators_15m, indicators_4h, kline_history_1h,
        )
        breakdown["technical"] = {"score": tech_score, "max": 40, "details": tech_detail}
        total += tech_score

        # 2. Regime Score (20 points)
        regime_score = self._regime_score()
        breakdown["regime"] = {"score": regime_score, "max": 20}
        total += regime_score

        # 3. Sentiment Score (15 points)
        sentiment_score = self._sentiment_score()
        breakdown["sentiment"] = {"score": sentiment_score, "max": 15}
        total += sentiment_score

        # 4. Volume Score (15 points) — includes spike detection
        volume_score, vol_detail = self._volume_score(indicators_1h)
        breakdown["volume"] = {"score": volume_score, "max": 15, "details": vol_detail}
        total += volume_score

        # 5. Risk/Reward Score (10 points)
        rr_score, rr_detail = self._risk_reward_score(symbol, indicators_1h)
        breakdown["risk_reward"] = {"score": rr_score, "max": 10, "details": rr_detail}
        total += rr_score

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
            "threshold": regime_params.get("confluence_threshold", 70),
            "passes": total >= regime_params.get("confluence_threshold", 70),
        }

    def _technical_score(self, symbol: str, ind_1h: dict, ind_15m: dict,
                         ind_4h: dict, history_1h: list) -> tuple:
        score = 0
        details = {}

        # --- S/R proximity (10 pts) ---
        price = ind_1h.get("close", 0)
        sr_score = 0
        if self.sr.is_near_support(symbol, price):
            nearest = self.sr.get_nearest_support(symbol, price)
            if nearest:
                touches = nearest.get("touches", 1)
                sr_score = min(10, 5 + touches)
                details["sr_level"] = f"Near support ${nearest['level']:.2f} ({touches} touches)"
        breakdown_sr = self.sr.levels_cache.get(symbol, {})
        if not sr_score and breakdown_sr:
            res = self.sr.get_nearest_resistance(symbol, price)
            if res:
                dist = (res["level"] - price) / price if price > 0 else 1
                if dist > 0.05:
                    sr_score = 3
                    details["sr_level"] = f"Room to resistance ${res['level']:.2f} ({dist:.1%} away)"
        score += sr_score

        # --- RSI zone (10 pts) ---
        rsi = ind_1h.get("rsi", 50)
        rsi_score = 0
        if 30 <= rsi <= 45:
            rsi_score = 10
            details["rsi"] = f"RSI={rsi:.1f} (recovering from oversold — ideal)"
        elif 45 < rsi <= 55:
            rsi_score = 6
            details["rsi"] = f"RSI={rsi:.1f} (neutral zone)"
        elif 20 <= rsi < 30:
            rsi_score = 7
            details["rsi"] = f"RSI={rsi:.1f} (oversold — reversal potential)"
        elif rsi > 70:
            rsi_score = 0
            details["rsi"] = f"RSI={rsi:.1f} (overbought — no entry)"
        else:
            rsi_score = 3
            details["rsi"] = f"RSI={rsi:.1f}"
        score += rsi_score

        # --- MACD (10 pts) ---
        macd_score = 0
        macd_hist = ind_1h.get("macd_histogram", 0)
        macd_line = ind_1h.get("macd", 0)
        macd_signal = ind_1h.get("macd_signal", 0)
        if macd_line > macd_signal and macd_hist > 0:
            macd_score = 10
            details["macd"] = "MACD bullish crossover + positive histogram"
        elif macd_hist > 0:
            macd_score = 5
            details["macd"] = "MACD histogram turning positive"
        elif macd_line > macd_signal:
            macd_score = 3
            details["macd"] = "MACD above signal but histogram negative"
        else:
            details["macd"] = "MACD bearish"
        score += macd_score

        # --- EMA alignment + Bollinger (10 pts) ---
        ema_score = 0
        ema9 = ind_1h.get("ema9", 0)
        ema21 = ind_1h.get("ema21", 0)
        bb_lower = ind_1h.get("bb_lower", 0)
        bb_upper = ind_1h.get("bb_upper", 0)
        bb_middle = ind_1h.get("bb_middle", 0)

        if price > ema9 > ema21 and ema9 > 0:
            ema_score += 5
            details["ema"] = "Price > EMA9 > EMA21 (perfect alignment)"
        elif price > ema21 and ema21 > 0:
            ema_score += 3
            details["ema"] = "Price above EMA21"
        else:
            details["ema"] = "EMA bearish alignment"

        # Bollinger bounce from lower band
        if bb_lower > 0 and price <= bb_lower * 1.01:
            ema_score += 5
            details["bollinger"] = "Price at lower Bollinger band (bounce setup)"
        elif bb_middle > 0 and price < bb_middle:
            ema_score += 2
            details["bollinger"] = "Price below Bollinger middle"
        else:
            details["bollinger"] = "Price in upper Bollinger range"
        score += min(ema_score, 10)

        # --- Multi-timeframe confirmation bonus (up to +5 capped at 40) ---
        mtf_bonus = 0
        # 15m confirmation: price above EMA9 on 15m
        if ind_15m.get("close", 0) > ind_15m.get("ema9", float("inf")):
            mtf_bonus += 2
            details["mtf_15m"] = "15m price above EMA9 (entry timing confirmed)"

        # 4h trend confirmation: price above EMA21 on 4h
        if ind_4h.get("close", 0) > ind_4h.get("ema21", float("inf")):
            mtf_bonus += 3
            details["mtf_4h"] = "4h trend bullish (price > EMA21)"

        score += mtf_bonus

        # --- RSI divergence bonus ---
        closes = [k["close"] for k in history_1h] if history_1h else []
        if len(closes) >= 30:
            from src.indicators import IncrementalRSI
            rsi_calc = IncrementalRSI(14)
            rsi_values = []
            for k in history_1h:
                rsi_calc.update(k["close"])
                rsi_values.append(rsi_calc.value)
            if detect_rsi_divergence(closes, rsi_values):
                score += 3
                details["divergence"] = "Bullish RSI divergence detected (+3 bonus)"

        return min(score, 40), details

    def _regime_score(self) -> int:
        regime_map = {
            "BULL": 20,
            "SIDEWAYS": 10,
            "HIGH_VOLATILITY": 5,
            "BEAR": 0,
        }
        return regime_map.get(self.regime.current_regime, 0)

    def _sentiment_score(self) -> int:
        return self.news.get_sentiment_points()

    def _volume_score(self, ind: dict) -> tuple:
        volume = ind.get("volume", 0)
        vol_sma = ind.get("volume_sma", 1)
        spike = VolumeSpikeDetector.detect_spike(volume, vol_sma)

        if spike["is_spike"]:
            if spike["strength"] == "EXTREME":
                return 15, f"EXTREME volume spike {spike['ratio']:.1f}x (very strong confirmation)"
            elif spike["strength"] == "STRONG":
                return 13, f"STRONG volume spike {spike['ratio']:.1f}x (strong confirmation)"
            else:
                return 10, f"Volume spike {spike['ratio']:.1f}x (moderate confirmation)"
        elif spike["ratio"] > 1.2:
            return 5, f"Volume {spike['ratio']:.1f}x average (above normal)"
        else:
            return 0, f"Volume {spike['ratio']:.1f}x average (below threshold)"

    def _risk_reward_score(self, symbol: str, ind: dict) -> tuple:
        price = ind.get("close", 0)
        if price == 0:
            return 0, "Price unavailable"

        # Use fixed 3% SL and 6% TP from settings
        sl_pct = Settings.risk.STOP_LOSS_PCT
        tp_pct = Settings.risk.TAKE_PROFIT_PCT
        rr = tp_pct / sl_pct if sl_pct > 0 else 0

        # Bonus if near support (tighter stop possible)
        nearest_support = self.sr.get_nearest_support(symbol, price)
        if nearest_support:
            support_dist = (price - nearest_support["level"]) / price
            if 0 < support_dist < sl_pct:
                effective_rr = tp_pct / support_dist
                if effective_rr > rr:
                    rr = effective_rr

        if rr > 3.0:
            return 10, f"R:R = {rr:.1f}:1 (excellent)"
        elif rr > 2.0:
            return 7, f"R:R = {rr:.1f}:1 (good)"
        elif rr > 1.5:
            return 4, f"R:R = {rr:.1f}:1 (acceptable)"
        else:
            return 0, f"R:R = {rr:.1f}:1 (too low)"


# ---------------------------------------------------------------------------
# Signal Generator
# ---------------------------------------------------------------------------

class SignalGenerator:
    """Generates trade signals using multi-timeframe confluence + AI advisor."""

    def __init__(self, scorer: ConfluenceScorer, regime_detector, news_intel,
                 ai_advisor=None):
        self.scorer = scorer
        self.regime = regime_detector
        self.news = news_intel
        self.ai = ai_advisor

    async def evaluate(self, symbol: str, indicators_15m: dict,
                       indicators_1h: dict, indicators_4h: dict,
                       kline_history_1h: list,
                       portfolio_context: dict = None) -> Optional[dict]:
        """Full evaluation pipeline: confluence -> AI -> final signal."""
        regime_params = self.regime.get_regime_params()

        # Gate 1: Regime allows entries?
        if not regime_params.get("entries_allowed", False):
            return None

        # Gate 2: Not extreme greed
        if self.news.is_extreme_greed():
            return None

        # Gate 3: 4h trend filter — price must be above SMA50 on daily (unless BULL)
        if indicators_4h.get("close", 0) < indicators_4h.get("sma50", float("inf")):
            if self.regime.current_regime != "BULL":
                return None

        # Score the setup
        result = self.scorer.score(
            symbol, indicators_15m, indicators_1h, indicators_4h,
            kline_history_1h, regime_params,
        )

        if not result["passes"]:
            return None

        # Gate 4: 15m entry timing — price should be above EMA9 on 15m
        if indicators_15m.get("close", 0) <= indicators_15m.get("ema9", float("inf")):
            result["total_score"] = max(result["total_score"] - 5, 0)
            if result["total_score"] < regime_params.get("confluence_threshold", 70):
                return None

        price = indicators_1h.get("close", 0)
        atr = indicators_1h.get("atr", 0)

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
                indicators_1h.get("volume", 0),
                indicators_1h.get("volume_sma", 1),
            ),
            "support_levels": self.scorer.sr.get_support_prices(symbol),
            "resistance_levels": self.scorer.sr.get_resistance_prices(symbol),
        }

        # Gate 5: AI analysis (if enabled)
        if self.ai and Settings.ai.ENABLED:
            recent_news = self.news.get_top_headlines(5)
            ai_result = await self.ai.analyze_trade(
                symbol=symbol,
                indicators=indicators_1h,
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

            if not approval["approved"]:
                logger.info("AI blocked entry for %s: %s", symbol, approval["reason"])
                self.scorer.db.save_signal({
                    "timestamp": utc_now(), "symbol": symbol,
                    "confluence_score": result["total_score"],
                    "score_breakdown": json.dumps(result["breakdown"]),
                    "action_taken": f"AI_VETOED",
                    "regime": self.regime.current_regime,
                })
                return None

            # AI may suggest tighter SL/TP
            if approval.get("ai_sl"):
                ai_sl = price * (1 - approval["ai_sl"])
                signal["stop_loss"] = max(signal["stop_loss"], ai_sl)
            if approval.get("ai_tp"):
                ai_tp = price * (1 + approval["ai_tp"])
                signal["take_profit"] = min(signal["take_profit"], ai_tp)

        return signal
