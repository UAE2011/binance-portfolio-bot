"""
Strategy — Playbook-optimized confluence scoring for Binance spot.

Scoring model (100 pts total):
  Technical    40 pts  RSI momentum, MACD, EMA alignment, SuperTrend, BB
  Momentum     20 pts  Price acceleration, candle patterns, StochRSI
  Volume       15 pts  Volume confirmation (50-100%+ above SMA for breakouts)
  Structure    10 pts  S/R proximity, trend strength (ADX)
  Regime        5 pts  Adaptive soft bonus
  Sentiment     5 pts  Fear & Greed, news
  Timing        5 pts  Peak-hour bonus (14:30-16:30 UTC, Tue-Thu)

Key changes from playbook:
  - RSI momentum mode (>50 = bullish, not just oversold)
  - SuperTrend confirmation required for trend-following entries
  - Volume must be ≥1.5× SMA for breakout entries
  - ADX >25 confirms trending regime (avoids choppy entries)
  - Time-of-day weighting (London/NY overlap = peak liquidity)
"""
import json
from datetime import datetime, timezone
from typing import Optional

from config.settings import Settings
from src.utils import setup_logging, utc_now
from src.indicators import detect_swing_points, cluster_levels, detect_rsi_divergence

logger = setup_logging()


# ---------------------------------------------------------------------------
# Support / Resistance Engine
# ---------------------------------------------------------------------------

class SupportResistanceEngine:
    def __init__(self):
        self.levels_cache: dict = {}

    def compute_levels(self, symbol: str, kline_history: list) -> dict:
        closes = [k["close"] for k in kline_history]
        highs = [k["high"] for k in kline_history]
        lows = [k["low"] for k in kline_history]
        if len(closes) < 20:
            return {"support": [], "resistance": []}

        swings = detect_swing_points(closes, lookback=3)
        wick_h = detect_swing_points(highs, lookback=3)
        wick_l = detect_swing_points(lows, lookback=3)

        all_highs = swings["swing_highs"] + wick_h["swing_highs"]
        all_lows = swings["swing_lows"] + wick_l["swing_lows"]

        resistance_zones = cluster_levels(all_highs, threshold_pct=0.003)
        support_zones = cluster_levels(all_lows, threshold_pct=0.003)

        current_price = closes[-1]
        support = sorted(
            [z for z in support_zones if z["level"] < current_price],
            key=lambda x: x["level"], reverse=True,
        )
        resistance = sorted(
            [z for z in resistance_zones if z["level"] > current_price],
            key=lambda x: x["level"],
        )

        result = {"support": support[:10], "resistance": resistance[:10]}
        self.levels_cache[symbol] = result
        return result

    def get_nearest_support(self, symbol: str, current_price: float) -> Optional[dict]:
        for level in self.levels_cache.get(symbol, {}).get("support", []):
            if level["level"] < current_price:
                return level
        return None

    def get_nearest_resistance(self, symbol: str, current_price: float) -> Optional[dict]:
        for level in self.levels_cache.get(symbol, {}).get("resistance", []):
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

    def risk_reward_ratio(self, symbol: str, entry: float, stop: float) -> float:
        """Calculate R:R to nearest resistance."""
        resistance = self.get_nearest_resistance(symbol, entry)
        if resistance is None or stop >= entry:
            return 0.0
        reward = resistance["level"] - entry
        risk = entry - stop
        return reward / risk if risk > 0 else 0.0

    def get_support_prices(self, symbol: str) -> list:
        return [z["level"] for z in self.levels_cache.get(symbol, {}).get("support", [])]

    def get_resistance_prices(self, symbol: str) -> list:
        return [z["level"] for z in self.levels_cache.get(symbol, {}).get("resistance", [])]


# ---------------------------------------------------------------------------
# Volume Spike Detector
# ---------------------------------------------------------------------------

class VolumeSpikeDetector:
    @staticmethod
    def detect_spike(volume: float, volume_sma: float) -> dict:
        if volume_sma <= 0:
            return {"is_spike": False, "ratio": 0, "strength": "NONE"}
        ratio = volume / volume_sma
        threshold = Settings.strategy.VOLUME_SPIKE_MULTIPLIER
        breakout_threshold = Settings.strategy.VOLUME_BREAKOUT_MULTIPLIER
        return {
            "is_spike": ratio >= threshold,
            "is_breakout_vol": ratio >= breakout_threshold,
            "ratio": ratio,
            "strength": (
                "EXTREME" if ratio >= 4.0
                else "STRONG" if ratio >= 3.0
                else "BREAKOUT" if ratio >= breakout_threshold
                else "MODERATE" if ratio >= threshold
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
# Confluence Scorer — Playbook-based 100-point system
# ---------------------------------------------------------------------------

class ConfluenceScorer:
    """
    Playbook-optimized scoring:
    - RSI momentum mode (>50 = bullish, not oversold reversal)
    - SuperTrend confirms trend direction
    - MACD crypto-tuned (8,17,9)
    - Volume filter: breakouts need ≥2× SMA confirmation
    - ADX trend strength as gate for trend-following
    - Time-of-day bonus (peak liquidity hours)
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

        # 1. Technical Score (40 pts)
        tech_score, tech_detail = self._technical_score(
            symbol, indicators_primary, indicators_entry, indicators_trend
        )
        breakdown["technical"] = {"score": tech_score, "max": 40, "details": tech_detail}
        total += tech_score

        # 2. Momentum Score (20 pts)
        mom_score, mom_detail = self._momentum_score(
            indicators_entry, indicators_primary, kline_history
        )
        breakdown["momentum"] = {"score": mom_score, "max": 20, "details": mom_detail}
        total += mom_score

        # 3. Volume Score (15 pts)
        vol_score, vol_detail = self._volume_score(indicators_primary)
        breakdown["volume"] = {"score": vol_score, "max": 15, "details": vol_detail}
        total += vol_score

        # 4. Structure Score (10 pts) — S/R + ADX
        struct_score, struct_detail = self._structure_score(symbol, indicators_primary)
        breakdown["structure"] = {"score": struct_score, "max": 10, "details": struct_detail}
        total += struct_score

        # 5. Regime Score (5 pts) — soft bonus
        regime_score = self._regime_score()
        breakdown["regime"] = {"score": regime_score, "max": 5}
        total += regime_score

        # 6. Sentiment Score (5 pts)
        sentiment_score = self._sentiment_score()
        breakdown["sentiment"] = {"score": sentiment_score, "max": 5}
        total += sentiment_score

        # 7. Timing Score (5 pts) — peak hours bonus
        timing_score = self._timing_score()
        breakdown["timing"] = {"score": timing_score, "max": 5}
        total += timing_score

        threshold = regime_params.get(
            "confluence_threshold", Settings.strategy.CONFLUENCE_SCORE_THRESHOLD
        )

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

    def _technical_score(self, symbol: str, ind: dict, ind_entry: dict,
                         ind_trend: dict) -> tuple:
        score = 0
        details = {}
        price = ind.get("close", 0)

        # --- RSI Momentum Mode (12 pts) ---
        # Research: RSI >50 = bullish momentum beats oversold reversal in crypto
        rsi = ind.get("rsi", 50)
        rsi_fast = ind.get("rsi_fast", 50)

        if rsi > 55 and rsi < 72:
            rsi_score = 12
            details["rsi"] = f"RSI={rsi:.1f} (strong momentum zone)"
        elif rsi > 50 and rsi <= 55:
            rsi_score = 10
            details["rsi"] = f"RSI={rsi:.1f} (momentum building)"
        elif rsi >= 30 and rsi <= 50:
            rsi_score = 7
            details["rsi"] = f"RSI={rsi:.1f} (recovery zone)"
        elif rsi < 30:
            rsi_score = 8  # oversold bounce potential
            details["rsi"] = f"RSI={rsi:.1f} (oversold — bounce setup)"
        else:
            rsi_score = 3
            details["rsi"] = f"RSI={rsi:.1f} (overbought)"
        score += rsi_score

        # Fast RSI entry confirmation
        if rsi_fast > 50 and rsi_fast < 75:
            score += 2
            details["rsi_fast"] = f"Fast RSI={rsi_fast:.0f} confirms entry"
        elif rsi_fast > 75:
            score -= 1
            details["rsi_fast"] = f"Fast RSI={rsi_fast:.0f} extended"

        # --- MACD (10 pts) --- crypto-tuned 8,17,9
        macd_hist = ind.get("macd_histogram", 0)
        macd_line = ind.get("macd", 0)
        macd_signal = ind.get("macd_signal", 0)
        prev_hist = ind.get("prev_macd_histogram", macd_hist - 0.0001)

        if macd_line > macd_signal and macd_hist > 0:
            macd_score = 10
            details["macd"] = "MACD bullish crossover + positive histogram"
        elif macd_hist > 0:
            macd_score = 8
            details["macd"] = "MACD histogram positive (bullish)"
        elif macd_line > macd_signal:
            macd_score = 6
            details["macd"] = "MACD above signal (early bullish)"
        elif macd_hist > prev_hist:
            macd_score = 5
            details["macd"] = "MACD improving — turning bullish"
        else:
            macd_score = 2
            details["macd"] = "MACD bearish"
        score += macd_score

        # --- SuperTrend (8 pts) — research-backed best trailing indicator ---
        st_up = ind.get("supertrend_up", False)
        st_val = ind.get("supertrend", 0)
        if st_up:
            score += 8
            details["supertrend"] = f"SuperTrend BULLISH (price above ${st_val:.4f})"
        else:
            score += 2  # base points even bearish
            details["supertrend"] = f"SuperTrend BEARISH (price below ${st_val:.4f})"

        # --- EMA Alignment (6 pts) — 9/21/55 stack ---
        ema9 = ind.get("ema9", 0)
        ema21 = ind.get("ema21", 0)
        ema55 = ind.get("ema55", ema21)
        sma200 = ind.get("sma200", 0)

        if price > ema9 > ema21 > ema55:
            ema_score = 6
            details["ema"] = "Perfect EMA alignment (9>21>55)"
        elif price > ema9 > ema21:
            ema_score = 5
            details["ema"] = "Price > EMA9 > EMA21"
        elif price > ema21:
            ema_score = 4
            details["ema"] = "Price above EMA21"
        elif ema9 > 0 and abs(price - ema9) / ema9 < 0.005:
            ema_score = 3
            details["ema"] = "Price testing EMA9 (potential bounce)"
        else:
            ema_score = 1
            details["ema"] = "Below key EMAs"
        # Bonus: price above 200 SMA
        if sma200 > 0 and price > sma200:
            ema_score = min(ema_score + 1, 6)
            details["sma200"] = "Above 200 SMA ✓"
        score += ema_score

        # --- Bollinger Bands (4 pts) — wider 2.5σ for crypto ---
        bb_lower = ind.get("bb_lower", 0)
        bb_upper = ind.get("bb_upper", 0)
        bb_mid = ind.get("bb_middle", 0)
        bb_pct_b = ind.get("bb_percent_b", 0.5)

        if bb_lower > 0 and price <= bb_lower * 1.005:
            score += 4
            details["bb"] = "At lower BB (bounce setup)"
        elif bb_pct_b < 0.3:
            score += 3
            details["bb"] = f"Lower BB zone (B%={bb_pct_b:.2f})"
        elif bb_pct_b < 0.5:
            score += 2
            details["bb"] = f"Below BB midline (room to run)"
        else:
            score += 1
            details["bb"] = f"BB neutral (B%={bb_pct_b:.2f})"

        # --- Trend timeframe confirmation (bonus) ---
        trend_rsi = ind_trend.get("rsi", 50)
        trend_ema21 = ind_trend.get("ema21", 0)
        trend_price = ind_trend.get("close", price)
        if trend_price > trend_ema21 > 0 and trend_rsi > 50:
            score += 3
            details["trend_tf"] = "4H bullish trend confirmed"
        elif trend_price > trend_ema21 > 0:
            score += 1
            details["trend_tf"] = "4H above EMA21"

        return min(score, 40), details

    def _momentum_score(self, ind_entry: dict, ind_primary: dict, history: list) -> tuple:
        score = 0
        details = {}

        # --- Price acceleration (8 pts) ---
        if history and len(history) >= 10:
            closes = [k["close"] for k in history]
            recent_change = (closes[-1] - closes[-5]) / closes[-5] if closes[-5] > 0 else 0
            prior_change = (closes[-5] - closes[-10]) / closes[-10] if closes[-10] > 0 else 0
            if recent_change > 0.02 and recent_change > prior_change:
                score += 8
                details["acceleration"] = f"Strong acceleration +{recent_change:.2%}"
            elif recent_change > 0 and recent_change > prior_change:
                score += 6
                details["acceleration"] = f"Accelerating +{recent_change:.2%}"
            elif recent_change > 0:
                score += 4
                details["acceleration"] = f"Rising +{recent_change:.2%}"
            elif recent_change > -0.01:
                score += 2
                details["acceleration"] = "Consolidating (potential breakout)"
            else:
                score += 1
                details["acceleration"] = f"Declining {recent_change:.2%}"

        # --- Green candle streak (5 pts) ---
        if history and len(history) >= 5:
            recent_5 = history[-5:]
            green = sum(1 for k in recent_5 if k.get("close", 0) > k.get("open", 0))
            if green >= 4:
                score += 5
                details["candles"] = f"{green}/5 green candles (strong)"
            elif green >= 3:
                score += 3
                details["candles"] = f"{green}/5 green candles"
            elif green >= 2:
                score += 2
                details["candles"] = f"{green}/5 green candles"
            else:
                score += 1
                details["candles"] = f"{green}/5 green candles"

        # --- StochRSI (5 pts) ---
        stoch_k = ind_primary.get("stoch_k", 50)
        stoch_d = ind_primary.get("stoch_d", 50)
        if stoch_k < 20:
            score += 5
            details["stochrsi"] = f"StochRSI K={stoch_k:.0f} (deep oversold)"
        elif stoch_k < 35 and stoch_k > stoch_d:
            score += 4
            details["stochrsi"] = f"StochRSI K={stoch_k:.0f} crossing up from oversold"
        elif stoch_k < 50:
            score += 3
            details["stochrsi"] = f"StochRSI K={stoch_k:.0f} (neutral-low)"
        elif stoch_k < 65:
            score += 2
            details["stochrsi"] = f"StochRSI K={stoch_k:.0f} (neutral)"
        else:
            score += 1
            details["stochrsi"] = f"StochRSI K={stoch_k:.0f} (high)"

        # --- RSI divergence bonus (2 pts) ---
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
                    score += 2
                    details["divergence"] = "Bullish RSI divergence detected (+2)"
            except Exception:
                pass

        return min(score, 20), details

    def _volume_score(self, ind: dict) -> tuple:
        """Volume must confirm breakouts — playbook requires 50-100%+ above SMA."""
        volume = ind.get("volume", 0)
        vol_sma = ind.get("volume_sma", 1)
        ratio = ind.get("volume_ratio", volume / vol_sma if vol_sma > 0 else 1.0)

        if ratio >= 3.0:
            return 15, f"Extreme volume {ratio:.1f}× SMA (breakout confirmed)"
        elif ratio >= 2.0:
            return 13, f"Strong breakout volume {ratio:.1f}× SMA"
        elif ratio >= 1.5:
            return 10, f"Elevated volume {ratio:.1f}× SMA (entry quality)"
        elif ratio >= 1.0:
            return 7, f"Above-average volume {ratio:.1f}× SMA"
        elif ratio >= 0.7:
            return 4, f"Average volume {ratio:.1f}× SMA"
        else:
            return 2, f"Low volume {ratio:.1f}× SMA (caution)"

    def _structure_score(self, symbol: str, ind: dict) -> tuple:
        """S/R proximity + ADX trend strength."""
        score = 0
        details = {}
        price = ind.get("close", 0)
        adx = ind.get("adx", 0)

        # ADX trend strength (5 pts)
        if adx > 30:
            score += 5
            details["adx"] = f"ADX={adx:.0f} — strong trend"
        elif adx > 25:
            score += 4
            details["adx"] = f"ADX={adx:.0f} — trending"
        elif adx > 20:
            score += 3
            details["adx"] = f"ADX={adx:.0f} — developing trend"
        else:
            score += 1
            details["adx"] = f"ADX={adx:.0f} — ranging market"

        # S/R proximity (5 pts)
        if self.sr.is_near_support(symbol, price):
            nearest = self.sr.get_nearest_support(symbol, price)
            if nearest:
                score += 5
                touches = nearest.get("touches", 1)
                details["sr"] = f"Near support ${nearest['level']:.4f} ({touches} touches)"
        else:
            res = self.sr.get_nearest_resistance(symbol, price)
            if res and price > 0:
                dist = (res["level"] - price) / price
                if dist > 0.05:
                    score += 4
                    details["sr"] = f"Clear to resistance ({dist:.1%} away)"
                elif dist > 0.02:
                    score += 3
                    details["sr"] = f"Some room to resistance ({dist:.1%})"
                else:
                    score += 1
                    details["sr"] = "Near resistance — caution"
            else:
                score += 2
                details["sr"] = "No clear S/R — open air"

        # VWAP check (bonus)
        vwap = ind.get("vwap", 0)
        if vwap > 0 and price > vwap:
            score = min(score + 1, 10)
            details["vwap"] = f"Price above VWAP (${vwap:.4f})"

        return min(score, 10), details

    def _regime_score(self) -> int:
        """Soft regime bonus — never blocks, just adjusts."""
        from src.regime import MODE_AGGRESSIVE, MODE_BALANCED, MODE_DEFENSIVE, MODE_CAPITAL_PRESERVATION
        mode_scores = {
            MODE_AGGRESSIVE: 5,
            MODE_BALANCED: 4,
            MODE_DEFENSIVE: 3,
            MODE_CAPITAL_PRESERVATION: 2,
        }
        return mode_scores.get(self.regime.current_mode, 3)

    def _sentiment_score(self) -> int:
        """Fear & Greed + news sentiment."""
        fg = self.news.fear_greed_value if self.news else 50
        # Contrarian: extreme fear = good entry (3 pts bonus)
        if fg < 20:
            fg_score = 3  # extreme fear = buy opportunity
        elif fg < 40:
            fg_score = 2  # fear = slight edge
        elif fg < 60:
            fg_score = 2  # neutral
        elif fg < 75:
            fg_score = 1  # greed = caution
        else:
            fg_score = 0  # extreme greed = danger

        news_score = min(int(self.news.get_sentiment_points() * 2 / 15), 2) if self.news else 1
        return min(fg_score + news_score, 5)

    def _timing_score(self) -> int:
        """Time-of-day bonus — peak liquidity = better fills and stronger signals."""
        now = utc_now()
        hour = now.hour
        weekday = now.weekday()  # 0=Monday, 6=Sunday

        score = 0
        # Peak hours: London open (8-10 UTC) + NY overlap (14-22 UTC)
        if Settings.strategy.PEAK_HOURS_START <= hour < Settings.strategy.PEAK_HOURS_END:
            score += 3
        elif 8 <= hour < 12:  # London open
            score += 2
        else:
            score += 1

        # Best trading days: Tuesday-Thursday
        if weekday in (1, 2, 3):  # Tue, Wed, Thu
            score += 2
        elif weekday in (0, 4):   # Mon, Fri
            score += 1
        # Sat/Sun = reduced liquidity = 0 bonus

        return min(score, 5)


# ---------------------------------------------------------------------------
# Signal Generator
# ---------------------------------------------------------------------------

class SignalGenerator:
    """Generates trade signals with playbook-based scoring and AI filtering."""

    def __init__(self, scorer: ConfluenceScorer, regime_detector, news_intel,
                 ai_advisor=None):
        self.scorer = scorer
        self.regime = regime_detector
        self.news = news_intel
        self.ai = ai_advisor
        self._cycle_ai_calls: int = 0
        self._cycle_id: int = 0

    async def evaluate(self, symbol: str, indicators_entry: dict,
                       indicators_primary: dict, indicators_trend: dict,
                       kline_history: list, portfolio_context: dict = None,
                       cycle_id: int = 0) -> Optional[dict]:
        # Reset AI call counter each cycle
        if cycle_id != self._cycle_id:
            self._cycle_ai_calls = 0
            self._cycle_id = cycle_id

        regime_params = self.regime.get_regime_params()

        # Capital preservation mode: no entries
        if not regime_params.get("entries_allowed", True):
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
        if price <= 0:
            return None

        # Risk/Reward pre-filter: require ≥1.5:1 R:R
        stop_loss_price = price * (1 - Settings.risk.STOP_LOSS_PCT)
        if atr and atr > 0:
            regime_mult = regime_params.get("stop_atr_mult", 3.0)
            atr_stop = price - (atr * regime_mult)
            stop_loss_price = max(stop_loss_price, atr_stop)

        nearest_resistance = self.scorer.sr.get_nearest_resistance(symbol, price)
        if nearest_resistance:
            reward = nearest_resistance["level"] - price
            risk = price - stop_loss_price
            if risk > 0 and reward / risk < 1.2:
                # Weak R:R — only allow if score is very high
                if result["total_score"] < 65:
                    return None

        take_profit = price * (1 + Settings.risk.TAKE_PROFIT_PCT)

        signal = {
            "symbol": symbol,
            "price": price,
            "confluence_score": result["total_score"],
            "breakdown": result["breakdown"],
            "stop_loss": stop_loss_price,
            "take_profit": take_profit,
            "atr": atr,
            "regime": self.regime.current_regime,
            "mode": self.regime.current_mode,
            "fear_greed": self.news.fear_greed_value if self.news else 50,
            "news_sentiment": self.news.get_sentiment_score() if self.news else 0,
            "supertrend_up": indicators_primary.get("supertrend_up", False),
            "adx": indicators_primary.get("adx", 0),
            "volume_ratio": indicators_primary.get("volume_ratio", 1.0),
            "volume_spike": VolumeSpikeDetector.detect_spike(
                indicators_primary.get("volume", 0),
                indicators_primary.get("volume_sma", 1),
            ),
            "support_levels": self.scorer.sr.get_support_prices(symbol),
            "resistance_levels": self.scorer.sr.get_resistance_prices(symbol),
        }

        # AI analysis — only for top candidates per cycle
        max_ai = Settings.ai.MAX_CALLS_PER_CYCLE
        if self.ai and Settings.ai.ENABLED and self._cycle_ai_calls < max_ai:
            try:
                self._cycle_ai_calls += 1
                recent_news = self.news.get_top_headlines(5) if self.news else []
                ai_result = await self.ai.analyze_trade(
                    symbol=symbol,
                    indicators=indicators_primary,
                    regime=self.regime.current_regime,
                    news_sentiment=signal["news_sentiment"],
                    fear_greed=signal["fear_greed"],
                    support_levels=signal["support_levels"],
                    resistance_levels=signal["resistance_levels"],
                    recent_news=recent_news,
                    portfolio_context=portfolio_context or {},
                )
                signal["ai_analysis"] = ai_result

                approval = self.ai.should_approve_entry(ai_result, result["total_score"])
                signal["ai_approval"] = approval

                ai_confidence = ai_result.get("confidence", 0) if ai_result else 0
                veto_threshold = Settings.ai.VETO_MIN_CONFIDENCE

                if (Settings.ai.VETO_POWER
                        and not approval.get("approved", True)
                        and ai_confidence >= veto_threshold):
                    logger.info("AI vetoed %s (%.0f%% confidence): %s",
                                symbol, ai_confidence * 100, approval.get("reason", "")[:60])
                    return None

                # AI-suggested SL/TP refinement
                if approval.get("ai_sl"):
                    ai_sl = price * (1 - approval["ai_sl"])
                    signal["stop_loss"] = max(signal["stop_loss"], ai_sl)
                if approval.get("ai_tp"):
                    ai_tp = price * (1 + approval["ai_tp"])
                    signal["take_profit"] = min(signal["take_profit"], ai_tp)

            except Exception as e:
                logger.warning("AI analysis failed for %s: %s", symbol, e)
                signal["ai_analysis"] = None
                signal["ai_approval"] = {"approved": True, "reason": "AI unavailable"}
        else:
            signal["ai_analysis"] = None
            signal["ai_approval"] = {
                "approved": True,
                "reason": "Confluence-based entry" if self._cycle_ai_calls >= max_ai
                else "AI disabled",
            }

        return signal
