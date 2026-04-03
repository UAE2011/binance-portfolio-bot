"""
Regime Detector — 4-mode adaptive system based on playbook research.

Modes:
  AGGRESSIVE        — Bull market: full deployment, trend-following
  BALANCED          — Sideways: partial deployment, mean-reversion blend
  DEFENSIVE         — High volatility / early bear: reduced size, tight stops
  CAPITAL_PRESERVATION — Bear market: 70%+ stablecoins, DCA only

Detection uses:
  - HMM (Hidden Markov Model) on BTC daily returns + volatility
  - BTC vs 50/200 SMA (death/golden cross)
  - ADX trend strength
  - BTC Dominance (altcoin season detection)
  - Fear & Greed Index (contrarian extremes)
  - Volume trend analysis
"""
import asyncio
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional

from src.utils import setup_logging, utc_now

logger = setup_logging()

MODE_AGGRESSIVE = "AGGRESSIVE"
MODE_BALANCED = "BALANCED"
MODE_DEFENSIVE = "DEFENSIVE"
MODE_CAPITAL_PRESERVATION = "CAPITAL_PRESERVATION"

# Legacy compatibility aliases
REGIME_BULL = "BULL"
REGIME_BEAR = "BEAR"
REGIME_SIDEWAYS = "SIDEWAYS"
REGIME_HIGH_VOL = "HIGH_VOLATILITY"

MODEL_PATH = Path("data/hmm_model.pkl")


class RegimeDetector:
    def __init__(self, exchange, database, lookback_days: int = 90):
        self.exchange = exchange
        self.db = database
        self.lookback_days = lookback_days
        self.model = None
        self.current_regime: str = REGIME_SIDEWAYS   # legacy compat
        self.current_mode: str = MODE_BALANCED        # new mode system
        self.regime_confidence: float = 0.0
        self.btc_indicators: dict = {}
        self.last_train_time: Optional[datetime] = None
        self.btc_dominance: float = 50.0
        self.altcoin_season_index: int = 50
        self.mode_history: list = []  # track recent mode changes

    async def initialize(self):
        if MODEL_PATH.exists():
            try:
                import joblib
                self.model = joblib.load(MODEL_PATH)
                logger.info("Loaded existing HMM model")
            except Exception as e:
                logger.warning("HMM load failed: %s, retraining", e)
                await self.train_hmm()
        else:
            await self.train_hmm()
        await self.detect_regime()

    async def train_hmm(self):
        logger.info("Training HMM on %d days BTC data...", self.lookback_days)
        try:
            klines = await self.exchange.get_klines(
                "BTCUSDT", interval="1d", limit=self.lookback_days + 20
            )
            if len(klines) < 30:
                return
            closes = np.array([k["close"] for k in klines])
            volumes = np.array([k["volume"] for k in klines])
            log_returns = np.diff(np.log(closes))
            rolling_vol = pd.Series(log_returns).rolling(14).std().values * np.sqrt(365)
            vol_change = pd.Series(volumes[1:]).pct_change(14).values
            valid_start = max(14, np.argmax(~np.isnan(rolling_vol)))
            log_returns = log_returns[valid_start:]
            rolling_vol = rolling_vol[valid_start:]
            vol_change = vol_change[valid_start:]
            mask = ~(np.isnan(rolling_vol) | np.isnan(vol_change) | np.isinf(vol_change))
            log_returns = log_returns[mask]
            rolling_vol = rolling_vol[mask]
            vol_change = np.clip(vol_change[mask], -5, 5)
            if len(log_returns) < 20:
                return
            features = np.column_stack([log_returns, rolling_vol, vol_change])
            from hmmlearn.hmm import GaussianHMM
            self.model = GaussianHMM(
                n_components=4, covariance_type="full",
                n_iter=200, random_state=42, tol=0.01
            )
            self.model.fit(features)
            import joblib
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.model, MODEL_PATH)
            self.last_train_time = utc_now()
            logger.info("HMM trained and saved")
        except Exception as e:
            logger.error("HMM training failed: %s", e)

    def _map_states_to_regimes(self) -> dict:
        if self.model is None:
            return {}
        means = self.model.means_
        state_info = [(i, means[i][0], means[i][1]) for i in range(4)]
        sorted_by_vol = sorted(state_info, key=lambda x: x[2], reverse=True)
        high_vol_state = sorted_by_vol[0][0]
        state_map = {high_vol_state: REGIME_HIGH_VOL}
        remaining = [s for s in state_info if s[0] != high_vol_state]
        sorted_by_return = sorted(remaining, key=lambda x: x[1], reverse=True)
        state_map[sorted_by_return[0][0]] = REGIME_BULL
        state_map[sorted_by_return[2][0]] = REGIME_BEAR
        state_map[sorted_by_return[1][0]] = REGIME_SIDEWAYS
        return state_map

    async def detect_regime(self) -> str:
        """Detect regime using HMM + trend analysis + on-chain proxies."""
        hmm_regime = await self._hmm_regime()
        trend_regime = await self._trend_regime()
        sentiment_regime = self._sentiment_regime()

        # Weighted scoring
        scores = {REGIME_BULL: 0, REGIME_BEAR: 0, REGIME_SIDEWAYS: 0, REGIME_HIGH_VOL: 0}
        if hmm_regime:
            scores[hmm_regime] += 50
        if trend_regime:
            scores[trend_regime] += 35
        if sentiment_regime:
            scores[sentiment_regime] += 15

        self.current_regime = max(scores, key=scores.get)
        total = sum(scores.values())
        self.regime_confidence = scores[self.current_regime] / total if total > 0 else 0.5

        # Map to operating mode
        self.current_mode = self._regime_to_mode(self.current_regime)

        btc_price = await self.exchange.get_price("BTCUSDT")
        self.db.save_regime({
            "timestamp": utc_now(),
            "regime": self.current_regime,
            "confidence": self.regime_confidence,
            "btc_price": btc_price,
            "fear_greed": self.btc_indicators.get("fear_greed"),
            "news_sentiment": self.btc_indicators.get("news_sentiment"),
            "btc_above_50sma": self.btc_indicators.get("above_50sma", False),
            "btc_above_200sma": self.btc_indicators.get("above_200sma", False),
        })
        logger.info("Regime: %s → Mode: %s (confidence: %.0f%%)",
                    self.current_regime, self.current_mode, self.regime_confidence * 100)
        return self.current_regime

    def _regime_to_mode(self, regime: str) -> str:
        """Convert HMM regime to operating mode with nuance."""
        fg = self.btc_indicators.get("fear_greed", 50)
        above_200 = self.btc_indicators.get("above_200sma", True)

        if regime == REGIME_BULL:
            if fg > 75:  # extreme greed in bull = defensive (late stage)
                return MODE_BALANCED
            return MODE_AGGRESSIVE
        elif regime == REGIME_SIDEWAYS:
            if not above_200:  # sideways below 200SMA = bearish bias
                return MODE_DEFENSIVE
            return MODE_BALANCED
        elif regime == REGIME_HIGH_VOL:
            return MODE_DEFENSIVE
        elif regime == REGIME_BEAR:
            if fg < 15:  # extreme fear in bear = capital preservation + DCA
                return MODE_CAPITAL_PRESERVATION
            return MODE_DEFENSIVE
        return MODE_BALANCED

    def _sentiment_regime(self) -> Optional[str]:
        """Use Fear & Greed as regime override at extremes."""
        fg = self.btc_indicators.get("fear_greed", 50)
        if fg > 80:
            return REGIME_BULL   # extreme greed confirms bull
        elif fg < 15:
            return REGIME_BEAR   # extreme fear confirms bear
        elif fg > 60:
            return REGIME_BULL
        elif fg < 35:
            return REGIME_BEAR
        return REGIME_SIDEWAYS

    async def _hmm_regime(self) -> Optional[str]:
        if self.model is None:
            return None
        try:
            klines = await self.exchange.get_klines("BTCUSDT", interval="1d", limit=30)
            if len(klines) < 20:
                return None
            closes = np.array([k["close"] for k in klines])
            volumes = np.array([k["volume"] for k in klines])
            log_returns = np.diff(np.log(closes))
            rolling_vol = pd.Series(log_returns).rolling(14).std().values * np.sqrt(365)
            vol_change = pd.Series(volumes[1:]).pct_change(14).values
            lr = log_returns[-1:]
            rv = rolling_vol[-1:]
            vc = vol_change[-1:]
            if np.isnan(rv[0]) or np.isnan(vc[0]) or np.isinf(vc[0]):
                return None
            vc = np.clip(vc, -5, 5)
            features = np.column_stack([lr, rv, vc])
            state = self.model.predict(features)[0]
            state_map = self._map_states_to_regimes()
            return state_map.get(state, REGIME_SIDEWAYS)
        except Exception as e:
            logger.error("HMM prediction failed: %s", e)
            return None

    async def _trend_regime(self) -> Optional[str]:
        try:
            klines = await self.exchange.get_klines("BTCUSDT", interval="1d", limit=210)
            if len(klines) < 55:
                return None
            closes = [k["close"] for k in klines]
            highs = [k["high"] for k in klines]
            lows = [k["low"] for k in klines]
            volumes = [k["volume"] for k in klines]
            current_price = closes[-1]

            sma50 = np.mean(closes[-50:]) if len(closes) >= 50 else current_price
            sma200 = np.mean(closes[-200:]) if len(closes) >= 200 else current_price

            self.btc_indicators["above_50sma"] = current_price > sma50
            self.btc_indicators["above_200sma"] = current_price > sma200
            self.btc_indicators["sma50"] = sma50
            self.btc_indicators["sma200"] = sma200
            self.btc_indicators["btc_price"] = current_price

            # ADX for trend strength
            from src.indicators import IncrementalADX
            adx_calc = IncrementalADX(14)
            for i in range(len(klines)):
                adx_calc.update(highs[i], lows[i], closes[i])
            adx_value = adx_calc.adx
            self.btc_indicators["adx"] = adx_value

            # Volume trend (increasing = healthy trend)
            avg_vol_20 = np.mean(volumes[-20:])
            avg_vol_60 = np.mean(volumes[-60:]) if len(volumes) >= 60 else avg_vol_20
            vol_trend_up = avg_vol_20 > avg_vol_60
            self.btc_indicators["volume_trend_up"] = vol_trend_up

            # 30-day price momentum
            momentum_30d = (current_price / closes[-30] - 1) if len(closes) >= 30 else 0
            self.btc_indicators["momentum_30d"] = momentum_30d

            # Classify
            if current_price > sma50 > sma200:
                if adx_value > 25:
                    return REGIME_BULL
                return REGIME_SIDEWAYS
            elif current_price < sma50 < sma200:
                if adx_value > 25:
                    return REGIME_BEAR
                return REGIME_SIDEWAYS
            elif current_price > sma200:
                return REGIME_SIDEWAYS
            else:
                return REGIME_BEAR
        except Exception as e:
            logger.error("Trend regime failed: %s", e)
            return None

    def update_sentiment(self, fear_greed: int = None, news_sentiment: float = None,
                        btc_dominance: float = None, altcoin_season_index: int = None):
        if fear_greed is not None:
            self.btc_indicators["fear_greed"] = fear_greed
        if news_sentiment is not None:
            self.btc_indicators["news_sentiment"] = news_sentiment
        if btc_dominance is not None:
            self.btc_dominance = btc_dominance
            self.btc_indicators["btc_dominance"] = btc_dominance
        if altcoin_season_index is not None:
            self.altcoin_season_index = altcoin_season_index

    async def should_retrain(self) -> bool:
        if self.last_train_time is None:
            return True
        return (utc_now() - self.last_train_time).days >= 7

    def is_altcoin_season(self) -> bool:
        """BTC dominance below 50% = altcoin season."""
        from config.settings import Settings
        return (self.btc_dominance < Settings.portfolio_cfg.BTC_DOM_ALTSEASON_THRESHOLD
                and self.altcoin_season_index >= Settings.portfolio_cfg.ALTSEASON_INDEX_ENTRY)

    def get_regime_params(self) -> dict:
        """
        Regime parameters — adaptive per mode.
        AGGRESSIVE: maximum deployment, trend-following entries
        BALANCED:   moderate deployment, mixed strategies
        DEFENSIVE:  reduced sizes, tighter stops, fewer entries
        CAPITAL_PRESERVATION: minimal deployment, DCA only, preserve cash
        """
        params_by_mode = {
            MODE_AGGRESSIVE: {
                "entries_allowed": True,
                "position_multiplier": 1.0,
                "stop_atr_mult": 3.5,
                "tp_atr_mult": 5.0,
                "max_exposure": 0.85,
                "confluence_threshold": 38,
                "cash_reserve": 0.15,
                "news_interval_min": 15,
                "dca_multiplier": 1.0,
                "prefer_trend_following": True,
            },
            MODE_BALANCED: {
                "entries_allowed": True,
                "position_multiplier": 0.75,
                "stop_atr_mult": 3.0,
                "tp_atr_mult": 4.0,
                "max_exposure": 0.60,
                "confluence_threshold": 45,
                "cash_reserve": 0.40,
                "news_interval_min": 10,
                "dca_multiplier": 1.5,
                "prefer_trend_following": False,
            },
            MODE_DEFENSIVE: {
                "entries_allowed": True,
                "position_multiplier": 0.40,
                "stop_atr_mult": 2.0,
                "tp_atr_mult": 2.5,
                "max_exposure": 0.35,
                "confluence_threshold": 55,
                "cash_reserve": 0.65,
                "news_interval_min": 5,
                "dca_multiplier": 2.0,
                "prefer_trend_following": False,
            },
            MODE_CAPITAL_PRESERVATION: {
                "entries_allowed": False,   # no new entries, DCA only
                "position_multiplier": 0.15,
                "stop_atr_mult": 1.5,
                "tp_atr_mult": 2.0,
                "max_exposure": 0.20,
                "confluence_threshold": 65,
                "cash_reserve": 0.80,
                "news_interval_min": 5,
                "dca_multiplier": 3.0,      # 3× DCA during extreme fear
                "prefer_trend_following": False,
            },
        }
        return params_by_mode.get(self.current_mode, params_by_mode[MODE_BALANCED])

    def get_mode_emoji(self) -> str:
        return {
            MODE_AGGRESSIVE: "🚀",
            MODE_BALANCED: "⚖️",
            MODE_DEFENSIVE: "🛡️",
            MODE_CAPITAL_PRESERVATION: "🏦",
        }.get(self.current_mode, "⚖️")

    def get_altcoin_allocation_advice(self) -> str:
        """Return sector rotation advice based on BTC dominance."""
        if self.btc_dominance > 65:
            return "BTC_FOCUS"       # 80-90% BTC/ETH, avoid alts
        elif self.btc_dominance < 50:
            return "ALTCOIN_SEASON"  # rotate into quality alts
        return "NEUTRAL"             # balanced approach
