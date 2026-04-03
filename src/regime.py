import asyncio
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional

from src.utils import setup_logging, utc_now

logger = setup_logging()

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
        self.current_regime: str = REGIME_SIDEWAYS
        self.regime_confidence: float = 0.0
        self.btc_indicators: dict = {}
        self.last_train_time: Optional[datetime] = None

    async def initialize(self):
        if MODEL_PATH.exists():
            try:
                import joblib
                self.model = joblib.load(MODEL_PATH)
                logger.info("Loaded existing HMM model from disk")
            except Exception as e:
                logger.warning("Failed to load HMM model: %s, will retrain", e)
                await self.train_hmm()
        else:
            await self.train_hmm()
        await self.detect_regime()

    async def train_hmm(self):
        logger.info("Training HMM on last %d days of BTC data...", self.lookback_days)
        try:
            klines = await self.exchange.get_klines(
                "BTCUSDT", interval="1d", limit=self.lookback_days + 20
            )
            if len(klines) < 30:
                logger.error("Insufficient BTC data for HMM training (%d candles)", len(klines))
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
                logger.error("Not enough valid data points for HMM after cleaning")
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
            logger.info("HMM trained and saved successfully")

        except Exception as e:
            logger.error("HMM training failed: %s", e)

    def _map_states_to_regimes(self) -> dict:
        if self.model is None:
            return {}
        means = self.model.means_
        state_map = {}
        state_info = []
        for i in range(4):
            mean_return = means[i][0]
            mean_vol = means[i][1]
            state_info.append((i, mean_return, mean_vol))

        sorted_by_vol = sorted(state_info, key=lambda x: x[2], reverse=True)
        high_vol_state = sorted_by_vol[0][0]
        state_map[high_vol_state] = REGIME_HIGH_VOL

        remaining = [s for s in state_info if s[0] != high_vol_state]
        sorted_by_return = sorted(remaining, key=lambda x: x[1], reverse=True)

        state_map[sorted_by_return[0][0]] = REGIME_BULL
        state_map[sorted_by_return[2][0]] = REGIME_BEAR
        state_map[sorted_by_return[1][0]] = REGIME_SIDEWAYS

        return state_map

    async def detect_regime(self) -> str:
        hmm_regime = await self._hmm_regime()
        trend_regime = await self._trend_regime()

        regime_scores = {REGIME_BULL: 0, REGIME_BEAR: 0, REGIME_SIDEWAYS: 0, REGIME_HIGH_VOL: 0}

        if hmm_regime:
            regime_scores[hmm_regime] += 60

        if trend_regime:
            regime_scores[trend_regime] += 40

        self.current_regime = max(regime_scores, key=regime_scores.get)

        if regime_scores[REGIME_BULL] > 0 and regime_scores[REGIME_SIDEWAYS] > 0:
            if regime_scores[REGIME_SIDEWAYS] >= regime_scores[REGIME_BULL] * 0.6:
                self.current_regime = REGIME_SIDEWAYS

        total = sum(regime_scores.values())
        self.regime_confidence = regime_scores[self.current_regime] / total if total > 0 else 0.5

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

        logger.info("Regime detected: %s (confidence: %.1f%%)",
                     self.current_regime, self.regime_confidence * 100)
        return self.current_regime

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
            if len(klines) < 200:
                return None

            closes = [k["close"] for k in klines]
            highs = [k["high"] for k in klines]
            lows = [k["low"] for k in klines]
            current_price = closes[-1]

            sma50 = np.mean(closes[-50:])
            sma200 = np.mean(closes[-200:])
            prev_sma50 = np.mean(closes[-51:-1])
            prev_sma200 = np.mean(closes[-201:-1])

            self.btc_indicators["above_50sma"] = current_price > sma50
            self.btc_indicators["above_200sma"] = current_price > sma200
            self.btc_indicators["sma50"] = sma50
            self.btc_indicators["sma200"] = sma200

            golden_cross = prev_sma50 <= prev_sma200 and sma50 > sma200
            death_cross = prev_sma50 >= prev_sma200 and sma50 < sma200

            from src.indicators import IncrementalADX
            adx_calc = IncrementalADX(14)
            for i in range(len(klines)):
                adx_calc.update(highs[i], lows[i], closes[i])
            adx_value = adx_calc.adx
            self.btc_indicators["adx"] = adx_value

            if golden_cross or (current_price > sma50 and current_price > sma200):
                if adx_value > 25:
                    return REGIME_BULL
                return REGIME_SIDEWAYS
            elif death_cross or (current_price < sma50 and current_price < sma200):
                if adx_value > 25:
                    return REGIME_BEAR
                return REGIME_SIDEWAYS
            else:
                if adx_value < 20:
                    return REGIME_SIDEWAYS
                return REGIME_SIDEWAYS

        except Exception as e:
            logger.error("Trend regime detection failed: %s", e)
            return None

    def update_sentiment(self, fear_greed: int = None, news_sentiment: float = None):
        if fear_greed is not None:
            self.btc_indicators["fear_greed"] = fear_greed
        if news_sentiment is not None:
            self.btc_indicators["news_sentiment"] = news_sentiment

    async def should_retrain(self) -> bool:
        now = utc_now()
        if self.last_train_time is None:
            return True
        days_since = (now - self.last_train_time).days
        return days_since >= 7

    def get_regime_params(self) -> dict:
        """Regime adjusts position sizing and risk, but NEVER blocks entries.
        The bot always trades — regime only controls how aggressively."""
        params = {
            REGIME_BULL: {
                "entries_allowed": True, "position_multiplier": 1.0,
                "stop_atr_mult": 3.0, "tp_atr_mult": 4.5,
                "max_exposure": 0.80, "confluence_threshold": 38,
                "news_interval_min": 15,
            },
            REGIME_SIDEWAYS: {
                "entries_allowed": True, "position_multiplier": 0.75,
                "stop_atr_mult": 2.0, "tp_atr_mult": 2.5,
                "max_exposure": 0.65, "confluence_threshold": 45,
                "news_interval_min": 10,
            },
            REGIME_HIGH_VOL: {
                "entries_allowed": True, "position_multiplier": 0.5,
                "stop_atr_mult": 1.5, "tp_atr_mult": 2.0,
                "max_exposure": 0.45, "confluence_threshold": 52,
                "news_interval_min": 5,
            },
            REGIME_BEAR: {
                "entries_allowed": True, "position_multiplier": 0.3,
                "stop_atr_mult": 1.0, "tp_atr_mult": 1.5,
                "max_exposure": 0.30, "confluence_threshold": 58,
                "news_interval_min": 5,
            },
        }
        return params.get(self.current_regime, params[REGIME_SIDEWAYS])
