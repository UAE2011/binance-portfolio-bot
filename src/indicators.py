"""
Indicators — Playbook-optimized incremental indicators for crypto trading.
Includes: RSI (momentum mode), MACD (8,17,9), SuperTrend, Chandelier Exit,
Bollinger Bands, EMA/SMA, ATR, ADX, StochRSI, OBV, Volume SMA, VWAP.
"""
import numpy as np
import pandas as pd
from src.utils import setup_logging

logger = setup_logging()


class IncrementalEMA:
    def __init__(self, period: int):
        self.period = period
        self.k = 2.0 / (period + 1)
        self.value: float = None

    def update(self, price: float) -> float:
        if self.value is None:
            self.value = price
        else:
            self.value = price * self.k + self.value * (1.0 - self.k)
        return self.value

    def seed(self, prices: list):
        self.value = None
        for p in prices:
            self.update(p)


class IncrementalSMA:
    def __init__(self, period: int):
        self.period = period
        self.window: list = []
        self.value: float = None

    def update(self, price: float) -> float:
        self.window.append(price)
        if len(self.window) > self.period:
            self.window.pop(0)
        self.value = sum(self.window) / len(self.window)
        return self.value

    def seed(self, prices: list):
        self.window = []
        self.value = None
        for p in prices:
            self.update(p)


class IncrementalRSI:
    """RSI in momentum mode: >50 = bullish, <50 = bearish (research-backed for crypto)."""
    def __init__(self, period: int = 14):
        self.period = period
        self.avg_gain: float = None
        self.avg_loss: float = None
        self.prev_close: float = None
        self.value: float = 50.0

    def update(self, close: float) -> float:
        if self.prev_close is None:
            self.prev_close = close
            return self.value
        change = close - self.prev_close
        gain = max(change, 0)
        loss = abs(min(change, 0))
        self.prev_close = close
        if self.avg_gain is None:
            self.avg_gain = gain
            self.avg_loss = loss
        else:
            self.avg_gain = (self.avg_gain * (self.period - 1) + gain) / self.period
            self.avg_loss = (self.avg_loss * (self.period - 1) + loss) / self.period
        if self.avg_loss == 0:
            self.value = 100.0
        else:
            rs = self.avg_gain / self.avg_loss
            self.value = 100.0 - (100.0 / (1.0 + rs))
        return self.value

    def seed(self, prices: list):
        self.avg_gain = None
        self.avg_loss = None
        self.prev_close = None
        for p in prices:
            self.update(p)

    @property
    def is_bullish_momentum(self) -> bool:
        """Momentum mode: above 50 = bullish momentum (research-backed vs. oversold reversal)."""
        return self.value > 50

    @property
    def is_oversold(self) -> bool:
        return self.value < 30

    @property
    def is_overbought(self) -> bool:
        return self.value > 70


class IncrementalATR:
    def __init__(self, period: int = 14):
        self.period = period
        self.value: float = None
        self.prev_close: float = None

    def update(self, high: float, low: float, close: float) -> float:
        if self.prev_close is None:
            tr = high - low
        else:
            tr = max(high - low, abs(high - self.prev_close), abs(low - self.prev_close))
        self.prev_close = close
        if self.value is None:
            self.value = tr
        else:
            self.value = (self.value * (self.period - 1) + tr) / self.period
        return self.value

    def seed(self, candles: list):
        self.value = None
        self.prev_close = None
        for c in candles:
            self.update(c["high"], c["low"], c["close"])


class IncrementalMACD:
    """Crypto-optimized MACD: 8,17,9 for intraday (vs standard 12,26,9)."""
    def __init__(self, fast: int = 8, slow: int = 17, signal: int = 9):
        self.fast_ema = IncrementalEMA(fast)
        self.slow_ema = IncrementalEMA(slow)
        self.signal_ema = IncrementalEMA(signal)
        self.macd_line: float = 0.0
        self.signal_line: float = 0.0
        self.histogram: float = 0.0
        self.prev_histogram: float = 0.0

    def update(self, price: float) -> dict:
        self.prev_histogram = self.histogram
        fast_val = self.fast_ema.update(price)
        slow_val = self.slow_ema.update(price)
        self.macd_line = fast_val - slow_val
        self.signal_line = self.signal_ema.update(self.macd_line)
        self.histogram = self.macd_line - self.signal_line
        return {
            "macd": self.macd_line,
            "signal": self.signal_line,
            "histogram": self.histogram,
            "prev_histogram": self.prev_histogram,
        }

    def seed(self, prices: list):
        for p in prices:
            self.fast_ema.update(p)
            self.slow_ema.update(p)
        self.macd_line = self.fast_ema.value - self.slow_ema.value
        self.signal_line = self.signal_ema.update(self.macd_line)
        self.histogram = self.macd_line - self.signal_line
        self.prev_histogram = self.histogram

    @property
    def is_bullish(self) -> bool:
        return self.macd_line > self.signal_line and self.histogram > 0

    @property
    def is_improving(self) -> bool:
        return self.histogram > self.prev_histogram


class IncrementalBollinger:
    def __init__(self, period: int = 20, std_dev: float = 2.5):
        """Wider bands for crypto: 2.5 std dev vs traditional 2.0."""
        self.period = period
        self.std_dev = std_dev
        self.window: list = []
        self.upper: float = 0.0
        self.middle: float = 0.0
        self.lower: float = 0.0
        self.width: float = 0.0
        self.percent_b: float = 0.5

    def update(self, price: float) -> dict:
        self.window.append(price)
        if len(self.window) > self.period:
            self.window.pop(0)
        if len(self.window) >= 2:
            self.middle = sum(self.window) / len(self.window)
            std = (sum((x - self.middle) ** 2 for x in self.window) / len(self.window)) ** 0.5
            self.upper = self.middle + self.std_dev * std
            self.lower = self.middle - self.std_dev * std
            self.width = (self.upper - self.lower) / self.middle if self.middle > 0 else 0
            rng = self.upper - self.lower
            self.percent_b = (price - self.lower) / rng if rng > 0 else 0.5
        return {"upper": self.upper, "middle": self.middle, "lower": self.lower,
                "width": self.width, "percent_b": self.percent_b}

    def seed(self, prices: list):
        self.window = []
        for p in prices:
            self.update(p)


class IncrementalADX:
    def __init__(self, period: int = 14):
        self.period = period
        self.adx: float = 0.0
        self.plus_di: float = 0.0
        self.minus_di: float = 0.0
        self._prev_high: float = None
        self._prev_low: float = None
        self._prev_close: float = None
        self._tr_list: list = []
        self._plus_dm_list: list = []
        self._minus_dm_list: list = []
        self._smoothed_tr: float = None
        self._smoothed_plus_dm: float = None
        self._smoothed_minus_dm: float = None
        self._dx_list: list = []

    def update(self, high: float, low: float, close: float) -> float:
        if self._prev_high is None:
            self._prev_high = high
            self._prev_low = low
            self._prev_close = close
            return self.adx

        tr = max(high - low, abs(high - self._prev_close), abs(low - self._prev_close))
        plus_dm = high - self._prev_high if (high - self._prev_high) > (self._prev_low - low) else 0
        minus_dm = self._prev_low - low if (self._prev_low - low) > (high - self._prev_high) else 0
        plus_dm = max(plus_dm, 0)
        minus_dm = max(minus_dm, 0)

        self._prev_high = high
        self._prev_low = low
        self._prev_close = close

        self._tr_list.append(tr)
        self._plus_dm_list.append(plus_dm)
        self._minus_dm_list.append(minus_dm)

        if len(self._tr_list) < self.period:
            return self.adx

        if self._smoothed_tr is None:
            self._smoothed_tr = sum(self._tr_list[-self.period:])
            self._smoothed_plus_dm = sum(self._plus_dm_list[-self.period:])
            self._smoothed_minus_dm = sum(self._minus_dm_list[-self.period:])
        else:
            self._smoothed_tr = self._smoothed_tr - (self._smoothed_tr / self.period) + tr
            self._smoothed_plus_dm = self._smoothed_plus_dm - (self._smoothed_plus_dm / self.period) + plus_dm
            self._smoothed_minus_dm = self._smoothed_minus_dm - (self._smoothed_minus_dm / self.period) + minus_dm

        self.plus_di = 100 * self._smoothed_plus_dm / self._smoothed_tr if self._smoothed_tr > 0 else 0
        self.minus_di = 100 * self._smoothed_minus_dm / self._smoothed_tr if self._smoothed_tr > 0 else 0

        di_sum = self.plus_di + self.minus_di
        di_diff = abs(self.plus_di - self.minus_di)
        dx = 100 * di_diff / di_sum if di_sum > 0 else 0

        self._dx_list.append(dx)
        if len(self._dx_list) > self.period:
            self._dx_list.pop(0)

        if len(self._dx_list) >= self.period:
            if self.adx == 0:
                self.adx = sum(self._dx_list) / self.period
            else:
                self.adx = (self.adx * (self.period - 1) + dx) / self.period
        return self.adx

    def seed(self, candles: list):
        self._prev_high = None
        self._prev_low = None
        self._prev_close = None
        self._tr_list = []
        self._plus_dm_list = []
        self._minus_dm_list = []
        self._smoothed_tr = None
        self._smoothed_plus_dm = None
        self._smoothed_minus_dm = None
        self._dx_list = []
        self.adx = 0.0
        for c in candles:
            self.update(c["high"], c["low"], c["close"])

    @property
    def is_trending(self) -> bool:
        return self.adx > 25

    @property
    def is_ranging(self) -> bool:
        return self.adx < 20


class IncrementalSuperTrend:
    """SuperTrend indicator — best trailing stop for crypto.
    ATR 14, multiplier 3.5 (research-backed optimal for crypto volatility).
    Signals trend direction with dynamic support/resistance.
    """
    def __init__(self, atr_period: int = 14, multiplier: float = 3.5):
        self.atr_period = atr_period
        self.multiplier = multiplier
        self._atr = IncrementalATR(atr_period)
        self.upper_band: float = 0.0
        self.lower_band: float = 0.0
        self.supertrend: float = 0.0
        self.is_uptrend: bool = True
        self._prev_close: float = None
        self._prev_upper: float = 0.0
        self._prev_lower: float = 0.0
        self._prev_uptrend: bool = True

    def update(self, high: float, low: float, close: float) -> dict:
        atr = self._atr.update(high, low, close)
        if atr is None or atr == 0:
            self._prev_close = close
            return {"supertrend": 0, "is_uptrend": True,
                    "upper_band": 0, "lower_band": 0}

        mid = (high + low) / 2
        basic_upper = mid + self.multiplier * atr
        basic_lower = mid - self.multiplier * atr

        # Adjust bands (never move against trend)
        if self._prev_close is not None:
            if basic_upper < self._prev_upper or self._prev_close > self._prev_upper:
                self.upper_band = basic_upper
            else:
                self.upper_band = self._prev_upper

            if basic_lower > self._prev_lower or self._prev_close < self._prev_lower:
                self.lower_band = basic_lower
            else:
                self.lower_band = self._prev_lower

            # Determine trend direction
            if self._prev_uptrend:
                self.is_uptrend = close >= self.lower_band
            else:
                self.is_uptrend = close > self.upper_band

            self.supertrend = self.lower_band if self.is_uptrend else self.upper_band
        else:
            self.upper_band = basic_upper
            self.lower_band = basic_lower
            self.supertrend = basic_lower
            self.is_uptrend = close > self.supertrend

        self._prev_close = close
        self._prev_upper = self.upper_band
        self._prev_lower = self.lower_band
        self._prev_uptrend = self.is_uptrend

        return {
            "supertrend": self.supertrend,
            "is_uptrend": self.is_uptrend,
            "upper_band": self.upper_band,
            "lower_band": self.lower_band,
        }

    def seed(self, candles: list):
        self._atr = IncrementalATR(self.atr_period)
        self._prev_close = None
        self._prev_upper = 0.0
        self._prev_lower = 0.0
        self._prev_uptrend = True
        self.is_uptrend = True
        self.supertrend = 0.0
        for c in candles:
            self.update(c["high"], c["low"], c["close"])


class IncrementalChandelierExit:
    """Chandelier Exit — best trailing stop for trending markets.
    Uses ATR period 22, multiplier 3.0 for crypto (avoids premature exits).
    Activates after position reaches 1R profit.
    """
    def __init__(self, period: int = 22, multiplier: float = 3.0):
        self.period = period
        self.multiplier = multiplier
        self._highs: list = []
        self._atr = IncrementalATR(period)
        self.long_stop: float = 0.0
        self.prev_long_stop: float = 0.0

    def update(self, high: float, low: float, close: float) -> float:
        self._atr.update(high, low, close)
        self._highs.append(high)
        if len(self._highs) > self.period:
            self._highs.pop(0)

        if self._atr.value and len(self._highs) >= self.period:
            highest_high = max(self._highs)
            new_stop = highest_high - self.multiplier * self._atr.value
            self.long_stop = max(new_stop, self.prev_long_stop)
            self.prev_long_stop = self.long_stop

        return self.long_stop

    def seed(self, candles: list):
        self._highs = []
        self._atr = IncrementalATR(self.period)
        self.long_stop = 0.0
        self.prev_long_stop = 0.0
        for c in candles:
            self.update(c["high"], c["low"], c["close"])


class IncrementalStochRSI:
    def __init__(self, rsi_period: int = 14, stoch_period: int = 14,
                 k_period: int = 3, d_period: int = 3):
        self.rsi = IncrementalRSI(rsi_period)
        self.stoch_period = stoch_period
        self.k_sma = IncrementalSMA(k_period)
        self.d_sma = IncrementalSMA(d_period)
        self._rsi_history: list = []
        self.k_value: float = 50.0
        self.d_value: float = 50.0

    def update(self, close: float) -> dict:
        rsi_val = self.rsi.update(close)
        self._rsi_history.append(rsi_val)
        if len(self._rsi_history) > self.stoch_period:
            self._rsi_history.pop(0)
        if len(self._rsi_history) >= self.stoch_period:
            rsi_min = min(self._rsi_history)
            rsi_max = max(self._rsi_history)
            rsi_range = rsi_max - rsi_min
            stoch_rsi = (rsi_val - rsi_min) / rsi_range * 100 if rsi_range > 0 else 50.0
        else:
            stoch_rsi = 50.0
        self.k_value = self.k_sma.update(stoch_rsi)
        self.d_value = self.d_sma.update(self.k_value)
        return {"k": self.k_value, "d": self.d_value}

    def seed(self, prices: list):
        self.rsi = IncrementalRSI(self.rsi.period)
        self._rsi_history = []
        for p in prices:
            self.update(p)


class IncrementalVolumeSMA:
    def __init__(self, period: int = 20):
        self.period = period
        self.window: list = []
        self.value: float = 0.0

    def update(self, volume: float) -> float:
        self.window.append(volume)
        if len(self.window) > self.period:
            self.window.pop(0)
        self.value = sum(self.window) / len(self.window)
        return self.value


class IncrementalOBV:
    def __init__(self):
        self.value: float = 0.0
        self.prev_close: float = None

    def update(self, close: float, volume: float) -> float:
        if self.prev_close is not None:
            if close > self.prev_close:
                self.value += volume
            elif close < self.prev_close:
                self.value -= volume
        self.prev_close = close
        return self.value


class IncrementalVWAP:
    """Volume-Weighted Average Price — resets each session (daily)."""
    def __init__(self):
        self._cum_tp_vol: float = 0.0
        self._cum_vol: float = 0.0
        self.value: float = 0.0
        self._candle_count: int = 0

    def update(self, high: float, low: float, close: float, volume: float) -> float:
        tp = (high + low + close) / 3
        self._cum_tp_vol += tp * volume
        self._cum_vol += volume
        self._candle_count += 1
        if self._cum_vol > 0:
            self.value = self._cum_tp_vol / self._cum_vol
        return self.value

    def reset(self):
        self._cum_tp_vol = 0.0
        self._cum_vol = 0.0
        self.value = 0.0
        self._candle_count = 0

    def seed(self, candles: list):
        self.reset()
        for c in candles:
            self.update(c["high"], c["low"], c["close"], c["volume"])


# ─── Composite Indicator Set ─────────────────────────────────────────────────

class IndicatorSet:
    """Full indicator set for one symbol on one timeframe."""

    def __init__(self):
        from config.settings import Settings
        cfg = Settings.strategy

        self.ema9 = IncrementalEMA(9)
        self.ema21 = IncrementalEMA(21)
        self.ema55 = IncrementalEMA(55)
        self.sma50 = IncrementalSMA(50)
        self.sma200 = IncrementalSMA(200)
        self.rsi = IncrementalRSI(cfg.RSI_PERIOD)
        self.rsi_fast = IncrementalRSI(cfg.RSI_FAST_PERIOD)
        self.macd = IncrementalMACD(cfg.MACD_FAST, cfg.MACD_SLOW, cfg.MACD_SIGNAL_PERIOD)
        self.bollinger = IncrementalBollinger(20, 2.5)
        self.atr = IncrementalATR(cfg.ATR_PERIOD)
        self.adx = IncrementalADX(14)
        self.supertrend = IncrementalSuperTrend(
            cfg.SUPERTREND_ATR_PERIOD, cfg.SUPERTREND_MULTIPLIER
        )
        self.chandelier = IncrementalChandelierExit(
            Settings.risk.CHANDELIER_ATR_PERIOD,
            Settings.risk.CHANDELIER_ATR_MULT
        )
        self.stoch_rsi = IncrementalStochRSI(14, 14, 3, 3)
        self.volume_sma = IncrementalVolumeSMA(cfg.VOLUME_SPIKE_LOOKBACK)
        self.obv = IncrementalOBV()
        self.vwap = IncrementalVWAP()
        self.latest: dict = {}
        self._sma50_window: list = []
        self._sma200_window: list = []

    def update_candle(self, candle: dict) -> dict:
        high = candle["high"]
        low = candle["low"]
        close = candle["close"]
        volume = candle["volume"]

        # Core indicators
        ema9 = self.ema9.update(close)
        ema21 = self.ema21.update(close)
        ema55 = self.ema55.update(close)
        self._sma50_window.append(close)
        if len(self._sma50_window) > 50:
            self._sma50_window.pop(0)
        self._sma200_window.append(close)
        if len(self._sma200_window) > 200:
            self._sma200_window.pop(0)
        sma50 = sum(self._sma50_window) / len(self._sma50_window)
        sma200 = sum(self._sma200_window) / len(self._sma200_window)

        rsi = self.rsi.update(close)
        rsi_fast = self.rsi_fast.update(close)
        macd = self.macd.update(close)
        bb = self.bollinger.update(close)
        atr = self.atr.update(high, low, close)
        adx = self.adx.update(high, low, close)
        st = self.supertrend.update(high, low, close)
        chandelier_stop = self.chandelier.update(high, low, close)
        stoch = self.stoch_rsi.update(close)
        vol_sma = self.volume_sma.update(volume)
        obv = self.obv.update(close, volume)
        vwap = self.vwap.update(high, low, close, volume)

        # Compute volume ratio
        vol_ratio = volume / vol_sma if vol_sma > 0 else 1.0

        self.latest = {
            "close": close, "high": high, "low": low, "volume": volume,
            "ema9": ema9, "ema21": ema21, "ema55": ema55,
            "sma50": sma50, "sma200": sma200,
            "rsi": rsi, "rsi_fast": rsi_fast,
            "macd": macd["macd"], "macd_signal": macd["signal"],
            "macd_histogram": macd["histogram"],
            "prev_macd_histogram": macd["prev_histogram"],
            "bb_upper": bb["upper"], "bb_middle": bb["middle"], "bb_lower": bb["lower"],
            "bb_width": bb["width"], "bb_percent_b": bb["percent_b"],
            "atr": atr,
            "adx": adx, "plus_di": self.adx.plus_di, "minus_di": self.adx.minus_di,
            "supertrend": st["supertrend"], "supertrend_up": st["is_uptrend"],
            "chandelier_stop": chandelier_stop,
            "stoch_k": stoch["k"], "stoch_d": stoch["d"],
            "volume_sma": vol_sma, "volume_ratio": vol_ratio,
            "obv": obv, "vwap": vwap,
            "open_time": candle.get("open_time", 0),
        }
        return self.latest

    def seed_from_klines(self, klines: list):
        for k in klines:
            self.update_candle(k)

    def update_tick(self, price: float) -> dict:
        ema9 = self.ema9.update(price)
        ema21 = self.ema21.update(price)
        rsi = self.rsi.update(price)
        macd = self.macd.update(price)
        self.latest.update({
            "close": price, "ema9": ema9, "ema21": ema21,
            "rsi": rsi, "rsi_fast": self.rsi_fast.update(price),
            "macd": macd["macd"], "macd_signal": macd["signal"],
            "macd_histogram": macd["histogram"],
            "prev_macd_histogram": macd["prev_histogram"],
        })
        return self.latest


# ─── Helper Functions ─────────────────────────────────────────────────────────

def detect_swing_points(prices: list, lookback: int = 3) -> dict:
    highs, lows = [], []
    for i in range(lookback, len(prices) - lookback):
        window = prices[i - lookback:i + lookback + 1]
        if prices[i] == max(window):
            highs.append(prices[i])
        if prices[i] == min(window):
            lows.append(prices[i])
    return {"swing_highs": highs, "swing_lows": lows}


def cluster_levels(prices: list, threshold_pct: float = 0.003) -> list:
    if not prices:
        return []
    sorted_prices = sorted(prices)
    clusters = []
    current_cluster = [sorted_prices[0]]
    for price in sorted_prices[1:]:
        if abs(price - current_cluster[-1]) / current_cluster[-1] <= threshold_pct:
            current_cluster.append(price)
        else:
            clusters.append({
                "level": sum(current_cluster) / len(current_cluster),
                "touches": len(current_cluster),
            })
            current_cluster = [price]
    if current_cluster:
        clusters.append({
            "level": sum(current_cluster) / len(current_cluster),
            "touches": len(current_cluster),
        })
    return clusters


def detect_rsi_divergence(closes: list, rsi_values: list, lookback: int = 10) -> bool:
    """Detect bullish RSI divergence (price lower lows, RSI higher lows)."""
    if len(closes) < lookback or len(rsi_values) < lookback:
        return False
    recent_closes = closes[-lookback:]
    recent_rsi = rsi_values[-lookback:]
    price_min1_idx = recent_closes.index(min(recent_closes[:lookback // 2]))
    price_min2_idx = lookback // 2 + recent_closes[lookback // 2:].index(
        min(recent_closes[lookback // 2:]))
    if recent_closes[price_min2_idx] < recent_closes[price_min1_idx]:
        if recent_rsi[price_min2_idx] > recent_rsi[price_min1_idx]:
            return True
    return False


def compute_batch_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Batch indicator computation for backtesting."""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    df["SMA_50"] = close.rolling(50).mean()
    df["SMA_200"] = close.rolling(200).mean()
    df["EMA_9"] = close.ewm(span=9, adjust=False).mean()
    df["EMA_21"] = close.ewm(span=21, adjust=False).mean()
    df["EMA_55"] = close.ewm(span=55, adjust=False).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss.replace(0, float("nan"))))
    ema8 = close.ewm(span=8, adjust=False).mean()
    ema17 = close.ewm(span=17, adjust=False).mean()
    df["MACD"] = ema8 - ema17
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    df["Volume_SMA"] = volume.rolling(20).mean()
    df["Volume_Ratio"] = volume / df["Volume_SMA"]
    return df
