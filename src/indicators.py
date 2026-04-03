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


class IncrementalRSI:
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
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast_ema = IncrementalEMA(fast)
        self.slow_ema = IncrementalEMA(slow)
        self.signal_ema = IncrementalEMA(signal)
        self.macd_line: float = 0.0
        self.signal_line: float = 0.0
        self.histogram: float = 0.0

    def update(self, price: float) -> dict:
        fast_val = self.fast_ema.update(price)
        slow_val = self.slow_ema.update(price)
        self.macd_line = fast_val - slow_val
        self.signal_line = self.signal_ema.update(self.macd_line)
        self.histogram = self.macd_line - self.signal_line
        return {
            "macd": self.macd_line,
            "signal": self.signal_line,
            "histogram": self.histogram,
        }

    def seed(self, prices: list):
        self.fast_ema.seed(prices)
        self.slow_ema.seed(prices)
        for p in prices:
            fast_val = self.fast_ema.update(p)
            slow_val = self.slow_ema.update(p)
            ml = fast_val - slow_val
            self.signal_ema.update(ml)
        self.macd_line = self.fast_ema.value - self.slow_ema.value
        self.signal_line = self.signal_ema.value
        self.histogram = self.macd_line - self.signal_line


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


class IncrementalBollinger:
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        self.period = period
        self.std_dev = std_dev
        self.window: list = []
        self.middle: float = 0.0
        self.upper: float = 0.0
        self.lower: float = 0.0

    def update(self, close: float) -> dict:
        self.window.append(close)
        if len(self.window) > self.period:
            self.window.pop(0)
        self.middle = np.mean(self.window)
        std = np.std(self.window) if len(self.window) > 1 else 0
        self.upper = self.middle + self.std_dev * std
        self.lower = self.middle - self.std_dev * std
        return {"middle": self.middle, "upper": self.upper, "lower": self.lower}


class IncrementalADX:
    def __init__(self, period: int = 14):
        self.period = period
        self.prev_high: float = None
        self.prev_low: float = None
        self.prev_close: float = None
        self.smoothed_plus_dm: float = 0.0
        self.smoothed_minus_dm: float = 0.0
        self.smoothed_tr: float = 0.0
        self.adx: float = 0.0
        self.count: int = 0

    def update(self, high: float, low: float, close: float) -> float:
        if self.prev_high is None:
            self.prev_high = high
            self.prev_low = low
            self.prev_close = close
            return 0.0

        plus_dm = max(high - self.prev_high, 0) if (high - self.prev_high) > (self.prev_low - low) else 0
        minus_dm = max(self.prev_low - low, 0) if (self.prev_low - low) > (high - self.prev_high) else 0
        tr = max(high - low, abs(high - self.prev_close), abs(low - self.prev_close))

        self.prev_high = high
        self.prev_low = low
        self.prev_close = close
        self.count += 1

        if self.count == 1:
            self.smoothed_plus_dm = plus_dm
            self.smoothed_minus_dm = minus_dm
            self.smoothed_tr = tr
        else:
            self.smoothed_plus_dm = self.smoothed_plus_dm - (self.smoothed_plus_dm / self.period) + plus_dm
            self.smoothed_minus_dm = self.smoothed_minus_dm - (self.smoothed_minus_dm / self.period) + minus_dm
            self.smoothed_tr = self.smoothed_tr - (self.smoothed_tr / self.period) + tr

        if self.smoothed_tr == 0:
            return self.adx

        plus_di = 100 * self.smoothed_plus_dm / self.smoothed_tr
        minus_di = 100 * self.smoothed_minus_dm / self.smoothed_tr
        di_sum = plus_di + minus_di
        dx = 100 * abs(plus_di - minus_di) / di_sum if di_sum > 0 else 0

        if self.count <= self.period:
            self.adx = dx
        else:
            self.adx = (self.adx * (self.period - 1) + dx) / self.period
        return self.adx


class IncrementalStochRSI:
    def __init__(self, rsi_period: int = 14, stoch_period: int = 14, k: int = 3, d: int = 3):
        self.rsi = IncrementalRSI(rsi_period)
        self.stoch_period = stoch_period
        self.rsi_window: list = []
        self.k_sma = IncrementalEMA(k)
        self.d_sma = IncrementalEMA(d)
        self.k_value: float = 50.0
        self.d_value: float = 50.0

    def update(self, close: float) -> dict:
        rsi_val = self.rsi.update(close)
        self.rsi_window.append(rsi_val)
        if len(self.rsi_window) > self.stoch_period:
            self.rsi_window.pop(0)

        rsi_min = min(self.rsi_window)
        rsi_max = max(self.rsi_window)
        rsi_range = rsi_max - rsi_min

        if rsi_range > 0:
            stoch_rsi = (rsi_val - rsi_min) / rsi_range * 100
        else:
            stoch_rsi = 50.0

        self.k_value = self.k_sma.update(stoch_rsi)
        self.d_value = self.d_sma.update(self.k_value)
        return {"k": self.k_value, "d": self.d_value}


class IndicatorSet:
    """Full set of incremental indicators for a single symbol/timeframe."""

    def __init__(self):
        self.ema9 = IncrementalEMA(9)
        self.ema21 = IncrementalEMA(21)
        self.sma50_window: list = []
        self.sma200_window: list = []
        self.rsi = IncrementalRSI(14)
        self.macd = IncrementalMACD(12, 26, 9)
        self.bollinger = IncrementalBollinger(20, 2.0)
        self.atr = IncrementalATR(14)
        self.adx = IncrementalADX(14)
        self.volume_sma = IncrementalVolumeSMA(20)
        self.obv = IncrementalOBV()
        self.stoch_rsi = IncrementalStochRSI(14, 14, 3, 3)
        self.latest: dict = {}

    @property
    def sma50(self) -> float:
        return np.mean(self.sma50_window) if len(self.sma50_window) >= 50 else 0.0

    @property
    def sma200(self) -> float:
        return np.mean(self.sma200_window) if len(self.sma200_window) >= 200 else 0.0

    def update_candle(self, candle: dict):
        close = candle["close"]
        high = candle["high"]
        low = candle["low"]
        volume = candle["volume"]

        self.sma50_window.append(close)
        if len(self.sma50_window) > 50:
            self.sma50_window.pop(0)
        self.sma200_window.append(close)
        if len(self.sma200_window) > 200:
            self.sma200_window.pop(0)

        ema9 = self.ema9.update(close)
        ema21 = self.ema21.update(close)
        rsi = self.rsi.update(close)
        macd = self.macd.update(close)
        bb = self.bollinger.update(close)
        atr = self.atr.update(high, low, close)
        adx = self.adx.update(high, low, close)
        vol_sma = self.volume_sma.update(volume)
        obv = self.obv.update(close, volume)
        stoch = self.stoch_rsi.update(close)

        prev_macd_histogram = self.latest.get("macd_histogram", macd["histogram"])
        self.latest = {
            "close": close, "high": high, "low": low, "volume": volume,
            "ema9": ema9, "ema21": ema21,
            "sma50": self.sma50, "sma200": self.sma200,
            "rsi": rsi,
            "macd": macd["macd"], "macd_signal": macd["signal"],
            "macd_histogram": macd["histogram"],
            "prev_macd_histogram": prev_macd_histogram,
            "bb_upper": bb["upper"], "bb_middle": bb["middle"], "bb_lower": bb["lower"],
            "atr": atr, "adx": adx,
            "volume_sma": vol_sma, "obv": obv,
            "stoch_k": stoch["k"], "stoch_d": stoch["d"],
        }
        return self.latest

    def seed_from_klines(self, klines: list):
        for k in klines:
            self.update_candle(k)

    def update_tick(self, price: float) -> dict:
        """Incremental update from a live tick (between candle closes)."""
        ema9 = self.ema9.update(price)
        ema21 = self.ema21.update(price)
        rsi = self.rsi.update(price)
        macd = self.macd.update(price)
        self.latest.update({
            "close": price, "ema9": ema9, "ema21": ema21,
            "rsi": rsi, "macd": macd["macd"],
            "macd_signal": macd["signal"], "macd_histogram": macd["histogram"],
        })
        return self.latest


def compute_batch_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute full indicators on a DataFrame of OHLCV data using pure pandas/numpy."""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # SMA
    df["SMA_50"] = close.rolling(window=50).mean()
    df["SMA_200"] = close.rolling(window=200).mean()

    # EMA
    df["EMA_9"] = close.ewm(span=9, adjust=False).mean()
    df["EMA_21"] = close.ewm(span=21, adjust=False).mean()

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).ewm(alpha=1.0 / 14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1.0 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI_14"] = 100.0 - (100.0 / (1.0 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD_12_26_9"] = ema12 - ema26
    df["MACDs_12_26_9"] = df["MACD_12_26_9"].ewm(span=9, adjust=False).mean()
    df["MACDh_12_26_9"] = df["MACD_12_26_9"] - df["MACDs_12_26_9"]

    # Bollinger Bands
    sma20 = close.rolling(window=20).mean()
    std20 = close.rolling(window=20).std()
    df["BBU_20_2.0"] = sma20 + 2.0 * std20
    df["BBM_20_2.0"] = sma20
    df["BBL_20_2.0"] = sma20 - 2.0 * std20

    # ATR
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    df["ATRr_14"] = tr.ewm(alpha=1.0 / 14, adjust=False).mean()

    # ADX
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    atr14 = df["ATRr_14"]
    plus_di = 100.0 * (plus_dm.ewm(alpha=1.0 / 14, adjust=False).mean() / atr14.replace(0, np.nan))
    minus_di = 100.0 * (minus_dm.ewm(alpha=1.0 / 14, adjust=False).mean() / atr14.replace(0, np.nan))
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["ADX_14"] = dx.ewm(alpha=1.0 / 14, adjust=False).mean()

    # OBV
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    df["OBV"] = obv

    # StochRSI
    rsi = df["RSI_14"]
    rsi_min = rsi.rolling(window=14).min()
    rsi_max = rsi.rolling(window=14).max()
    stochrsi = (rsi - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)
    df["STOCHRSIk_14_14_3_3"] = stochrsi.rolling(window=3).mean() * 100
    df["STOCHRSId_14_14_3_3"] = df["STOCHRSIk_14_14_3_3"].rolling(window=3).mean()

    return df


def detect_swing_points(closes: list, lookback: int = 5) -> dict:
    """Detect swing highs and swing lows for S/R level detection."""
    swing_highs = []
    swing_lows = []
    for i in range(lookback, len(closes) - lookback):
        is_high = all(closes[i] >= closes[i - j] for j in range(1, lookback + 1)) and \
                  all(closes[i] >= closes[i + j] for j in range(1, lookback + 1))
        is_low = all(closes[i] <= closes[i - j] for j in range(1, lookback + 1)) and \
                 all(closes[i] <= closes[i + j] for j in range(1, lookback + 1))
        if is_high:
            swing_highs.append(closes[i])
        if is_low:
            swing_lows.append(closes[i])
    return {"swing_highs": swing_highs, "swing_lows": swing_lows}


def cluster_levels(levels: list, threshold_pct: float = 0.005) -> list:
    """Cluster nearby price levels into zones."""
    if not levels:
        return []
    sorted_levels = sorted(levels)
    clusters = [[sorted_levels[0]]]
    for level in sorted_levels[1:]:
        if abs(level - clusters[-1][-1]) / clusters[-1][-1] <= threshold_pct:
            clusters[-1].append(level)
        else:
            clusters.append([level])
    result = []
    for cluster in clusters:
        result.append({
            "level": np.mean(cluster),
            "touches": len(cluster),
            "strength": len(cluster),
        })
    return sorted(result, key=lambda x: x["touches"], reverse=True)


def detect_rsi_divergence(prices: list, rsi_values: list, lookback: int = 30) -> bool:
    """Detect bullish RSI divergence."""
    if len(prices) < lookback or len(rsi_values) < lookback:
        return False
    recent_prices = prices[-lookback:]
    recent_rsi = rsi_values[-lookback:]

    price_lows = []
    rsi_at_lows = []
    for i in range(2, len(recent_prices) - 2):
        if recent_prices[i] <= min(recent_prices[i - 2:i]) and \
           recent_prices[i] <= min(recent_prices[i + 1:i + 3]):
            price_lows.append((i, recent_prices[i]))
            rsi_at_lows.append((i, recent_rsi[i]))

    if len(price_lows) < 2:
        return False

    last_two_price = [price_lows[-2], price_lows[-1]]
    last_two_rsi = [rsi_at_lows[-2], rsi_at_lows[-1]]

    price_lower_low = last_two_price[1][1] < last_two_price[0][1]
    rsi_higher_low = last_two_rsi[1][1] > last_two_rsi[0][1]
    rsi_below_45 = last_two_rsi[1][1] < 45

    return price_lower_low and rsi_higher_low and rsi_below_45
