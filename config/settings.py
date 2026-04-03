"""
Configuration — Playbook-optimized parameters for exponential growth
with capital preservation in bearish environments.
"""
import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent.parent


class BinanceConfig:
    API_KEY: str = os.getenv("BINANCE_API_KEY", "")
    API_SECRET: str = os.getenv("BINANCE_API_SECRET", "")
    USE_TESTNET: bool = os.getenv("USE_TESTNET", "true").lower() == "true"
    BASE_URL = "https://testnet.binance.vision" if USE_TESTNET else "https://api.binance.com"
    WS_URL = "wss://testnet.binance.vision/ws" if USE_TESTNET else "wss://stream.binance.com:9443/ws"
    STREAM_URL = ("wss://testnet.binance.vision/stream" if USE_TESTNET
                  else "wss://stream.binance.com:9443/stream")


class TelegramConfig:
    BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")


class AIConfig:
    ENABLED: bool = os.getenv("AI_TRADING_ENABLED", "true").lower() == "true"
    API_KEY: str = os.getenv("GEMINI_API_KEY", os.getenv("GROQ_API_KEY", ""))
    BASE_URL: str = os.getenv("AI_BASE_URL",
                              "https://generativelanguage.googleapis.com/v1beta/openai/")
    FAST_MODEL: str = os.getenv("AI_FAST_MODEL", "gemini-2.0-flash")
    STRONG_MODEL: str = os.getenv("AI_STRONG_MODEL", "gemini-2.5-flash-preview-04-17")
    MODEL: str = os.getenv("AI_MODEL", "gemini-2.5-flash-preview-04-17")
    MIN_CONFIDENCE: float = float(os.getenv("AI_MIN_CONFIDENCE", "0.60"))
    VETO_MIN_CONFIDENCE: float = float(os.getenv("AI_VETO_MIN_CONFIDENCE", "0.60"))
    ANALYSIS_TIMEOUT: int = int(os.getenv("AI_ANALYSIS_TIMEOUT", "30"))
    MAX_DAILY_CALLS: int = int(os.getenv("AI_MAX_DAILY_CALLS", "500"))
    MAX_CALLS_PER_CYCLE: int = int(os.getenv("AI_MAX_CALLS_PER_CYCLE", "5"))
    VETO_POWER: bool = os.getenv("AI_VETO_POWER", "true").lower() == "true"
    DAILY_COST_LIMIT_USD: float = float(os.getenv("AI_DAILY_COST_LIMIT", "0.00"))


class RiskConfig:
    KELLY_FRACTION: float = float(os.getenv("KELLY_FRACTION", "0.25"))
    MAX_RISK_PER_TRADE: float = float(os.getenv("MAX_RISK_PER_TRADE", "0.02"))
    MAX_POSITION_SIZE: float = float(os.getenv("MAX_POSITION_SIZE", "0.15"))
    MAX_OPEN_POSITIONS: int = int(os.getenv("MAX_OPEN_POSITIONS", "8"))
    MAX_PORTFOLIO_HEAT: float = float(os.getenv("MAX_PORTFOLIO_HEAT", "0.10"))
    MIN_CASH_RESERVE: float = float(os.getenv("MIN_CASH_RESERVE", "0.15"))
    STOP_LOSS_PCT: float = float(os.getenv("STOP_LOSS_PCT", "0.03"))
    TAKE_PROFIT_PCT: float = float(os.getenv("TAKE_PROFIT_PCT", "0.06"))
    ATR_STOP_MULTIPLIER: float = float(os.getenv("ATR_STOP_MULTIPLIER", "3.0"))
    ATR_TP_MULTIPLIER: float = float(os.getenv("ATR_TP_MULTIPLIER", "4.5"))
    TRAILING_STOP_ACTIVATION_PCT: float = float(os.getenv("TRAILING_STOP_ACTIVATION_PCT", "0.02"))
    TRAILING_STOP_DISTANCE_PCT: float = float(os.getenv("TRAILING_STOP_DISTANCE_PCT", "0.015"))
    CHANDELIER_ATR_PERIOD: int = int(os.getenv("CHANDELIER_ATR_PERIOD", "22"))
    CHANDELIER_ATR_MULT: float = float(os.getenv("CHANDELIER_ATR_MULT", "3.0"))
    PARTIAL_TP_ENABLED: bool = os.getenv("PARTIAL_TP_ENABLED", "true").lower() == "true"
    PARTIAL_TP_PCT: float = float(os.getenv("PARTIAL_TP_PCT", "0.50"))
    PARTIAL_TP_TRIGGER: float = float(os.getenv("PARTIAL_TP_TRIGGER", "0.06"))
    RUNNER_TRAILING_DISTANCE: float = float(os.getenv("RUNNER_TRAILING_DISTANCE", "0.02"))
    MAX_PORTFOLIO_DRAWDOWN: float = float(os.getenv("MAX_PORTFOLIO_DRAWDOWN", "0.20"))
    CIRCUIT_BREAKER_1: float = float(os.getenv("CIRCUIT_BREAKER_1", "0.05"))
    CIRCUIT_BREAKER_2: float = float(os.getenv("CIRCUIT_BREAKER_2", "0.10"))
    CIRCUIT_BREAKER_3: float = float(os.getenv("CIRCUIT_BREAKER_3", "0.15"))
    MAX_DAILY_LOSS: float = float(os.getenv("MAX_DAILY_LOSS", "0.03"))
    WIN_SCALE_UP: float = float(os.getenv("WIN_SCALE_UP", "1.10"))
    LOSS_SCALE_DOWN: float = float(os.getenv("LOSS_SCALE_DOWN", "0.90"))
    MAX_SCALE_FACTOR: float = float(os.getenv("MAX_SCALE_FACTOR", "2.0"))
    MIN_SCALE_FACTOR: float = float(os.getenv("MIN_SCALE_FACTOR", "0.25"))
    SUPPORT_BREAK_EXIT: bool = os.getenv("SUPPORT_BREAK_EXIT", "true").lower() == "true"
    SUPPORT_BREAK_THRESHOLD: float = float(os.getenv("SUPPORT_BREAK_THRESHOLD", "0.005"))


class StrategyConfig:
    CONFLUENCE_SCORE_THRESHOLD: int = int(os.getenv("CONFLUENCE_SCORE_THRESHOLD", "45"))
    TIMEFRAMES: list = os.getenv("TIMEFRAMES", "15m,1h,4h").split(",")
    PRIMARY_TIMEFRAME: str = os.getenv("PRIMARY_TIMEFRAME", "1h")
    ENTRY_TIMEFRAME: str = os.getenv("ENTRY_TIMEFRAME", "15m")
    TREND_TIMEFRAME: str = os.getenv("TREND_TIMEFRAME", "4h")
    SCAN_INTERVAL_SECONDS: int = int(os.getenv("SCAN_INTERVAL_SECONDS", "60"))
    SCAN_INTERVAL_HOURS: int = int(os.getenv("SCAN_INTERVAL_HOURS", "4"))
    RSI_PERIOD: int = int(os.getenv("RSI_PERIOD", "14"))
    RSI_FAST_PERIOD: int = int(os.getenv("RSI_FAST_PERIOD", "7"))
    RSI_MOMENTUM_THRESHOLD: float = float(os.getenv("RSI_MOMENTUM_THRESHOLD", "50.0"))
    MACD_FAST: int = int(os.getenv("MACD_FAST", "8"))
    MACD_SLOW: int = int(os.getenv("MACD_SLOW", "17"))
    MACD_SIGNAL_PERIOD: int = int(os.getenv("MACD_SIGNAL_PERIOD", "9"))
    ATR_PERIOD: int = int(os.getenv("ATR_PERIOD", "14"))
    ATR_STOP_MULTIPLIER: float = float(os.getenv("ATR_STOP_MULTIPLIER", "3.0"))
    SUPERTREND_ATR_PERIOD: int = int(os.getenv("SUPERTREND_ATR_PERIOD", "14"))
    SUPERTREND_MULTIPLIER: float = float(os.getenv("SUPERTREND_MULTIPLIER", "3.5"))
    VOLUME_SPIKE_MULTIPLIER: float = float(os.getenv("VOLUME_SPIKE_MULTIPLIER", "1.5"))
    VOLUME_BREAKOUT_MULTIPLIER: float = float(os.getenv("VOLUME_BREAKOUT_MULTIPLIER", "2.0"))
    VOLUME_SPIKE_LOOKBACK: int = int(os.getenv("VOLUME_SPIKE_LOOKBACK", "20"))
    SR_LOOKBACK_CANDLES: int = int(os.getenv("SR_LOOKBACK_CANDLES", "100"))
    SR_PROXIMITY_PCT: float = float(os.getenv("SR_PROXIMITY_PCT", "0.015"))
    PEAK_HOURS_START: int = int(os.getenv("PEAK_HOURS_START", "14"))
    PEAK_HOURS_END: int = int(os.getenv("PEAK_HOURS_END", "22"))
    PEAK_HOURS_BONUS: int = int(os.getenv("PEAK_HOURS_BONUS", "5"))
    NEWS_CHECK_INTERVAL_MINUTES: int = int(os.getenv("NEWS_CHECK_INTERVAL_MINUTES", "15"))
    REBALANCE_DAY: int = int(os.getenv("REBALANCE_DAY", "6"))
    HMM_RETRAIN_DAY: int = int(os.getenv("HMM_RETRAIN_DAY", "6"))
    HMM_LOOKBACK_DAYS: int = int(os.getenv("HMM_LOOKBACK_DAYS", "90"))
    MIN_24H_VOLUME: float = float(os.getenv("MIN_24H_VOLUME", "5000000"))
    MAX_WATCHLIST_SIZE: int = int(os.getenv("MAX_WATCHLIST_SIZE", "50"))


class PortfolioConfig:
    CORE_ALLOCATION: float = float(os.getenv("CORE_ALLOCATION", "0.50"))
    SATELLITE_ALLOCATION: float = float(os.getenv("SATELLITE_ALLOCATION", "0.35"))
    SPECULATIVE_ALLOCATION: float = float(os.getenv("SPECULATIVE_ALLOCATION", "0.05"))
    DRY_POWDER: float = float(os.getenv("DRY_POWDER", "0.10"))
    BTC_DOM_ALTSEASON_THRESHOLD: float = float(os.getenv("BTC_DOM_ALTSEASON_THRESHOLD", "0.50"))
    BTC_DOM_BTC_FOCUS_THRESHOLD: float = float(os.getenv("BTC_DOM_BTC_FOCUS_THRESHOLD", "0.65"))
    ALTSEASON_INDEX_ENTRY: int = int(os.getenv("ALTSEASON_INDEX_ENTRY", "50"))
    ALTSEASON_INDEX_EXIT: int = int(os.getenv("ALTSEASON_INDEX_EXIT", "75"))
    REBALANCE_DRIFT_THRESHOLD: float = float(os.getenv("REBALANCE_DRIFT_THRESHOLD", "0.12"))
    MAX_SINGLE_POSITION_PCT: float = float(os.getenv("MAX_SINGLE_POSITION_PCT", "0.15"))
    DCA_FEAR_THRESHOLD: int = int(os.getenv("DCA_FEAR_THRESHOLD", "15"))
    DCA_BUDGET_PCT: float = float(os.getenv("DCA_BUDGET_PCT", "0.05"))
    DCA_TRANCHES: int = int(os.getenv("DCA_TRANCHES", "5"))
    DCA_MULTIPLIER: float = float(os.getenv("DCA_MULTIPLIER", "3.0"))


class RegimeConfig:
    BEAR_CASH_RESERVE: float = float(os.getenv("BEAR_CASH_RESERVE", "0.70"))
    SIDEWAYS_CASH_RESERVE: float = float(os.getenv("SIDEWAYS_CASH_RESERVE", "0.40"))
    BULL_CASH_RESERVE: float = float(os.getenv("BULL_CASH_RESERVE", "0.15"))
    REQUIRE_BTC_ABOVE_200SMA: bool = os.getenv("REQUIRE_BTC_ABOVE_200SMA", "false").lower() == "true"
    ENABLE_ALTCOIN_ROTATION: bool = os.getenv("ENABLE_ALTCOIN_ROTATION", "true").lower() == "true"


class CalibrationConfig:
    ENABLED: bool = os.getenv("CALIBRATION_ENABLED", "true").lower() == "true"
    MIN_TRADES: int = int(os.getenv("CALIBRATION_MIN_TRADES", "20"))
    LOOKBACK_TRADES: int = int(os.getenv("CALIBRATION_LOOKBACK_TRADES", "50"))
    CALIBRATION_INTERVAL_TRADES: int = int(os.getenv("CALIBRATION_INTERVAL_TRADES", "20"))
    WALK_FORWARD_ENABLED: bool = os.getenv("WALK_FORWARD_ENABLED", "true").lower() == "true"


class OperationalConfig:
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    DB_PATH: str = os.getenv("DB_PATH", "data/bot.db")
    TIMEZONE: str = os.getenv("TIMEZONE", "UTC")
    WATCHDOG_INTERVAL: int = int(os.getenv("WATCHDOG_INTERVAL_SECONDS", "10"))
    HEARTBEAT_FILE: str = os.getenv("HEARTBEAT_FILE", "data/heartbeat.txt")
    STATE_FILE: str = os.getenv("STATE_FILE", "data/bot_state.json")
    WEEKLY_REPORT_DAY: int = int(os.getenv("WEEKLY_REPORT_DAY", "0"))
    WEEKLY_REPORT_HOUR: int = int(os.getenv("WEEKLY_REPORT_HOUR", "9"))


def load_sectors() -> dict:
    sectors_path = BASE_DIR / "config" / "sectors.json"
    with open(sectors_path, "r") as f:
        return json.load(f)


class Settings:
    binance = BinanceConfig()
    telegram = TelegramConfig()
    ai = AIConfig()
    risk = RiskConfig()
    strategy = StrategyConfig()
    portfolio_cfg = PortfolioConfig()
    regime_cfg = RegimeConfig()
    calibration = CalibrationConfig()
    ops = OperationalConfig()
    _sectors_data = None

    BINANCE_API_KEY: str = BinanceConfig.API_KEY
    BINANCE_API_SECRET: str = BinanceConfig.API_SECRET
    TESTNET: bool = BinanceConfig.USE_TESTNET
    DATABASE_PATH: str = OperationalConfig.DB_PATH
    TELEGRAM_BOT_TOKEN: str = TelegramConfig.BOT_TOKEN
    TELEGRAM_CHAT_ID: str = TelegramConfig.CHAT_ID

    @classmethod
    def sectors(cls) -> dict:
        if cls._sectors_data is None:
            cls._sectors_data = load_sectors()
        return cls._sectors_data

    @classmethod
    def get_sector_for_asset(cls, symbol: str) -> str:
        base = symbol.replace("USDT", "").replace("BUSD", "")
        data = cls.sectors()
        for sector_name, sector_info in data.get("sectors", {}).items():
            if base in sector_info.get("assets", []):
                return sector_name
        return "Other"

    @classmethod
    def is_stablecoin(cls, symbol: str) -> bool:
        base = symbol.replace("USDT", "").replace("BUSD", "")
        return base in cls.sectors().get("stablecoins", [])

    @classmethod
    def is_leveraged_token(cls, symbol: str) -> bool:
        excluded = cls.sectors().get("excluded_suffixes", [])
        return any(symbol.endswith(s) for s in excluded)
