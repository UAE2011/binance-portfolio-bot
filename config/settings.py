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
    STREAM_URL = (
        "wss://testnet.binance.vision/stream"
        if USE_TESTNET
        else "wss://stream.binance.com:9443/stream"
    )


class TelegramConfig:
    BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")


class AIConfig:
    """Dual-model AI routing via Groq (FREE, no credit card required).

    Groq free tier (April 2026):
    - llama-3.3-70b-versatile: 30 RPM, 14,400 RPD, 6000 tokens/min — strong reasoning
    - llama-3.1-8b-instant: 30 RPM, 14,400 RPD, 6000 tokens/min — ultra-fast

    Routing logic:
    - FAST model (90% of calls): news sentiment, quick confirmations, data extraction
    - STRONG model (10% of calls): trade entry/exit decisions, portfolio rebalancing, regime analysis

    Both models are 100% FREE via Groq's OpenAI-compatible API.
    """
    ENABLED: bool = os.getenv("AI_TRADING_ENABLED", "true").lower() == "true"
    # Groq API configuration
    API_KEY: str = os.getenv("GROQ_API_KEY", "")
    BASE_URL: str = os.getenv("AI_BASE_URL", "https://api.groq.com/openai/v1")
    # Dual-model routing (Groq free models)
    FAST_MODEL: str = os.getenv("AI_FAST_MODEL", "llama-3.1-8b-instant")
    STRONG_MODEL: str = os.getenv("AI_STRONG_MODEL", "llama-3.3-70b-versatile")
    # Legacy single-model fallback
    MODEL: str = os.getenv("AI_MODEL", "llama-3.3-70b-versatile")
    MIN_CONFIDENCE: float = float(os.getenv("AI_MIN_CONFIDENCE", "0.65"))
    ANALYSIS_TIMEOUT: int = int(os.getenv("AI_ANALYSIS_TIMEOUT", "30"))
    MAX_DAILY_CALLS: int = int(os.getenv("AI_MAX_DAILY_CALLS", "500"))
    VETO_POWER: bool = os.getenv("AI_VETO_POWER", "true").lower() == "true"
    # Cost tracking (Groq is free, but keep for future paid providers)
    DAILY_COST_LIMIT_USD: float = float(os.getenv("AI_DAILY_COST_LIMIT", "0.00"))


class RiskConfig:
    """Risk management — mirrors SOL bot's proven parameters."""
    RISK_LEVEL: int = int(os.getenv("RISK_LEVEL", "3"))
    # 10% wallet rule — max 10% of portfolio per trade
    MAX_POSITION_SIZE: float = float(os.getenv("MAX_POSITION_SIZE", "0.10"))
    MAX_RISK_PER_TRADE: float = float(os.getenv("MAX_RISK_PER_TRADE", "0.02"))
    MAX_OPEN_POSITIONS: int = int(os.getenv("MAX_OPEN_POSITIONS", "10"))
    MIN_CASH_RESERVE: float = float(os.getenv("MIN_CASH_RESERVE", "0.20"))
    MAX_PORTFOLIO_DRAWDOWN: float = float(os.getenv("MAX_PORTFOLIO_DRAWDOWN", "0.20"))
    MAX_DAILY_LOSS: float = float(os.getenv("MAX_DAILY_LOSS", "0.03"))
    # Fixed stop-loss and take-profit (SOL bot defaults)
    STOP_LOSS_PCT: float = float(os.getenv("STOP_LOSS_PCT", "0.03"))        # 3%
    TAKE_PROFIT_PCT: float = float(os.getenv("TAKE_PROFIT_PCT", "0.06"))    # 6%
    # Trailing stop activation and trail distance
    TRAILING_STOP_ACTIVATION_PCT: float = float(os.getenv("TRAILING_STOP_ACTIVATION_PCT", "0.03"))  # activate after 3% gain
    TRAILING_STOP_DISTANCE_PCT: float = float(os.getenv("TRAILING_STOP_DISTANCE_PCT", "0.015"))     # trail by 1.5%
    # Partial take-profit: 50/50 split (SOL bot style)
    PARTIAL_TP_ENABLED: bool = os.getenv("PARTIAL_TP_ENABLED", "true").lower() == "true"
    PARTIAL_TP_PCT: float = float(os.getenv("PARTIAL_TP_PCT", "0.50"))       # sell 50% at TP1
    PARTIAL_TP_TRIGGER: float = float(os.getenv("PARTIAL_TP_TRIGGER", "0.06"))  # trigger at 6%
    RUNNER_TRAILING_DISTANCE: float = float(os.getenv("RUNNER_TRAILING_DISTANCE", "0.02"))  # trail runner by 2%
    # Support break exit
    SUPPORT_BREAK_EXIT: bool = os.getenv("SUPPORT_BREAK_EXIT", "true").lower() == "true"
    SUPPORT_BREAK_THRESHOLD: float = float(os.getenv("SUPPORT_BREAK_THRESHOLD", "0.005"))  # 0.5% below support


class StrategyConfig:
    CONFLUENCE_SCORE_THRESHOLD: int = int(os.getenv("CONFLUENCE_SCORE_THRESHOLD", "70"))
    SCAN_INTERVAL_SECONDS: int = int(os.getenv("SCAN_INTERVAL_SECONDS", "60"))
    SCAN_INTERVAL_HOURS: int = int(os.getenv("SCAN_INTERVAL_HOURS", "4"))
    # Multi-timeframe: 15m + 1hr (SOL bot) + 4h (portfolio bot)
    TIMEFRAMES: list = os.getenv("TIMEFRAMES", "15m,1h,4h").split(",")
    PRIMARY_TIMEFRAME: str = os.getenv("PRIMARY_TIMEFRAME", "1h")
    ENTRY_TIMEFRAME: str = os.getenv("ENTRY_TIMEFRAME", "15m")
    TREND_TIMEFRAME: str = os.getenv("TREND_TIMEFRAME", "4h")
    # Indicator settings
    ATR_PERIOD: int = int(os.getenv("ATR_PERIOD", "14"))
    ATR_STOP_MULTIPLIER: float = float(os.getenv("ATR_STOP_MULTIPLIER", "2.5"))
    RSI_PERIOD: int = int(os.getenv("RSI_PERIOD", "14"))
    # Volume spike detection
    VOLUME_SPIKE_MULTIPLIER: float = float(os.getenv("VOLUME_SPIKE_MULTIPLIER", "2.0"))
    VOLUME_SPIKE_LOOKBACK: int = int(os.getenv("VOLUME_SPIKE_LOOKBACK", "20"))
    # S/R detection
    SR_LOOKBACK_CANDLES: int = int(os.getenv("SR_LOOKBACK_CANDLES", "100"))
    SR_PROXIMITY_PCT: float = float(os.getenv("SR_PROXIMITY_PCT", "0.01"))
    # Regime and news
    REBALANCE_DAY: int = int(os.getenv("REBALANCE_DAY", "6"))
    HMM_RETRAIN_DAY: int = int(os.getenv("HMM_RETRAIN_DAY", "6"))
    HMM_LOOKBACK_DAYS: int = int(os.getenv("HMM_LOOKBACK_DAYS", "90"))
    NEWS_CHECK_INTERVAL_MINUTES: int = int(os.getenv("NEWS_CHECK_INTERVAL_MINUTES", "15"))
    MIN_24H_VOLUME: float = float(os.getenv("MIN_24H_VOLUME", "1000000"))
    MAX_WATCHLIST_SIZE: int = int(os.getenv("MAX_WATCHLIST_SIZE", "50"))


class CalibrationConfig:
    ENABLED: bool = os.getenv("CALIBRATION_ENABLED", "true").lower() == "true"
    MIN_TRADES: int = int(os.getenv("CALIBRATION_MIN_TRADES", "50"))
    LOOKBACK_TRADES: int = int(os.getenv("CALIBRATION_LOOKBACK_TRADES", "50"))
    CALIBRATION_INTERVAL_TRADES: int = int(os.getenv("CALIBRATION_INTERVAL_TRADES", "50"))
    WALK_FORWARD_ENABLED: bool = os.getenv("WALK_FORWARD_ENABLED", "true").lower() == "true"


class OperationalConfig:
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    DB_PATH: str = os.getenv("DB_PATH", "data/bot.db")
    TIMEZONE: str = os.getenv("TIMEZONE", "UTC")
    WATCHDOG_INTERVAL: int = int(os.getenv("WATCHDOG_INTERVAL_SECONDS", "10"))
    HEARTBEAT_FILE: str = os.getenv("HEARTBEAT_FILE", "/tmp/bot_heartbeat")
    STATE_FILE: str = os.getenv("STATE_FILE", "data/bot_state.json")
    # Weekly report
    WEEKLY_REPORT_DAY: int = int(os.getenv("WEEKLY_REPORT_DAY", "0"))  # 0=Monday
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
    calibration = CalibrationConfig()
    ops = OperationalConfig()
    _sectors_data = None

    # Top-level aliases for backward compatibility
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
