import sys
import logging
import math
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN

_logger_initialized = False


def setup_logging(level: str = "INFO") -> logging.Logger:
    global _logger_initialized
    logger = logging.getLogger("bot")
    if _logger_initialized:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s")

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    file_handler = RotatingFileHandler(
        log_dir / "bot.log", maxBytes=10 * 1024 * 1024, backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    _logger_initialized = True
    return logger


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def round_step_size(quantity: float, step_size: float) -> float:
    if step_size == 0:
        return quantity
    precision = int(round(-math.log10(step_size)))
    return float(Decimal(str(quantity)).quantize(Decimal(str(step_size)), rounding=ROUND_DOWN))


def round_tick_size(price: float, tick_size: float) -> float:
    if tick_size == 0:
        return price
    precision = int(round(-math.log10(tick_size)))
    return float(Decimal(str(price)).quantize(Decimal(str(tick_size)), rounding=ROUND_DOWN))


def format_pnl(pnl: float) -> str:
    sign = "+" if pnl >= 0 else ""
    return f"{sign}{pnl:.2f}"


def format_pct(pct: float) -> str:
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.2f}%"


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def symbol_to_base(symbol: str) -> str:
    for quote in ("USDT", "BUSD", "USDC"):
        if symbol.endswith(quote):
            return symbol[: -len(quote)]
    return symbol
