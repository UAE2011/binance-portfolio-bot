"""
Asset Scanner — Filters Binance spot pairs for tradeable opportunities.
Minimum 24h volume: $5M (playbook: institutional liquidity threshold).
"""
import asyncio
from config.settings import Settings
from src.utils import setup_logging

logger = setup_logging()

QUOTE_ASSET = "USDT"
BLACKLIST = {
    "BTCUSDT", "ETHUSDT",  # handled separately in core
    "USDCUSDT", "BUSDUSDT", "TUSDUSDT", "FDUSDUSDT", "DAIUSDT",
    "PAXUSDT", "EURUSDT", "GBPUSDT",
}
EXCLUDED_SUFFIXES = ["UP", "DOWN", "BULL", "BEAR", "3L", "3S"]


class AssetScanner:
    def __init__(self, exchange, regime_detector, news_intel):
        self.exchange = exchange
        self.regime = regime_detector
        self.news = news_intel
        self._watchlist_cache: list = []
        self._cache_refresh_count: int = 0

    async def get_watchlist(self) -> list:
        """Refresh every 10 cycles (~40 min) or on regime change."""
        self._cache_refresh_count += 1
        if self._watchlist_cache and self._cache_refresh_count % 10 != 0:
            return self._watchlist_cache

        try:
            tickers = await self.exchange.get_24h_tickers()
        except Exception as e:
            logger.error("Ticker fetch failed: %s", e)
            return self._watchlist_cache or []

        min_volume = Settings.strategy.MIN_24H_VOLUME

        candidates = []
        for t in tickers:
            symbol = t.get("symbol", "")
            if not symbol.endswith(QUOTE_ASSET):
                continue
            if symbol in BLACKLIST:
                continue
            if Settings.is_stablecoin(symbol):
                continue
            if Settings.is_leveraged_token(symbol):
                continue
            if any(symbol.endswith(s) for s in EXCLUDED_SUFFIXES):
                continue

            try:
                vol_usdt = float(t.get("quoteVolume", 0))
                price = float(t.get("lastPrice", 0))
                price_change_pct = float(t.get("priceChangePercent", 0))

                if vol_usdt < min_volume:
                    continue
                if price <= 0:
                    continue
                # Avoid extreme dumpers (> 30% daily drop) — usually broken/hacked
                if price_change_pct < -30:
                    continue

                candidates.append({
                    "symbol": symbol,
                    "price": price,
                    "volume_usdt": vol_usdt,
                    "price_change_pct": price_change_pct,
                    "sector": Settings.get_sector_for_asset(symbol),
                })
            except (ValueError, TypeError):
                continue

        # Sort by volume desc (highest liquidity first)
        candidates.sort(key=lambda x: x["volume_usdt"], reverse=True)

        # Rotate list based on regime/rotation strategy
        rotation = self.news.get_dominance_strategy() if self.news else "NEUTRAL"
        if rotation == "ALTCOIN_SEASON":
            # Prioritize alt sectors: DeFi, AI, Gaming
            priority_sectors = {"DeFi", "AI", "Gaming", "Layer2"}
            alts = [c for c in candidates if c["sector"] in priority_sectors]
            others = [c for c in candidates if c["sector"] not in priority_sectors]
            candidates = alts + others

        max_size = Settings.strategy.MAX_WATCHLIST_SIZE
        self._watchlist_cache = candidates[:max_size]

        logger.info("Watchlist: %d pairs (top by volume, rotation=%s)",
                    len(self._watchlist_cache), rotation)
        return self._watchlist_cache

    async def get_core_assets(self) -> list:
        """Core portfolio assets — always monitored regardless of regime."""
        core = []
        for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]:
            try:
                price = await self.exchange.get_price(symbol)
                core.append({"symbol": symbol, "price": price, "sector": "Layer1"})
            except Exception:
                pass
        return core

    async def filter_momentum(self, candidates: list) -> list:
        """Quick momentum filter — keep assets with recent positive momentum."""
        result = []
        for c in candidates:
            pct = c.get("price_change_pct", 0)
            # Keep: gainers or small dips (potential bounce)
            if pct >= -5 or pct < -5:  # Keep all for full indicator analysis
                result.append(c)
        return result
