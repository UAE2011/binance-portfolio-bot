import asyncio
import numpy as np
from typing import Optional

from config.settings import Settings
from src.utils import setup_logging, utc_now

logger = setup_logging()

STABLECOIN_BASES = {"USDC", "BUSD", "DAI", "TUSD", "FDUSD", "UST", "USDP", "USDD"}
EXCLUDED_SYMBOLS = {"USDCUSDT", "BUSDUSDT", "TUSDUSDT", "FDUSDUSDT"}


class AssetScanner:
    def __init__(self, exchange, regime_detector, news_intel):
        self.exchange = exchange
        self.regime = regime_detector
        self.news = news_intel
        self.watchlist: list = []
        self.last_scan: Optional[str] = None

    async def scan_universe(self) -> list:
        logger.info("Scanning asset universe...")
        all_tickers = await self.exchange.get_all_tickers()
        if not all_tickers:
            logger.error("Failed to fetch tickers")
            return self.watchlist

        usdt_pairs = []
        for t in all_tickers:
            symbol = t["symbol"]
            if not symbol.endswith("USDT"):
                continue
            if symbol in EXCLUDED_SYMBOLS:
                continue
            base = symbol.replace("USDT", "")
            if base in STABLECOIN_BASES:
                continue
            filters = self.exchange.symbol_filters.get(symbol, {})
            if filters.get("status") != "TRADING":
                continue
            usdt_pairs.append({"symbol": symbol, "price": float(t["price"])})

        candidates = []
        batch_size = 10
        for i in range(0, len(usdt_pairs), batch_size):
            batch = usdt_pairs[i:i + batch_size]
            tasks = [self._evaluate_asset(asset) for asset in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, dict) and r.get("passes", False):
                    candidates.append(r)
            await asyncio.sleep(0.5)

        candidates.sort(key=lambda x: x.get("rank_score", 0), reverse=True)
        max_watchlist = Settings.strategy.MAX_WATCHLIST_SIZE
        self.watchlist = candidates[:max_watchlist]
        self.last_scan = utc_now().isoformat()

        logger.info(
            "Scan complete: %d/%d pairs passed filters, watchlist=%d",
            len(candidates), len(usdt_pairs), len(self.watchlist),
        )
        return self.watchlist

    async def _evaluate_asset(self, asset: dict) -> dict:
        symbol = asset["symbol"]
        try:
            ticker = await self.exchange.get_ticker_24h(symbol)
            if not ticker:
                return {"passes": False}

            quote_volume = float(ticker.get("quoteVolume", 0))
            if quote_volume < Settings.strategy.MIN_24H_VOLUME:
                return {"passes": False}

            price_change = float(ticker.get("priceChangePercent", 0))
            if price_change < -30:
                return {"passes": False}

            klines = await self.exchange.get_klines(symbol, "1d", 60)
            if len(klines) < 30:
                return {"passes": False}

            closes = [k["close"] for k in klines]
            volumes = [k["volume"] for k in klines]
            current_price = closes[-1]

            sma20 = np.mean(closes[-20:])
            sma50 = np.mean(closes[-50:]) if len(closes) >= 50 else sma20

            returns = np.diff(closes) / closes[:-1]
            volatility = np.std(returns) * np.sqrt(365)
            if volatility > 3.0:
                return {"passes": False}

            avg_volume = np.mean(volumes[-20:])
            recent_volume = np.mean(volumes[-5:])
            volume_trend = recent_volume / avg_volume if avg_volume > 0 else 0

            rank_score = 0

            if current_price > sma20:
                rank_score += 20
            if current_price > sma50:
                rank_score += 15

            if 0.5 < volatility < 1.5:
                rank_score += 15
            elif volatility <= 0.5:
                rank_score += 10

            if volume_trend > 1.5:
                rank_score += 20
            elif volume_trend > 1.2:
                rank_score += 10

            if quote_volume > 50_000_000:
                rank_score += 15
            elif quote_volume > 10_000_000:
                rank_score += 10
            elif quote_volume > 1_000_000:
                rank_score += 5

            if 0 < price_change < 10:
                rank_score += 10
            elif -5 < price_change <= 0:
                rank_score += 5

            asset_sentiment = self.news.get_asset_sentiment(symbol)
            if asset_sentiment > 0.3:
                rank_score += 10
            elif asset_sentiment > 0:
                rank_score += 5

            dominance_strategy = self.news.get_dominance_strategy()
            if dominance_strategy == "BTC_FOCUS" and symbol == "BTCUSDT":
                rank_score += 10
            elif dominance_strategy == "ALTCOIN_SEASON" and symbol != "BTCUSDT":
                rank_score += 5

            sector = Settings.get_sector_for_asset(symbol)

            return {
                "symbol": symbol,
                "price": current_price,
                "rank_score": rank_score,
                "quote_volume_24h": quote_volume,
                "price_change_24h": price_change,
                "volatility": volatility,
                "volume_trend": volume_trend,
                "above_sma20": current_price > sma20,
                "above_sma50": current_price > sma50,
                "sector": sector,
                "passes": True,
            }

        except Exception as e:
            logger.debug("Evaluation failed for %s: %s", symbol, e)
            return {"passes": False}

    async def quick_rescan(self) -> list:
        if not self.watchlist:
            return await self.scan_universe()

        updated = []
        for asset in self.watchlist:
            try:
                ticker = await self.exchange.get_ticker_24h(asset["symbol"])
                if ticker:
                    quote_vol = float(ticker.get("quoteVolume", 0))
                    price_change = float(ticker.get("priceChangePercent", 0))
                    if quote_vol >= Settings.strategy.MIN_24H_VOLUME and price_change > -30:
                        asset["quote_volume_24h"] = quote_vol
                        asset["price_change_24h"] = price_change
                        asset["price"] = float(ticker.get("lastPrice", asset["price"]))
                        updated.append(asset)
            except Exception:
                pass

        self.watchlist = sorted(updated, key=lambda x: x.get("rank_score", 0), reverse=True)
        return self.watchlist

    def get_watchlist_symbols(self) -> list:
        return [a["symbol"] for a in self.watchlist]

    def get_top_candidates(self, n: int = 10) -> list:
        return self.watchlist[:n]
