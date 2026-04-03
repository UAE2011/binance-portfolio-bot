"""
News Intelligence — Market sentiment, Fear & Greed, BTC dominance,
altcoin season index, and RSS headline analysis.
"""
import asyncio
import re
from datetime import datetime, timezone
from typing import Optional

import aiohttp
import feedparser

from src.utils import setup_logging, utc_now

logger = setup_logging()

POSITIVE_KEYWORDS = [
    "bullish", "surge", "rally", "adoption", "approval", "breakout", "soar",
    "gain", "pump", "upgrade", "partnership", "launch", "milestone", "record",
    "institutional", "accumulation", "buy", "growth", "inflow", "etf approved",
    "all-time high", "ath", "recover", "rebound",
]
NEGATIVE_KEYWORDS = [
    "crash", "ban", "hack", "fraud", "liquidation", "bearish", "dump", "plunge",
    "scam", "exploit", "sec", "lawsuit", "investigation", "collapse", "outflow",
    "sell-off", "selloff", "bankrupt", "default", "delisted", "rug pull",
    "warning", "risk", "concern", "drop", "fall",
]

RSS_FEEDS = [
    ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("CoinTelegraph", "https://cointelegraph.com/rss"),
    ("Bitcoin Magazine", "https://bitcoinmagazine.com/.rss/full/"),
]


class NewsIntelligence:
    def __init__(self):
        self.latest_news: list = []
        self.btc_sentiment: float = 0.0
        self.fear_greed_value: int = 50
        self.fear_greed_label: str = "Neutral"
        self.fear_greed_history: list = []   # last 7 days
        self.btc_dominance: float = 50.0
        self.altcoin_season_index: int = 50
        self.asset_sentiments: dict = {}
        self.last_update: Optional[datetime] = None
        self._breaking_news_cache: list = []

    async def update_all(self):
        tasks = [
            self._fetch_fear_greed(),
            self._fetch_btc_dominance(),
            self._fetch_rss_headlines(),
            self._fetch_crypto_news(),
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
        self.last_update = utc_now()
        logger.info("News: F&G=%d (%s) | BTC Dom=%.1f%% | AltSeason=%d | Sentiment=%.2f",
                    self.fear_greed_value, self.fear_greed_label,
                    self.btc_dominance, self.altcoin_season_index,
                    self.btc_sentiment)

    async def _fetch_fear_greed(self):
        url = "https://api.alternative.me/fng/"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params={"limit": 7, "format": "json"},
                                       timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        entries = data.get("data", [])
                        if entries:
                            latest = entries[0]
                            self.fear_greed_value = int(latest.get("value", 50))
                            self.fear_greed_label = latest.get("value_classification", "Neutral")
                            self.fear_greed_history = [
                                {"value": int(e.get("value", 50)),
                                 "label": e.get("value_classification", ""),
                                 "timestamp": e.get("timestamp", "")}
                                for e in entries
                            ]
        except Exception as e:
            logger.warning("Fear & Greed fetch failed: %s", e)

    async def _fetch_btc_dominance(self):
        """Fetch BTC dominance and altcoin season index from CoinGecko."""
        try:
            async with aiohttp.ClientSession() as session:
                # Global market data
                url = "https://api.coingecko.com/api/v3/global"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        market = data.get("data", {})
                        dom = market.get("market_cap_percentage", {})
                        self.btc_dominance = dom.get("btc", 50.0)
                        logger.debug("BTC dominance: %.1f%%", self.btc_dominance)
        except Exception as e:
            logger.debug("BTC dominance fetch failed: %s", e)

        # Calculate a rough altcoin season index based on dominance
        # AltSeason = high when BTC dom falls, low when BTC dom rises
        # Scale: BTC dom 70% = altseason 15, BTC dom 40% = altseason 85
        if self.btc_dominance > 0:
            self.altcoin_season_index = max(0, min(100,
                int(100 - (self.btc_dominance - 40) * 2.5)
            ))

    async def _fetch_rss_headlines(self):
        """Parse RSS feeds for news headlines."""
        headlines = []
        async with aiohttp.ClientSession() as session:
            for source, url in RSS_FEEDS:
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        if resp.status == 200:
                            content = await resp.text()
                            feed = feedparser.parse(content)
                            for entry in feed.entries[:5]:
                                title = entry.get("title", "")
                                if title:
                                    sentiment = self._keyword_sentiment(title)
                                    headlines.append({
                                        "title": title,
                                        "source": source,
                                        "url": entry.get("link", ""),
                                        "published": entry.get("published", ""),
                                        "sentiment": sentiment,
                                        "is_breaking": self._is_breaking(title),
                                    })
                except Exception as e:
                    logger.debug("RSS fetch failed (%s): %s", source, e)

        if headlines:
            if not self.latest_news:
                self.latest_news = headlines
            else:
                # Merge, keep latest
                existing_titles = {n["title"] for n in self.latest_news}
                new_ones = [h for h in headlines if h["title"] not in existing_titles]
                self._breaking_news_cache.extend([h for h in new_ones if h.get("is_breaking")])
                self.latest_news = headlines + [n for n in self.latest_news
                                                if n["title"] not in {h["title"] for h in headlines}]
                self.latest_news = self.latest_news[:50]

        # Update BTC sentiment from latest headlines
        if self.latest_news:
            scores = [n["sentiment"] for n in self.latest_news[:20]]
            self.btc_sentiment = sum(scores) / len(scores) if scores else 0.0

    async def _fetch_crypto_news(self):
        """Try CryptoPanic or similar for asset-specific sentiment."""
        try:
            url = "https://cryptopanic.com/api/free/v1/posts/"
            params = {"auth_token": "pub_free", "filter": "hot", "currencies": "BTC,ETH,SOL"}
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params,
                                       timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        results = data.get("results", [])
                        for item in results[:10]:
                            title = item.get("title", "")
                            currencies = [c.get("code", "") for c in item.get("currencies", [])]
                            sentiment = self._keyword_sentiment(title)
                            for currency in currencies:
                                symbol = f"{currency}USDT"
                                if symbol not in self.asset_sentiments:
                                    self.asset_sentiments[symbol] = []
                                self.asset_sentiments[symbol].append(sentiment)
        except Exception as e:
            logger.debug("CryptoPanic fetch failed: %s", e)

    def _keyword_sentiment(self, text: str) -> float:
        text_lower = text.lower()
        pos = sum(1 for kw in POSITIVE_KEYWORDS if kw in text_lower)
        neg = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text_lower)
        total = pos + neg
        if total == 0:
            return 0.0
        return (pos - neg) / total

    def _is_breaking(self, title: str) -> bool:
        breaking_kw = ["breaking", "urgent", "just in", "alert", "crash",
                       "hack", "ban", "liquidation", "etf approved", "sec"]
        return any(kw in title.lower() for kw in breaking_kw)

    def get_sentiment_score(self) -> float:
        """Composite sentiment: BTC price sentiment + news."""
        return (self.btc_sentiment * 0.6 +
                (self.fear_greed_value / 100 - 0.5) * 0.4)

    def get_sentiment_points(self) -> int:
        """0-15 point scale for scoring."""
        score = self.get_sentiment_score()  # -1 to +1
        return max(0, min(15, int((score + 1) / 2 * 15)))

    def is_extreme_fear(self) -> bool:
        from config.settings import Settings
        return self.fear_greed_value <= Settings.portfolio_cfg.DCA_FEAR_THRESHOLD

    def is_extreme_greed(self) -> bool:
        return self.fear_greed_value >= 80

    def get_asset_sentiment(self, symbol: str) -> float:
        sentiments = self.asset_sentiments.get(symbol, [])
        if not sentiments:
            return 0.0
        return sum(sentiments) / len(sentiments)

    def get_dominance_strategy(self) -> str:
        """BTC dominance-based rotation advice."""
        from config.settings import Settings
        cfg = Settings.portfolio_cfg
        if self.btc_dominance >= cfg.BTC_DOM_BTC_FOCUS_THRESHOLD * 100:
            return "BTC_FOCUS"
        elif self.btc_dominance <= cfg.BTC_DOM_ALTSEASON_THRESHOLD * 100:
            return "ALTCOIN_SEASON"
        return "NEUTRAL"

    def get_top_headlines(self, n: int = 5) -> list:
        return self.latest_news[:n]

    def check_breaking_news(self, symbols: list) -> list:
        """Check for breaking news affecting held positions."""
        alerts = []
        if not symbols or not self.latest_news:
            return alerts
        for news in self.latest_news[:10]:
            title_lower = news["title"].lower()
            for symbol in symbols:
                base = symbol.replace("USDT", "").lower()
                if base in title_lower or news.get("is_breaking"):
                    if abs(news.get("sentiment", 0)) > 0.3:
                        alerts.append({**news, "symbol": symbol})
                        break
        # Also return cached breaking news
        alerts.extend(self._breaking_news_cache[-3:])
        self._breaking_news_cache = []
        return alerts[:5]

    def get_market_summary(self) -> str:
        """One-line market summary for Telegram."""
        dom_strategy = self.get_dominance_strategy()
        dom_str = {
            "BTC_FOCUS": "🟠 BTC Season",
            "ALTCOIN_SEASON": "🟢 Alt Season",
            "NEUTRAL": "⚪ Neutral",
        }.get(dom_strategy, "⚪")

        fg_emoji = "😱" if self.fear_greed_value < 20 else \
                   "😰" if self.fear_greed_value < 40 else \
                   "😐" if self.fear_greed_value < 60 else \
                   "😊" if self.fear_greed_value < 80 else "🤑"

        return (f"{fg_emoji} F&G: {self.fear_greed_value} ({self.fear_greed_label}) | "
                f"BTC Dom: {self.btc_dominance:.1f}% | {dom_str} | "
                f"AltIdx: {self.altcoin_season_index}")
