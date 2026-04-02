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
]
NEGATIVE_KEYWORDS = [
    "crash", "ban", "hack", "fraud", "liquidation", "bearish", "dump", "plunge",
    "scam", "exploit", "sec", "lawsuit", "investigation", "collapse", "outflow",
    "sell-off", "selloff", "bankrupt", "default", "delisted", "rug pull",
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
        self.btc_dominance: float = 50.0
        self.asset_sentiments: dict = {}
        self.last_update: Optional[datetime] = None

    async def update_all(self):
        tasks = [
            self._fetch_crypto_news(),
            self._fetch_fear_greed(),
            self._fetch_btc_dominance(),
            self._fetch_rss_headlines(),
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
        self.last_update = utc_now()
        logger.info(
            "News updated: F&G=%d (%s), BTC sentiment=%.2f, BTC dom=%.1f%%",
            self.fear_greed_value, self.fear_greed_label,
            self.btc_sentiment, self.btc_dominance,
        )

    async def _fetch_crypto_news(self):
        url = "https://cryptocurrency.cv/api/news"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params={"limit": 20},
                                       timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        articles = data if isinstance(data, list) else data.get("data", [])
                        self.latest_news = []
                        for article in articles[:20]:
                            self.latest_news.append({
                                "title": article.get("title", ""),
                                "source": article.get("source", ""),
                                "url": article.get("url", ""),
                                "published": article.get("published_at", ""),
                                "sentiment": self._keyword_sentiment(article.get("title", "")),
                            })
        except Exception as e:
            logger.warning("Crypto news fetch failed: %s", e)

        try:
            async with aiohttp.ClientSession() as session:
                sentiment_url = "https://cryptocurrency.cv/api/ai/sentiment"
                async with session.get(sentiment_url, params={"asset": "BTC"},
                                       timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        score = data.get("sentiment", data.get("score", 0))
                        if isinstance(score, (int, float)):
                            self.btc_sentiment = float(score)
                        elif isinstance(score, str):
                            sentiment_map = {"bullish": 0.5, "bearish": -0.5, "neutral": 0.0}
                            self.btc_sentiment = sentiment_map.get(score.lower(), 0.0)
        except Exception as e:
            logger.warning("AI sentiment fetch failed, using keyword analysis: %s", e)
            if self.latest_news:
                scores = [n["sentiment"] for n in self.latest_news]
                self.btc_sentiment = sum(scores) / len(scores) if scores else 0.0

    async def _fetch_fear_greed(self):
        url = "https://api.alternative.me/fng/"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params={"limit": 1, "format": "json"},
                                       timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        fg_data = data.get("data", [{}])[0]
                        self.fear_greed_value = int(fg_data.get("value", 50))
                        self.fear_greed_label = fg_data.get("value_classification", "Neutral")
        except Exception as e:
            logger.warning("Fear & Greed fetch failed: %s", e)

    async def _fetch_btc_dominance(self):
        url = "https://api.coingecko.com/api/v3/global"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        market_data = data.get("data", {})
                        self.btc_dominance = market_data.get("market_cap_percentage", {}).get("btc", 50.0)
        except Exception as e:
            logger.warning("BTC dominance fetch failed: %s", e)

    async def _fetch_rss_headlines(self):
        for source_name, feed_url in RSS_FEEDS:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(feed_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        if resp.status == 200:
                            content = await resp.text()
                            feed = feedparser.parse(content)
                            for entry in feed.entries[:5]:
                                title = entry.get("title", "")
                                sentiment = self._keyword_sentiment(title)
                                self.latest_news.append({
                                    "title": title,
                                    "source": source_name,
                                    "url": entry.get("link", ""),
                                    "published": entry.get("published", ""),
                                    "sentiment": sentiment,
                                })
            except Exception as e:
                logger.debug("RSS feed %s failed: %s", source_name, e)

    def _keyword_sentiment(self, text: str) -> float:
        if not text:
            return 0.0
        text_lower = text.lower()
        pos_count = sum(1 for kw in POSITIVE_KEYWORDS if kw in text_lower)
        neg_count = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text_lower)
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        return (pos_count - neg_count) / total

    def get_sentiment_score(self) -> float:
        if not self.latest_news:
            return self.btc_sentiment
        news_scores = [n["sentiment"] for n in self.latest_news if n["sentiment"] != 0]
        if news_scores:
            avg_news = sum(news_scores) / len(news_scores)
            return 0.6 * self.btc_sentiment + 0.4 * avg_news
        return self.btc_sentiment

    def get_asset_sentiment(self, symbol: str) -> float:
        base = symbol.replace("USDT", "").replace("BUSD", "").lower()
        relevant = [
            n for n in self.latest_news
            if base in n["title"].lower() or symbol.lower() in n["title"].lower()
        ]
        if not relevant:
            return 0.0
        return sum(n["sentiment"] for n in relevant) / len(relevant)

    def check_breaking_news(self, held_symbols: list) -> list:
        alerts = []
        for symbol in held_symbols:
            base = symbol.replace("USDT", "").replace("BUSD", "").lower()
            for news in self.latest_news[:10]:
                title_lower = news["title"].lower()
                if base in title_lower and news["sentiment"] < -0.3:
                    alerts.append({
                        "symbol": symbol,
                        "headline": news["title"],
                        "source": news["source"],
                        "sentiment": news["sentiment"],
                    })
        return alerts

    def get_sentiment_points(self) -> int:
        fg = self.fear_greed_value
        if fg <= 25:
            return 15
        elif fg <= 40:
            return 12
        elif fg <= 60:
            return 8
        elif fg <= 75:
            return 3
        else:
            return 0

    def is_extreme_fear(self) -> bool:
        return self.fear_greed_value < 25

    def is_extreme_greed(self) -> bool:
        return self.fear_greed_value > 75

    def get_dominance_strategy(self) -> str:
        if self.btc_dominance > 60:
            return "BTC_FOCUS"
        elif self.btc_dominance < 45:
            return "ALTCOIN_SEASON"
        return "BALANCED"

    def get_top_headlines(self, count: int = 5) -> list:
        return self.latest_news[:count]
