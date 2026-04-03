"""
AI Trade Advisor — Dual-Model Routing Strategy via Groq (FREE)

Groq free tier (April 2026):
  FAST model (llama-3.1-8b-instant): ultra-fast inference, ~0.1s latency, FREE
  STRONG model (llama-3.3-70b-versatile): excellent reasoning, ~0.3s latency, FREE

Groq limits: 30 RPM, 14,400 RPD per model — more than enough for 24/7 bot.

Routing logic:
  - FAST model (90% of calls): news sentiment scoring, quick confirmations, data extraction,
    market scans, volume checks, support/resistance classification
  - STRONG model (10% of calls): trade entry/exit decisions, portfolio rebalancing,
    regime analysis, complex multi-indicator confluence interpretation

The AI acts as a senior crypto portfolio manager that:
1. Analyzes technical indicators, market regime, and news sentiment
2. Provides BUY/SKIP/SELL verdicts with confidence scores
3. Suggests optimal position sizing and risk parameters
4. Detects patterns humans might miss (divergences, traps, fakeouts)
5. Provides portfolio-level advice on diversification and rebalancing
"""

import json
import asyncio
import time
from datetime import datetime, timezone
from typing import Optional

from openai import OpenAI

from config.settings import Settings
from src.utils import setup_logging, utc_now

logger = setup_logging()

# ─── System Prompts ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an elite cryptocurrency portfolio manager and technical analyst.
You manage a diversified spot-only, long-only crypto portfolio on Binance.

Your job is to analyze trade setups and provide precise, actionable decisions.
You are conservative by nature — you only approve high-conviction trades.

RULES:
- You ONLY trade spot markets, LONG only (buy low, sell high)
- You follow strict risk management: 10% max per position, 3% stop-loss, 6% take-profit
- You look for confluence across multiple indicators before approving
- You consider market regime (bull/bear/sideways) in every decision
- You factor in news sentiment and Fear & Greed index
- You watch for bull traps, fake breakouts, and overextended moves
- You prefer entries near support levels with volume confirmation
- You NEVER chase pumps or FOMO into positions
- You protect capital above all else

RESPONSE FORMAT — You MUST respond with valid JSON only, no markdown:
{
    "verdict": "BUY" | "SKIP" | "SELL",
    "confidence": 0.0 to 1.0,
    "reasoning": "2-3 sentence explanation",
    "risk_notes": "any specific risks to watch",
    "suggested_sl_pct": 0.03,
    "suggested_tp_pct": 0.06,
    "position_size_pct": 0.05 to 0.10,
    "urgency": "HIGH" | "MEDIUM" | "LOW",
    "pattern_detected": "description of any chart pattern or setup"
}"""

PORTFOLIO_SYSTEM_PROMPT = """You are an elite cryptocurrency portfolio manager.
You analyze an entire crypto portfolio and provide rebalancing and risk advice.

You consider:
- Overall portfolio diversification across sectors
- Correlation between holdings
- Current market regime and sentiment
- Individual position performance and risk
- Optimal cash allocation

RESPONSE FORMAT — You MUST respond with valid JSON only, no markdown:
{
    "health_score": 1 to 10,
    "risk_level": "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
    "recommendations": ["action1", "action2", ...],
    "positions_to_trim": ["SYMBOL1", ...],
    "positions_to_add": ["SYMBOL1", ...],
    "reasoning": "2-3 sentence portfolio assessment",
    "cash_recommendation": "increase" | "maintain" | "deploy"
}"""

SENTIMENT_SYSTEM_PROMPT = """You are a crypto news sentiment analyzer.
Analyze the following news headlines and return a sentiment score.

RESPONSE FORMAT — You MUST respond with valid JSON only, no markdown:
{
    "sentiment_score": -1.0 to 1.0,
    "key_events": ["event1", "event2"],
    "market_impact": "HIGH" | "MEDIUM" | "LOW",
    "summary": "one sentence summary"
}"""


# ─── Model Routing ───────────────────────────────────────────────────────────

class ModelTier:
    """Defines which model tier to use for each call type."""
    FAST = "fast"      # llama-3.1-8b-instant — ultra-fast, good for extraction
    STRONG = "strong"  # llama-3.3-70b-versatile — strong reasoning, trade decisions


# ─── AI Advisor ──────────────────────────────────────────────────────────────

class AIAdvisor:
    def __init__(self):
        self.cfg = Settings.ai
        self.client = OpenAI(
            api_key=self.cfg.API_KEY,
            base_url=self.cfg.BASE_URL,
        )
        self.daily_call_count: int = 0
        self.daily_cost_usd: float = 0.0
        self.last_reset_date: str = ""
        self.call_history: list = []
        self.last_portfolio_analysis: Optional[dict] = None
        self.last_portfolio_analysis_time: float = 0
        self.model_stats: dict = {
            ModelTier.FAST: {"calls": 0, "tokens": 0, "errors": 0},
            ModelTier.STRONG: {"calls": 0, "tokens": 0, "errors": 0},
        }

    # ── Model Routing ────────────────────────────────────────────────────

    def _get_model(self, tier: str) -> str:
        """Return the model name for the given tier."""
        if tier == ModelTier.STRONG:
            return self.cfg.STRONG_MODEL
        return self.cfg.FAST_MODEL

    def _estimate_cost(self, tier: str, total_tokens: int) -> float:
        """Estimate cost in USD for a call based on tier and tokens.
        Groq free tier: both models are $0.00 — tracking for future paid providers."""
        # Groq is free; return 0 for accurate tracking
        return 0.0

    # ── Daily Limits ─────────────────────────────────────────────────────

    def _reset_daily_counter(self):
        today = utc_now().strftime("%Y-%m-%d")
        if today != self.last_reset_date:
            self.daily_call_count = 0
            self.daily_cost_usd = 0.0
            self.last_reset_date = today
            for tier in self.model_stats:
                self.model_stats[tier] = {"calls": 0, "tokens": 0, "errors": 0}

    def _can_call(self) -> bool:
        self._reset_daily_counter()
        if self.daily_call_count >= self.cfg.MAX_DAILY_CALLS:
            return False
        # Cost limit of 0.00 means unlimited (free providers like Groq)
        if self.cfg.DAILY_COST_LIMIT_USD > 0 and self.daily_cost_usd >= self.cfg.DAILY_COST_LIMIT_USD:
            return False
        return True

    # ── Core AI Call with Routing ────────────────────────────────────────

    def _call_ai(self, system_prompt: str, user_prompt: str,
                 tier: str = ModelTier.FAST) -> Optional[dict]:
        """AI call with model routing, timeout, retry, and error handling."""
        if not self.cfg.ENABLED:
            return None
        if not self._can_call():
            logger.warning("AI daily limit reached (calls=%d, cost=$%.2f)",
                           self.daily_call_count, self.daily_cost_usd)
            return None

        model = self._get_model(tier)
        max_retries = 2
        last_error = None

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.2,
                    max_tokens=1000,
                    timeout=self.cfg.ANALYSIS_TIMEOUT,
                )
                self.daily_call_count += 1

                total_tokens = response.usage.total_tokens if response.usage else 0
                cost = self._estimate_cost(tier, total_tokens)
                self.daily_cost_usd += cost
                self.model_stats[tier]["calls"] += 1
                self.model_stats[tier]["tokens"] += total_tokens

                content = response.choices[0].message.content.strip()
                # Strip markdown code fences if present
                if content.startswith("```"):
                    lines = content.split("\n")
                    lines = [l for l in lines if not l.startswith("```")]
                    content = "\n".join(lines).strip()

                result = json.loads(content)
                self.call_history.append({
                    "time": utc_now().isoformat(),
                    "model": model,
                    "tier": tier,
                    "tokens": total_tokens,
                    "cost_usd": cost,
                })
                return result

            except json.JSONDecodeError as e:
                last_error = e
                logger.warning("AI (%s) returned invalid JSON (attempt %d): %s",
                               model, attempt + 1, e)
                # On JSON error, try the STRONG model as fallback
                if tier == ModelTier.FAST and attempt == 0:
                    model = self._get_model(ModelTier.STRONG)
                    logger.info("Escalating to STRONG model for JSON reliability")
                    continue
            except Exception as e:
                last_error = e
                self.model_stats[tier]["errors"] += 1
                logger.error("AI call failed (%s, attempt %d): %s",
                             model, attempt + 1, e)
                # On API error, try the other model
                if tier == ModelTier.FAST and attempt == 0:
                    model = self._get_model(ModelTier.STRONG)
                    logger.info("Falling back to STRONG model after FAST error")
                elif tier == ModelTier.STRONG and attempt == 0:
                    model = self._get_model(ModelTier.FAST)
                    logger.info("Falling back to FAST model after STRONG error")
                continue

        logger.error("AI call exhausted retries. Last error: %s", last_error)
        return None

    # ── Trade Analysis (STRONG model — critical decision) ────────────────

    async def analyze_trade(self, symbol: str, indicators: dict,
                            regime: str, news_sentiment: float,
                            fear_greed: int, support_levels: list,
                            resistance_levels: list,
                            recent_news: list,
                            portfolio_context: dict) -> Optional[dict]:
        """Ask AI to analyze a potential trade entry. Uses STRONG model."""
        if not self.cfg.ENABLED:
            return None

        prompt = self._build_trade_prompt(
            symbol, indicators, regime, news_sentiment,
            fear_greed, support_levels, resistance_levels,
            recent_news, portfolio_context,
        )

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self._call_ai, SYSTEM_PROMPT, prompt, ModelTier.STRONG
        )

        if result:
            result["symbol"] = symbol
            result["analyzed_at"] = utc_now().isoformat()
            result["model_used"] = self._get_model(ModelTier.STRONG)
            logger.info(
                "AI [STRONG] verdict for %s: %s (confidence=%.2f) — %s",
                symbol, result.get("verdict", "?"),
                result.get("confidence", 0),
                result.get("reasoning", "")[:100],
            )
        return result

    # ── Exit Analysis (STRONG model — critical decision) ─────────────────

    async def analyze_exit(self, trade: dict, current_price: float,
                           indicators: dict, regime: str,
                           news_sentiment: float,
                           support_levels: list) -> Optional[dict]:
        """Ask AI whether to exit or hold a current position. Uses STRONG model."""
        if not self.cfg.ENABLED:
            return None

        entry_price = trade["entry_price"]
        pnl_pct = ((current_price / entry_price) - 1) * 100
        remaining_qty = trade.get("remaining_quantity", trade["quantity"])
        exits_done = trade.get("tranche_exits", [])
        if isinstance(exits_done, str):
            exits_done = json.loads(exits_done)

        prompt = f"""POSITION EXIT ANALYSIS for {trade['symbol']}

CURRENT POSITION:
- Entry price: ${entry_price:.4f}
- Current price: ${current_price:.4f}
- P&L: {pnl_pct:+.2f}%
- Remaining quantity: {remaining_qty:.6f}
- Stop-loss: ${trade.get('stop_loss', 0):.4f}
- Partial exits done: {len(exits_done)}
- Time held: since {trade.get('entry_time', 'unknown')}
- Entry regime: {trade.get('regime_at_entry', 'unknown')}

CURRENT MARKET:
- Regime: {regime}
- News sentiment: {news_sentiment:.2f}
- RSI: {indicators.get('rsi', 50):.1f}
- MACD histogram: {indicators.get('macd_histogram', 0):.6f}
- Price vs EMA9: {'above' if current_price > indicators.get('ema9', 0) else 'below'}
- Price vs EMA21: {'above' if current_price > indicators.get('ema21', 0) else 'below'}
- ATR: {indicators.get('atr', 0):.4f}
- Volume vs avg: {indicators.get('volume', 0) / max(indicators.get('volume_sma', 1), 1):.1f}x

SUPPORT LEVELS BELOW: {[f'${s:.2f}' for s in support_levels[:3]]}

Should I HOLD, SELL partially (50%), or SELL fully? Consider trailing stop optimization."""

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self._call_ai, SYSTEM_PROMPT, prompt, ModelTier.STRONG
        )

        if result:
            result["symbol"] = trade["symbol"]
            result["analyzed_at"] = utc_now().isoformat()
        return result

    # ── Sentiment Analysis (FAST model — quick extraction) ───────────────

    async def analyze_sentiment(self, headlines: list) -> Optional[dict]:
        """Quick sentiment scoring of news headlines. Uses FAST model."""
        if not self.cfg.ENABLED or not headlines:
            return None

        news_text = "\n".join(f"- {h}" for h in headlines[:20])
        prompt = f"""Analyze these crypto news headlines and score overall sentiment:

{news_text}

Score from -1.0 (extremely bearish) to +1.0 (extremely bullish)."""

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self._call_ai, SENTIMENT_SYSTEM_PROMPT, prompt, ModelTier.FAST
        )
        return result

    # ── Quick Market Check (FAST model — rapid classification) ───────────

    async def quick_market_check(self, btc_data: dict, eth_data: dict,
                                  fear_greed: int) -> Optional[dict]:
        """Quick market health check. Uses FAST model."""
        if not self.cfg.ENABLED:
            return None

        prompt = f"""Quick crypto market health check:

BTC: price=${btc_data.get('price', 0):.0f}, 24h_change={btc_data.get('change_24h', 0):.2f}%, RSI={btc_data.get('rsi', 50):.0f}
ETH: price=${eth_data.get('price', 0):.0f}, 24h_change={eth_data.get('change_24h', 0):.2f}%, RSI={eth_data.get('rsi', 50):.0f}
Fear & Greed: {fear_greed}

Respond with JSON:
{{"market_health": "GOOD" | "CAUTION" | "DANGER", "brief": "one sentence", "trade_aggressiveness": 0.0 to 1.0}}"""

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self._call_ai, SYSTEM_PROMPT, prompt, ModelTier.FAST
        )
        return result

    # ── Portfolio Analysis (STRONG model — complex reasoning) ────────────

    async def analyze_portfolio(self, portfolio_status: dict,
                                open_positions: list,
                                regime: str, fear_greed: int,
                                news_sentiment: float,
                                sector_exposure: dict) -> Optional[dict]:
        """Ask AI for portfolio-level advice (cached for 1 hour). Uses STRONG model."""
        if not self.cfg.ENABLED:
            return None

        if time.time() - self.last_portfolio_analysis_time < 3600:
            return self.last_portfolio_analysis

        positions_text = ""
        for pos in open_positions:
            pnl_pct = 0
            if pos.get("entry_price", 0) > 0:
                pnl_pct = ((pos.get("current_price", pos["entry_price"]) / pos["entry_price"]) - 1) * 100
            positions_text += (
                f"  - {pos['symbol']}: entry=${pos['entry_price']:.4f}, "
                f"P&L={pnl_pct:+.2f}%, sector={pos.get('sector', 'Other')}\n"
            )

        sector_text = "\n".join(f"  - {k}: ${v:.2f}" for k, v in sector_exposure.items())

        prompt = f"""PORTFOLIO ANALYSIS

PORTFOLIO OVERVIEW:
- Total value: ${portfolio_status.get('portfolio_value', 0):.2f}
- Cash available: ${portfolio_status.get('cash_available', 0):.2f}
- Invested: ${portfolio_status.get('invested_value', 0):.2f}
- Open positions: {portfolio_status.get('open_positions', 0)}
- Current drawdown: {portfolio_status.get('drawdown', 0):.2%}

OPEN POSITIONS:
{positions_text if positions_text else '  None'}

SECTOR EXPOSURE:
{sector_text if sector_text else '  None'}

MARKET CONDITIONS:
- Regime: {regime}
- Fear & Greed: {fear_greed}
- News sentiment: {news_sentiment:.2f}

Analyze portfolio health, diversification, and provide actionable recommendations."""

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self._call_ai, PORTFOLIO_SYSTEM_PROMPT, prompt, ModelTier.STRONG
        )

        if result:
            self.last_portfolio_analysis = result
            self.last_portfolio_analysis_time = time.time()
            logger.info(
                "AI [STRONG] portfolio analysis: health=%s, risk=%s — %s",
                result.get("health_score", "?"),
                result.get("risk_level", "?"),
                result.get("reasoning", "")[:100],
            )
        return result

    # ── Trade Prompt Builder ─────────────────────────────────────────────

    def _build_trade_prompt(self, symbol: str, indicators: dict,
                            regime: str, news_sentiment: float,
                            fear_greed: int, support_levels: list,
                            resistance_levels: list,
                            recent_news: list,
                            portfolio_context: dict) -> str:
        price = indicators.get("close", 0)
        atr = indicators.get("atr", 0)
        atr_pct = (atr / price * 100) if price > 0 else 0

        news_text = ""
        for n in recent_news[:5]:
            title = n.get("title", "")
            sent = n.get("sentiment", "neutral")
            news_text += f"  - [{sent}] {title}\n"

        return f"""TRADE ANALYSIS REQUEST for {symbol}

PRICE & INDICATORS (1hr timeframe):
- Price: ${price:.4f}
- RSI(14): {indicators.get('rsi', 50):.1f}
- MACD line: {indicators.get('macd', 0):.6f}
- MACD signal: {indicators.get('macd_signal', 0):.6f}
- MACD histogram: {indicators.get('macd_histogram', 0):.6f}
- EMA9: ${indicators.get('ema9', 0):.4f}
- EMA21: ${indicators.get('ema21', 0):.4f}
- SMA50: ${indicators.get('sma50', 0):.4f}
- Bollinger Upper: ${indicators.get('bb_upper', 0):.4f}
- Bollinger Middle: ${indicators.get('bb_middle', 0):.4f}
- Bollinger Lower: ${indicators.get('bb_lower', 0):.4f}
- ATR(14): ${atr:.4f} ({atr_pct:.2f}%)
- ADX: {indicators.get('adx', 0):.1f}
- StochRSI K: {indicators.get('stoch_k', 50):.1f}
- StochRSI D: {indicators.get('stoch_d', 50):.1f}
- Volume: {indicators.get('volume', 0):.0f}
- Volume SMA: {indicators.get('volume_sma', 0):.0f}
- Volume ratio: {indicators.get('volume', 0) / max(indicators.get('volume_sma', 1), 1):.1f}x

SUPPORT LEVELS: {[f'${s:.2f}' for s in support_levels[:5]]}
RESISTANCE LEVELS: {[f'${r:.2f}' for r in resistance_levels[:5]]}

MARKET CONTEXT:
- Regime: {regime}
- Fear & Greed Index: {fear_greed}
- News sentiment: {news_sentiment:.2f} (-1 bearish to +1 bullish)

RECENT NEWS:
{news_text if news_text else '  No recent news'}

PORTFOLIO CONTEXT:
- Portfolio value: ${portfolio_context.get('portfolio_value', 0):.2f}
- Cash available: ${portfolio_context.get('cash_available', 0):.2f}
- Open positions: {portfolio_context.get('open_positions', 0)}
- Current drawdown: {portfolio_context.get('drawdown', 0):.2%}

Should I BUY {symbol} now? Analyze the setup quality, risk/reward, and timing."""

    # ── Status & Decision Helpers ────────────────────────────────────────

    def get_status(self) -> dict:
        return {
            "enabled": self.cfg.ENABLED,
            "fast_model": self.cfg.FAST_MODEL,
            "strong_model": self.cfg.STRONG_MODEL,
            "daily_calls": self.daily_call_count,
            "max_daily_calls": self.cfg.MAX_DAILY_CALLS,
            "daily_cost_usd": round(self.daily_cost_usd, 4),
            "daily_cost_limit": self.cfg.DAILY_COST_LIMIT_USD,
            "veto_power": self.cfg.VETO_POWER,
            "model_stats": {
                tier: {
                    "calls": stats["calls"],
                    "tokens": stats["tokens"],
                    "errors": stats["errors"],
                }
                for tier, stats in self.model_stats.items()
            },
            "last_portfolio_analysis": (
                self.last_portfolio_analysis.get("reasoning", "None")[:100]
                if self.last_portfolio_analysis else "None"
            ),
            "recent_calls": len(self.call_history),
        }

    def should_approve_entry(self, ai_result: Optional[dict],
                             confluence_score: int) -> dict:
        """Combine AI verdict with confluence score for final decision.
        When VETO_POWER is off, AI is advisory only — trades proceed on confluence."""
        if ai_result is None:
            return {"approved": True, "reason": "AI unavailable, using confluence only"}

        verdict = ai_result.get("verdict", "SKIP")
        confidence = ai_result.get("confidence", 0)

        if verdict == "BUY" and confidence >= self.cfg.MIN_CONFIDENCE:
            return {
                "approved": True,
                "reason": f"AI approves (confidence={confidence:.0%}): {ai_result.get('reasoning', '')}",
                "ai_confidence": confidence,
                "ai_sl": ai_result.get("suggested_sl_pct"),
                "ai_tp": ai_result.get("suggested_tp_pct"),
                "ai_size": ai_result.get("position_size_pct"),
            }

        # If VETO_POWER is on, AI can block trades
        if verdict == "SKIP" and self.cfg.VETO_POWER:
            return {
                "approved": False,
                "reason": f"AI vetoed (confidence={confidence:.0%}): {ai_result.get('risk_notes', '')}",
                "ai_confidence": confidence,
            }

        # If VETO_POWER is off, AI is advisory only — always approve if confluence met
        if not self.cfg.VETO_POWER:
            return {
                "approved": True,
                "reason": f"AI says {verdict} (confidence={confidence:.0%}) but veto off, proceeding on confluence ({confluence_score})",
                "ai_confidence": confidence,
            }

        return {
            "approved": False,
            "reason": f"AI verdict={verdict}, confidence={confidence:.0%}",
            "ai_confidence": confidence,
        }

    def should_exit_position(self, ai_result: Optional[dict]) -> dict:
        """Interpret AI exit analysis."""
        if ai_result is None:
            return {"action": "HOLD", "reason": "AI unavailable"}

        verdict = ai_result.get("verdict", "HOLD")
        confidence = ai_result.get("confidence", 0)

        if verdict == "SELL" and confidence >= 0.7:
            return {
                "action": "SELL_ALL",
                "reason": f"AI recommends full exit: {ai_result.get('reasoning', '')}",
                "confidence": confidence,
            }
        elif verdict == "SELL" and confidence >= 0.5:
            return {
                "action": "SELL_HALF",
                "reason": f"AI suggests partial exit: {ai_result.get('reasoning', '')}",
                "confidence": confidence,
            }
        else:
            return {
                "action": "HOLD",
                "reason": f"AI says hold: {ai_result.get('reasoning', '')}",
                "confidence": confidence,
            }
