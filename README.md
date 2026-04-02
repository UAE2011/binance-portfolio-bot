# Binance Spot Portfolio Trading Bot

An intelligent, fully automated crypto portfolio manager for Binance spot markets. Long-only. Diversified. AI-powered.

## Features

### Core Trading
- **Diversified portfolio** — automatically scans and trades top crypto assets across sectors
- **Multi-timeframe analysis** — 15m (entry timing) + 1h (primary signals) + 4h (trend confirmation)
- **100-point confluence scoring** — technical, regime, sentiment, volume, risk/reward
- **AI trade advisor** — OpenAI-compatible API (gemini-2.5-flash) analyzes every trade with veto power

### Risk Management (SOL Bot Proven Defaults)
- **10% wallet rule** — max 10% of portfolio per position
- **3% stop-loss** — fixed with ATR validation
- **6% take-profit** — with 50/50 partial exit system
- **Trailing stop** — activates at 3% gain, trails by 1.5%
- **50/50 partial take-profit** — sell 50% at 6%, ride runner with trailing stop
- **Support break exit** — automatic exit when price breaks below support
- **Daily loss limit** — 3% max daily loss, then stops trading
- **Drawdown circuit breakers** — 5%/10%/15%/20% escalating responses

### Market Intelligence
- **HMM regime detection** — Hidden Markov Model identifies bull/bear/sideways/high-volatility
- **News sentiment** — real-time from CryptoPanic, CoinDesk, CoinTelegraph RSS feeds
- **Fear & Greed Index** — integrated into entry/exit decisions
- **Volume spike detection** — confirms breakouts and signals reversals

### Technical Indicators
- RSI, MACD, Bollinger Bands, EMA9/21, SMA50
- ATR, ADX, StochRSI, OBV, Volume SMA
- Support & Resistance levels (swing-point clustering)
- RSI divergence detection
- All computed incrementally (O(1) per candle)

### Automation
- **Self-calibrating** — auto-tunes Kelly fraction, ATR multipliers, confluence thresholds every 50 trades
- **Memory after restart** — persists all state to database and state file
- **Watchdog** — 10-second heartbeat, auto-recovery for API, WebSocket, database, memory
- **Smart DCA** — dollar-cost averages into positions during extreme fear

### Notifications (Telegram)
- Trade entry/exit alerts with P&L
- Regime change notifications
- Drawdown warnings
- Breaking news alerts for held positions
- Daily and weekly performance reports
- 16 commands: `/status`, `/trades`, `/pnl`, `/regime`, `/news`, `/ai`, `/risk`, `/pause`, `/resume`, `/sell`, `/sellall`, `/calibrate`, `/health`, `/report`, `/weekly`, `/help`

## Quick Start

### 1. Clone and configure

```bash
git clone <repo-url>
cd binance-portfolio-bot
cp .env.example .env
# Edit .env with your API keys
```

### 2. Deploy with Docker (recommended)

```bash
docker-compose up -d
```

The bot starts in **testnet mode** by default. Set `USE_TESTNET=false` for live trading.

### 3. Deploy without Docker

```bash
pip install -r requirements.txt
python main.py
```

## Configuration

All settings are in `.env`. Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `USE_TESTNET` | `true` | Use Binance testnet (safe mode) |
| `AI_TRADING_ENABLED` | `true` | Enable AI trade analysis |
| `AI_MODEL` | `gemini-2.5-flash` | AI model to use |
| `STOP_LOSS_PCT` | `0.03` | 3% stop-loss |
| `TAKE_PROFIT_PCT` | `0.06` | 6% take-profit |
| `PARTIAL_TP_PCT` | `0.50` | Sell 50% at TP |
| `MAX_POSITION_SIZE` | `0.10` | 10% wallet rule |
| `MAX_DAILY_LOSS` | `0.03` | 3% daily loss limit |
| `CONFLUENCE_SCORE_THRESHOLD` | `70` | Min score to enter |

## Architecture

```
main.py                    # Entry point — 3 async loops
├── src/
│   ├── ai_advisor.py      # AI trade analysis (OpenAI API)
│   ├── strategy.py        # Confluence scoring + S/R + volume spikes
│   ├── risk_manager.py    # 3% SL, 6% TP, 50/50 partials, trailing
│   ├── portfolio.py       # Portfolio management + execution
│   ├── exchange.py        # Binance REST + WebSocket
│   ├── indicators.py      # 10 incremental indicators
│   ├── regime.py          # HMM regime detection
│   ├── news_intelligence.py # News + sentiment + Fear & Greed
│   ├── scanner.py         # Asset universe scanning
│   ├── calibrator.py      # Self-calibration engine
│   ├── watchdog.py        # Health monitoring + self-healing
│   ├── notifier.py        # Telegram notifications + 16 commands
│   ├── database.py        # SQLAlchemy ORM (7 tables)
│   └── utils.py           # Logging + helpers
├── config/
│   ├── settings.py        # Configuration loader
│   └── sectors.json       # Crypto sector classification
├── backtest/
│   └── backtest.py        # Backtesting framework
├── healthcheck.py         # Docker health check
├── Dockerfile             # Multi-stage Docker build
└── docker-compose.yml     # One-command deployment
```

## Telegram Commands

| Command | Description |
|---|---|
| `/status` | Portfolio overview (value, cash, positions, regime) |
| `/trades` | Open positions with live P&L |
| `/pnl` | Today's realized P&L |
| `/regime` | Market regime and parameters |
| `/news` | Latest headlines and sentiment |
| `/ai` | AI advisor status and call count |
| `/risk` | Risk manager status (Kelly, win rate, etc.) |
| `/pause` | Pause all trading |
| `/resume` | Resume trading and reset kill switch |
| `/sell SYMBOL` | Force sell a specific position |
| `/sellall` | Force sell all positions |
| `/calibrate` | Force parameter recalibration |
| `/health` | System health (memory, CPU, components) |
| `/report` | Quick daily report |
| `/weekly` | Weekly performance summary |
| `/help` | List all commands |

## Safety

- Starts in **testnet mode** by default
- Kill switch at 20% portfolio drawdown
- Maximum 10 auto-restart attempts before requiring manual intervention
- All trades logged to SQLite database
- State persisted across restarts
