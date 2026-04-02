# Feature Integration Map: SOL Bot -> Diversified Portfolio Bot

## SOL Bot Features to Integrate

| Feature | SOL Bot Spec | Current Bot Status | Action Needed |
|---|---|---|---|
| AI Trading (Claude) | Uses Claude for trade analysis | NOT present | ADD - new ai_advisor.py module |
| 10% Wallet Rule | Max 10% per trade | Has 10% MAX_POSITION_SIZE | Already aligned |
| Stop-Loss 3% | Fixed 3% stop-loss | ATR-based dynamic SL | CHANGE - add fixed 3% as default, keep ATR as option |
| Take-Profit 6% | Fixed 6% take-profit | ATR-based multi-tranche | CHANGE - add fixed 6% as default |
| Trailing Stop | Active trailing stop | ATR-based trailing | Keep, enhance |
| Partial Take-Profit 50/50 | Sell 50% at TP1, ride 50% | 25/25/25/25 tranches | CHANGE to 50/50 split |
| Memory After Restart | Persists state across restarts | Has DB but needs state recovery | ENHANCE state recovery in main.py |
| RSI + MACD + Bollinger | Core indicators | All present | Keep |
| Support & Resistance | S/R level detection | Present in strategy.py | Keep, enhance |
| Support Break Exit | Exit when support breaks | NOT present | ADD to position monitor |
| Volume Spike Detection | Detects volume spikes | Has volume scoring | ENHANCE with spike alerts |
| Multi-Timeframe 15m + 1hr | Uses 15m and 1hr | Uses 1h and 4h | CHANGE to 15m + 1hr + 4h |
| Daily Loss Limit | Stops trading on daily loss | Present | Keep |
| Telegram Notifications | Trade alerts | Present | Keep |
| Telegram Commands | Remote control | Present | Keep, add missing commands |
| Weekly Report | Weekly performance report | Has daily report | ADD weekly report |
| News Sentiment Filter | Filters trades by news | Present | Keep |
| Auto-Restart Watchdog | Self-healing restart | Present | Keep |

## Key Architecture Change: SOL-only -> Diversified Portfolio
- Scan and trade top crypto assets, not just SOL
- Sector diversification with correlation filtering
- Dynamic asset selection based on market conditions
