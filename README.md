# 📊 NMDC Intraday & Swing Trade Analyzer

Automated NSE stock analyzer for **NMDC** running every **5 minutes** during market hours via GitHub Actions.

## 🔧 One-Time Setup (5 minutes)

### Step 1 — Fork / Create your repo
Create a new GitHub repo and upload all files maintaining this structure:
```
your-repo/
├── stock_analyzer.py
├── requirements.txt
├── results/                  ← auto-created by the bot
└── .github/
    └── workflows/
        └── nmdc_analysis.yml
```

### Step 2 — Create a Telegram Bot
1. Open Telegram → search **@BotFather**
2. Send `/newbot` → follow prompts → copy the **token**
3. Start a chat with your new bot
4. Get your **chat ID**:
   ```
   https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
   ```
   Look for `"chat":{"id":XXXXXXX}` — that number is your Chat ID

### Step 3 — Add GitHub Secrets
Go to your repo → **Settings → Secrets and variables → Actions → New secret**

| Secret Name        | Value                        |
|--------------------|------------------------------|
| `TELEGRAM_TOKEN`   | `123456789:ABCxyz...`        |
| `TELEGRAM_CHAT_ID` | `987654321`                  |

### Step 4 — Enable Write Permissions
Go to **Settings → Actions → General → Workflow permissions**  
Select ✅ **Read and write permissions** → Save

### Step 5 — Push and test
Push the files, then go to **Actions tab** → select workflow → **Run workflow** manually to verify.

---

## 📈 Strategy Summary

### Intraday (5-min chart)
| Indicator      | Role                              |
|----------------|-----------------------------------|
| EMA 9/21       | Entry trigger (crossover)         |
| Supertrend 7,3 | Trend direction confirmation      |
| RSI 14         | Momentum filter                   |
| Volume ratio   | Smart money confirmation (≥1.5×)  |
| VWAP           | Bias filter (above = bullish)     |
| MACD           | Momentum direction                |
| ATR 14         | Stop loss & target calculation    |

**Entry**: Score ≥ 7/12 with 2+ point margin  
**SL**: 1.5× ATR from entry  
**T1**: 2.0× ATR | **T2**: 3.5× ATR  

### Swing Trade (Daily chart)
| Indicator      | Role                              |
|----------------|-----------------------------------|
| EMA 21/50      | Trend alignment                   |
| MACD Daily     | Crossover signal                  |
| RSI 14         | Pullback entry zone               |
| Supertrend     | Direction filter                  |
| 52W range      | Context & value zone              |
| ATR 14         | Stop loss & target calculation    |

**Entry**: Score ≥ 8/14 with 2+ point margin  
**SL**: 2.0× ATR | **T1**: 3.0× ATR | **T2**: 5.0× ATR  
**Hold**: 2–15 days depending on signal strength

---

## 💰 Position Sizing (₹1,00,000 Capital)
- Risk per trade: **1.5%** = ₹1,500
- Qty = ₹1,500 ÷ (Entry − SL)
- Intraday cap: max **40%** of capital (₹40,000)
- Swing cap: max **60%** of capital (₹60,000)

---

## 📁 CSV Output — `results/nmdc_signals.csv`
Every 5 minutes a row is appended:

| Column | Description |
|---|---|
| timestamp | IST datetime |
| trade_type | INTRADAY / SWING |
| signal | BUY / SELL_SHORT / HOLD |
| price | LTP at signal time |
| sl | Stop loss |
| target1 / target2 | Price targets |
| qty | Suggested quantity |
| capital_deployed | ₹ deployed |
| confidence_pct | Signal strength % |
| rsi | RSI at signal time |
| atr | ATR value |
| bull_score / bear_score | Raw scoring |
| extra_info | Vol ratio, VWAP, etc. |

---

## ⚠️ Disclaimer
This tool is for **educational purposes only**. Not SEBI registered advice.  
Always apply your own judgement before placing any trade.
