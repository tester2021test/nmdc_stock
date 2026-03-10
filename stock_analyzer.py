"""
NMDC Stock Analyzer - Intraday & Swing Trade Strategy
Author: 30-Year Market Experience System
Capital: ₹1,00,000
Exchange: NSE (NMDC.NS)
"""

import os
import time
import logging
import traceback
import csv
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import yfinance as yf
import pandas as pd
import pandas_ta as ta
import requests

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger("NMDC")

# ── Constants ─────────────────────────────────────────────────────────────────
TICKER          = "NMDC.NS"
CAPITAL         = 100_000          # ₹1 lakh
RISK_PCT        = 0.015            # 1.5% risk per trade  → ₹1,500
MAX_INTRADAY_LOTS = 0.40           # max 40% capital in one intraday trade
MAX_SWING_LOTS    = 0.60           # max 60% capital in one swing trade
IST             = ZoneInfo("Asia/Kolkata")
MARKET_OPEN     = (9, 15)
MARKET_CLOSE    = (15, 30)

TELEGRAM_TOKEN  = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT   = os.environ.get("TELEGRAM_CHAT_ID", "")
CSV_PATH        = "results/nmdc_signals.csv"

RETRY_ATTEMPTS  = 5
RETRY_DELAY     = 8                # seconds between retries


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY
# ═══════════════════════════════════════════════════════════════════════════════

def is_market_open() -> bool:
    now = datetime.now(IST)
    if now.weekday() >= 5:          # Saturday / Sunday
        return False
    h, m = now.hour, now.minute
    open_min  = MARKET_OPEN[0]  * 60 + MARKET_OPEN[1]
    close_min = MARKET_CLOSE[0] * 60 + MARKET_CLOSE[1]
    return open_min <= (h * 60 + m) <= close_min


def retry(fn, attempts=RETRY_ATTEMPTS, delay=RETRY_DELAY):
    """Generic retry wrapper with exponential back-off."""
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            wait = delay * attempt
            log.warning(f"Attempt {attempt}/{attempts} failed: {exc}. Retrying in {wait}s…")
            time.sleep(wait)
    raise RuntimeError(f"All {attempts} attempts failed: {last_exc}") from last_exc


# ═══════════════════════════════════════════════════════════════════════════════
# DATA FETCH
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_intraday() -> pd.DataFrame:
    """Fetch last 5 days of 5-min OHLCV for intraday analysis."""
    def _fetch():
        df = yf.download(TICKER, period="5d", interval="5m",
                         auto_adjust=True, progress=False)
        if df.empty:
            raise ValueError("Empty intraday data returned")
        df.columns = [c.lower() if isinstance(c, str) else c[0].lower()
                      for c in df.columns]
        df = df.dropna()
        return df
    return retry(_fetch)


def fetch_daily() -> pd.DataFrame:
    """Fetch 200 days of daily OHLCV for swing analysis."""
    def _fetch():
        df = yf.download(TICKER, period="200d", interval="1d",
                         auto_adjust=True, progress=False)
        if df.empty:
            raise ValueError("Empty daily data returned")
        df.columns = [c.lower() if isinstance(c, str) else c[0].lower()
                      for c in df.columns]
        df = df.dropna()
        return df
    return retry(_fetch)


# ═══════════════════════════════════════════════════════════════════════════════
# INDICATORS
# ═══════════════════════════════════════════════════════════════════════════════

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Trend
    df["ema9"]  = ta.ema(df["close"], length=9)
    df["ema21"] = ta.ema(df["close"], length=21)
    df["ema50"] = ta.ema(df["close"], length=50)
    df["ema200"]= ta.ema(df["close"], length=200)

    # Momentum
    df["rsi"]   = ta.rsi(df["close"], length=14)

    # MACD
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df["macd"]        = macd["MACD_12_26_9"]
    df["macd_signal"] = macd["MACDs_12_26_9"]
    df["macd_hist"]   = macd["MACDh_12_26_9"]

    # Bollinger Bands
    bbands = ta.bbands(df["close"], length=20, std=2)
    df["bb_upper"] = bbands["BBU_20_2.0"]
    df["bb_mid"]   = bbands["BBM_20_2.0"]
    df["bb_lower"] = bbands["BBL_20_2.0"]

    # ATR (for position sizing & SL)
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    # Supertrend
    st = ta.supertrend(df["high"], df["low"], df["close"], length=7, multiplier=3)
    df["supertrend"]     = st[f"SUPERT_7_3.0"]
    df["supertrend_dir"] = st[f"SUPERTd_7_3.0"]

    # VWAP (intraday only — resets per day)
    if "volume" in df.columns:
        df["vwap"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"])

    # Volume MA
    df["vol_ma20"] = ta.sma(df["volume"], length=20)

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# INTRADAY STRATEGY  (5-min chart)
# ═══════════════════════════════════════════════════════════════════════════════

def intraday_signal(df5: pd.DataFrame) -> dict:
    """
    Multi-filter intraday strategy:
      1. EMA 9/21 crossover (entry trigger)
      2. Supertrend direction confirmation
      3. RSI momentum filter (35–65 zone gives momentum room)
      4. Volume > 1.5× 20-bar average (smart money confirmation)
      5. Price vs VWAP (bias filter)
      6. MACD histogram direction
    """
    df  = df5.copy().dropna(subset=["ema9","ema21","rsi","supertrend_dir","vwap"])
    row = df.iloc[-1]
    prev= df.iloc[-2]

    price   = float(row["close"])
    atr     = float(row["atr"])
    vol     = float(row["volume"])
    vol_avg = float(row["vol_ma20"])

    # ── Score-based signal ──
    bull_score = 0
    bear_score = 0

    # 1. EMA cross
    if prev["ema9"] < prev["ema21"] and row["ema9"] > row["ema21"]:
        bull_score += 3        # fresh bullish cross = strong
    elif row["ema9"] > row["ema21"]:
        bull_score += 1
    if prev["ema9"] > prev["ema21"] and row["ema9"] < row["ema21"]:
        bear_score += 3
    elif row["ema9"] < row["ema21"]:
        bear_score += 1

    # 2. Supertrend
    if row["supertrend_dir"] == 1:
        bull_score += 2
    else:
        bear_score += 2

    # 3. RSI
    rsi = float(row["rsi"])
    if 50 < rsi < 75:
        bull_score += 2
    elif 25 < rsi < 50:
        bear_score += 2
    elif rsi >= 75:
        bear_score += 1          # overbought → fade
    elif rsi <= 25:
        bull_score += 1          # oversold → potential reversal

    # 4. Volume spike
    vol_ratio = vol / vol_avg if vol_avg > 0 else 1
    if vol_ratio >= 1.5:
        if bull_score > bear_score:
            bull_score += 2
        else:
            bear_score += 2

    # 5. VWAP bias
    vwap = float(row["vwap"])
    if price > vwap:
        bull_score += 1
    else:
        bear_score += 1

    # 6. MACD histogram
    if float(row["macd_hist"]) > 0 and float(row["macd_hist"]) > float(prev["macd_hist"]):
        bull_score += 1
    elif float(row["macd_hist"]) < 0 and float(row["macd_hist"]) < float(prev["macd_hist"]):
        bear_score += 1

    # ── Decision  (need score ≥ 7 to fire) ──
    MIN_SCORE = 7
    if bull_score >= MIN_SCORE and bull_score > bear_score + 2:
        direction  = "BUY"
        sl         = round(price - 1.5 * atr, 2)
        target1    = round(price + 2.0 * atr, 2)
        target2    = round(price + 3.5 * atr, 2)
        confidence = min(100, int((bull_score / 12) * 100))
    elif bear_score >= MIN_SCORE and bear_score > bull_score + 2:
        direction  = "SELL_SHORT"
        sl         = round(price + 1.5 * atr, 2)
        target1    = round(price - 2.0 * atr, 2)
        target2    = round(price - 3.5 * atr, 2)
        confidence = min(100, int((bear_score / 12) * 100))
    else:
        direction  = "HOLD"
        sl         = target1 = target2 = 0.0
        confidence = 0

    # ── Position sizing (Risk-based) ──
    qty = 0
    capital_allocated = 0
    risk_per_share = abs(price - sl) if sl else 0
    if direction != "HOLD" and risk_per_share > 0:
        risk_amount = CAPITAL * RISK_PCT
        qty = int(risk_amount / risk_per_share)
        capital_allocated = round(qty * price, 2)
        max_cap = CAPITAL * MAX_INTRADAY_LOTS
        if capital_allocated > max_cap:
            qty = int(max_cap / price)
            capital_allocated = round(qty * price, 2)

    return {
        "type"             : "INTRADAY",
        "signal"           : direction,
        "price"            : round(price, 2),
        "sl"               : sl,
        "target1"          : target1,
        "target2"          : target2,
        "qty"              : qty,
        "capital_deployed" : capital_allocated,
        "confidence_pct"   : confidence,
        "rsi"              : round(rsi, 1),
        "vwap"             : round(vwap, 2),
        "vol_ratio"        : round(vol_ratio, 2),
        "atr"              : round(atr, 2),
        "bull_score"       : bull_score,
        "bear_score"       : bear_score,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SWING STRATEGY  (daily chart)
# ═══════════════════════════════════════════════════════════════════════════════

def swing_signal(dfd: pd.DataFrame) -> dict:
    """
    Swing trade strategy (2–15 day hold):
      1. Price above/below EMA50 (primary trend)
      2. EMA21 vs EMA50 slope
      3. MACD crossover on daily
      4. RSI pullback entries (50–60 for long; 40–50 for short)
      5. Bollinger Band squeeze (low ATR/BB width = breakout upcoming)
      6. 52-week range context
    """
    df   = dfd.copy().dropna(subset=["ema21","ema50","rsi","macd"])
    row  = df.iloc[-1]
    prev = df.iloc[-2]

    price   = float(row["close"])
    atr     = float(row["atr"])

    bull_score = 0
    bear_score = 0

    # 1. EMA50 trend
    if price > float(row["ema50"]):
        bull_score += 2
    else:
        bear_score += 2

    # 2. EMA21 > EMA50 (golden alignment)
    if float(row["ema21"]) > float(row["ema50"]):
        bull_score += 2
    else:
        bear_score += 2

    # 3. MACD crossover
    if prev["macd"] < prev["macd_signal"] and row["macd"] > row["macd_signal"]:
        bull_score += 3
    elif row["macd"] > row["macd_signal"]:
        bull_score += 1
    if prev["macd"] > prev["macd_signal"] and row["macd"] < row["macd_signal"]:
        bear_score += 3
    elif row["macd"] < row["macd_signal"]:
        bear_score += 1

    # 4. RSI zone
    rsi = float(row["rsi"])
    if 50 <= rsi <= 65:
        bull_score += 2          # momentum without being overbought
    elif 35 <= rsi < 50:
        bear_score += 2
    elif rsi > 70:
        bear_score += 1
    elif rsi < 30:
        bull_score += 1

    # 5. Supertrend
    if row["supertrend_dir"] == 1:
        bull_score += 2
    else:
        bear_score += 2

    # 6. 52-week range
    high52 = float(df["high"].rolling(252).max().iloc[-1])
    low52  = float(df["low"].rolling(252).min().iloc[-1])
    range52= high52 - low52
    pos52  = (price - low52) / range52 if range52 > 0 else 0.5
    if 0.5 <= pos52 <= 0.85:
        bull_score += 1          # mid-upper range — uptrend in motion
    elif pos52 < 0.3:
        bull_score += 1          # deep value

    # ── Decision  (score ≥ 8 for swing) ──
    MIN_SCORE = 8
    if bull_score >= MIN_SCORE and bull_score > bear_score + 2:
        direction = "BUY"
        sl        = round(price - 2.0 * atr, 2)
        target1   = round(price + 3.0 * atr, 2)
        target2   = round(price + 5.0 * atr, 2)
        confidence= min(100, int((bull_score / 14) * 100))
    elif bear_score >= MIN_SCORE and bear_score > bull_score + 2:
        direction = "SELL_SHORT"
        sl        = round(price + 2.0 * atr, 2)
        target1   = round(price - 3.0 * atr, 2)
        target2   = round(price - 5.0 * atr, 2)
        confidence= min(100, int((bear_score / 14) * 100))
    else:
        direction = "HOLD"
        sl = target1 = target2 = 0.0
        confidence = 0

    # ── Position sizing ──
    qty = 0
    capital_allocated = 0
    risk_per_share = abs(price - sl) if sl else 0
    if direction != "HOLD" and risk_per_share > 0:
        risk_amount = CAPITAL * RISK_PCT
        qty = int(risk_amount / risk_per_share)
        capital_allocated = round(qty * price, 2)
        max_cap = CAPITAL * MAX_SWING_LOTS
        if capital_allocated > max_cap:
            qty = int(max_cap / price)
            capital_allocated = round(qty * price, 2)

    # Hold period estimate
    hold_days = "2–5 days" if abs(bull_score - bear_score) <= 4 else "5–15 days"

    return {
        "type"             : "SWING",
        "signal"           : direction,
        "price"            : round(price, 2),
        "sl"               : sl,
        "target1"          : target1,
        "target2"          : target2,
        "qty"              : qty,
        "capital_deployed" : capital_allocated,
        "confidence_pct"   : confidence,
        "rsi"              : round(rsi, 1),
        "pos_52w_pct"      : round(pos52 * 100, 1),
        "atr"              : round(atr, 2),
        "hold_period"      : hold_days,
        "bull_score"       : bull_score,
        "bear_score"       : bear_score,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TELEGRAM
# ═══════════════════════════════════════════════════════════════════════════════

def format_telegram_message(intra: dict, swing: dict) -> str:
    ts   = datetime.now(IST).strftime("%d-%b-%Y %H:%M IST")
    emoji_map = {"BUY": "🟢", "SELL_SHORT": "🔴", "HOLD": "⚪"}

    def fmt_block(s: dict) -> str:
        e = emoji_map.get(s["signal"], "⚪")
        rr = 0
        if s["sl"] and s["target1"] and s["price"]:
            risk   = abs(s["price"] - s["sl"])
            reward = abs(s["target1"] - s["price"])
            rr     = round(reward / risk, 2) if risk > 0 else 0

        if s["signal"] != "HOLD":
            cap = f"₹{s['capital_deployed']:,}" if isinstance(s.get('capital_deployed'), (int, float)) else "—"
            lines = [
                f"{e} *{s['signal']}*  |  LTP: ₹{s['price']}",
                f"  📉 SL: ₹{s['sl']}   📈 T1: ₹{s['target1']}   🎯 T2: ₹{s['target2']}",
                f"  Qty: {s['qty']} shares  |  Capital: {cap}",
                f"  R:R = 1:{rr}  |  Confidence: {s['confidence_pct']}%",
                f"  RSI: {s['rsi']}  |  ATR: ₹{s['atr']}",
            ]
            if s.get("vol_ratio"):
                lines.append(f"  Volume ratio: {s['vol_ratio']}× avg")
            if s.get("vwap"):
                lines.append(f"  VWAP: ₹{s['vwap']}")
            if s.get("hold_period"):
                lines.append(f"  Hold: {s['hold_period']}")
            if s.get("pos_52w_pct"):
                lines.append(f"  52W position: {s['pos_52w_pct']}%")
        else:
            lines = [f"{e} *HOLD / NO TRADE*  |  LTP: ₹{s['price']}",
                     f"  RSI: {s['rsi']}  |  Score B:{s['bull_score']} S:{s['bear_score']}"]

        return "\n".join(lines)

    msg = (
        f"📊 *NMDC (NSE) Signal Report*\n"
        f"🕐 {ts}\n"
        f"💰 Capital: ₹{CAPITAL:,}  |  Risk/Trade: {RISK_PCT*100}%\n"
        f"{'─'*35}\n\n"
        f"⚡ *INTRADAY (5-min)*\n{fmt_block(intra)}\n\n"
        f"📅 *SWING TRADE (Daily)*\n{fmt_block(swing)}\n\n"
        f"{'─'*35}\n"
        f"⚠️ _Not SEBI advice. Trade at your own risk._"
    )
    return msg


def send_telegram(message: str) -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        log.warning("Telegram credentials not set — skipping notification.")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

    def _send():
        resp = requests.post(url, json={
            "chat_id"   : TELEGRAM_CHAT,
            "text"      : message,
            "parse_mode": "Markdown"
        }, timeout=15)
        resp.raise_for_status()
        return True

    try:
        return retry(_send, attempts=3, delay=5)
    except Exception as e:
        log.error(f"Telegram send failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# CSV LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

def log_to_csv(intra: dict, swing: dict):
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    ts = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

    fieldnames = [
        "timestamp", "trade_type", "signal", "price",
        "sl", "target1", "target2", "qty", "capital_deployed",
        "confidence_pct", "rsi", "atr", "bull_score", "bear_score",
        "extra_info"
    ]

    write_header = not os.path.exists(CSV_PATH)

    def _write():
        with open(CSV_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()

            for sig in [intra, swing]:
                extra = {}
                if sig["type"] == "INTRADAY":
                    extra = {"vol_ratio": sig.get("vol_ratio"), "vwap": sig.get("vwap")}
                else:
                    extra = {"pos_52w_pct": sig.get("pos_52w_pct"), "hold_period": sig.get("hold_period")}

                writer.writerow({
                    "timestamp"        : ts,
                    "trade_type"       : sig["type"],
                    "signal"           : sig["signal"],
                    "price"            : sig["price"],
                    "sl"               : sig["sl"],
                    "target1"          : sig["target1"],
                    "target2"          : sig["target2"],
                    "qty"              : sig["qty"],
                    "capital_deployed" : sig["capital_deployed"],
                    "confidence_pct"   : sig["confidence_pct"],
                    "rsi"              : sig["rsi"],
                    "atr"              : sig["atr"],
                    "bull_score"       : sig["bull_score"],
                    "bear_score"       : sig["bear_score"],
                    "extra_info"       : str(extra),
                })

    retry(_write, attempts=3, delay=2)
    log.info(f"Logged to CSV: {CSV_PATH}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def run():
    log.info("=" * 60)
    log.info(f"NMDC Analyzer starting — {datetime.now(IST).strftime('%d-%b-%Y %H:%M IST')}")

    if not is_market_open():
        log.info("Market is CLOSED. Exiting.")
        return

    # 1. Fetch data
    log.info("Fetching intraday data (5-min)…")
    df5  = fetch_intraday()
    log.info(f"  → {len(df5)} bars fetched")

    log.info("Fetching daily data…")
    dfd  = fetch_daily()
    log.info(f"  → {len(dfd)} bars fetched")

    # 2. Add indicators
    log.info("Computing indicators…")
    df5  = add_indicators(df5)
    dfd  = add_indicators(dfd)

    # 3. Generate signals
    log.info("Running intraday strategy…")
    intra = intraday_signal(df5)
    log.info(f"  → Intraday: {intra['signal']} @ ₹{intra['price']}  "
             f"(conf {intra['confidence_pct']}%)")

    log.info("Running swing strategy…")
    swing = swing_signal(dfd)
    log.info(f"  → Swing:    {swing['signal']} @ ₹{swing['price']}  "
             f"(conf {swing['confidence_pct']}%)")

    # 4. Log to CSV
    log.info("Writing to CSV…")
    log_to_csv(intra, swing)

    # 5. Telegram
    log.info("Sending Telegram alert…")
    msg = format_telegram_message(intra, swing)
    sent = send_telegram(msg)
    log.info(f"  → Telegram: {'✓ sent' if sent else '✗ skipped'}")

    log.info("Run complete.")
    log.info("=" * 60)


if __name__ == "__main__":
    run()
