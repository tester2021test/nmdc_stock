"""
NMDC Stock Analyzer — Intraday & Swing Trade Strategy
Uses: yfinance + ta (technical-analysis library, pandas 3.x compatible)
Capital: Rs 1,00,000  |  Exchange: NSE (NMDC.NS)
"""

import os, time, logging, csv
from datetime import datetime
from zoneinfo import ZoneInfo

import yfinance as yf
import pandas as pd
import ta
import requests

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger("NMDC")

# Config
TICKER            = "NMDC.NS"
CAPITAL           = 100_000
RISK_PCT          = 0.015
MAX_INTRADAY_CAP  = 0.40
MAX_SWING_CAP     = 0.60
IST               = ZoneInfo("Asia/Kolkata")
MARKET_OPEN       = (9, 15)
MARKET_CLOSE      = (15, 30)

TELEGRAM_TOKEN    = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT     = os.environ.get("TELEGRAM_CHAT_ID", "")
CSV_PATH          = "results/nmdc_signals.csv"

RETRY_ATTEMPTS    = 5
RETRY_DELAY       = 8

# ─── Utilities ────────────────────────────────────────────────────────────────

def is_market_open() -> bool:
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    t = now.hour * 60 + now.minute
    return (MARKET_OPEN[0]*60+MARKET_OPEN[1]) <= t <= (MARKET_CLOSE[0]*60+MARKET_CLOSE[1])

def retry(fn, attempts=RETRY_ATTEMPTS, delay=RETRY_DELAY):
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            wait = delay * attempt
            log.warning(f"Attempt {attempt}/{attempts} failed: {exc}. Retry in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"All {attempts} attempts failed. Last: {last_exc}") from last_exc

def sf(val, default=0.0) -> float:
    try:
        return float(val)
    except Exception:
        return default

# ─── Data Fetch ───────────────────────────────────────────────────────────────

def normalise(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]
    return df.dropna()

def fetch_intraday() -> pd.DataFrame:
    def _f():
        df = yf.download(TICKER, period="5d", interval="5m", auto_adjust=True, progress=False)
        if df.empty:
            raise ValueError("Empty intraday data")
        return normalise(df)
    return retry(_f)

def fetch_daily() -> pd.DataFrame:
    def _f():
        df = yf.download(TICKER, period="250d", interval="1d", auto_adjust=True, progress=False)
        if df.empty:
            raise ValueError("Empty daily data")
        return normalise(df)
    return retry(_f)

# ─── Indicators ───────────────────────────────────────────────────────────────

def add_supertrend(df: pd.DataFrame, period=7, multiplier=3.0) -> pd.DataFrame:
    atr   = df["atr"]
    hl2   = (df["high"] + df["low"]) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr

    st_dir  = [1] * len(df)
    st_line = [0.0] * len(df)

    for i in range(1, len(df)):
        prev_dir = st_dir[i-1]
        if prev_dir == 1:
            f_lower = max(lower.iloc[i], st_line[i-1])
            f_upper = upper.iloc[i]
        else:
            f_upper = min(upper.iloc[i], st_line[i-1])
            f_lower = lower.iloc[i]

        curr = df["close"].iloc[i]
        if prev_dir == 1:
            if curr > f_upper:
                st_dir[i]  = -1
                st_line[i] = f_upper
            else:
                st_dir[i]  = 1
                st_line[i] = f_lower
        else:
            if curr < f_lower:
                st_dir[i]  = 1
                st_line[i] = f_lower
            else:
                st_dir[i]  = -1
                st_line[i] = f_upper

    df["supertrend"]     = st_line
    df["supertrend_dir"] = st_dir
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df    = df.copy()
    c, h, l = df["close"], df["high"], df["low"]
    vol   = df.get("volume", pd.Series(0, index=df.index))

    df["ema9"]   = ta.trend.EMAIndicator(c, window=9).ema_indicator()
    df["ema21"]  = ta.trend.EMAIndicator(c, window=21).ema_indicator()
    df["ema50"]  = ta.trend.EMAIndicator(c, window=50).ema_indicator()
    df["ema200"] = ta.trend.EMAIndicator(c, window=200).ema_indicator()
    df["rsi"]    = ta.momentum.RSIIndicator(c, window=14).rsi()

    macd = ta.trend.MACD(c, window_fast=12, window_slow=26, window_sign=9)
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"]   = macd.macd_diff()

    bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()

    df["atr"] = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range()

    df = add_supertrend(df)

    if vol.sum() > 0:
        df["vwap"] = ta.volume.VolumeWeightedAveragePrice(h, l, c, vol, window=14).volume_weighted_average_price()
    else:
        df["vwap"] = c

    df["vol_ma20"] = vol.rolling(20).mean().replace(0, 1)
    return df

# ─── Position Sizer ───────────────────────────────────────────────────────────

def size_position(price, sl, max_cap_ratio):
    risk_ps = abs(price - sl)
    if risk_ps <= 0:
        return 0, 0.0
    qty = int((CAPITAL * RISK_PCT) / risk_ps)
    cap = qty * price
    max_cap = CAPITAL * max_cap_ratio
    if cap > max_cap:
        qty = int(max_cap / price)
        cap = qty * price
    return max(qty, 0), round(cap, 2)

# ─── Intraday Strategy ────────────────────────────────────────────────────────

def intraday_signal(df5: pd.DataFrame) -> dict:
    df   = df5.dropna(subset=["ema9","ema21","rsi","supertrend_dir"]).copy()
    row  = df.iloc[-1]
    prev = df.iloc[-2]

    price     = sf(row["close"])
    atr       = sf(row["atr"])
    vol       = sf(row.get("volume", 1))
    vol_avg   = sf(row.get("vol_ma20", 1))
    vwap      = sf(row.get("vwap", price))
    vol_ratio = round(vol / vol_avg, 2) if vol_avg > 0 else 1.0
    rsi       = sf(row["rsi"])

    bull = bear = 0

    # 1. EMA 9/21 cross
    if sf(prev["ema9"]) < sf(prev["ema21"]) and sf(row["ema9"]) > sf(row["ema21"]):
        bull += 3
    elif sf(row["ema9"]) > sf(row["ema21"]):
        bull += 1
    if sf(prev["ema9"]) > sf(prev["ema21"]) and sf(row["ema9"]) < sf(row["ema21"]):
        bear += 3
    elif sf(row["ema9"]) < sf(row["ema21"]):
        bear += 1

    # 2. Supertrend
    if row["supertrend_dir"] == 1: bull += 2
    else:                           bear += 2

    # 3. RSI
    if 50 < rsi < 75:    bull += 2
    elif 25 < rsi < 50:  bear += 2
    elif rsi >= 75:       bear += 1
    elif rsi <= 25:       bull += 1

    # 4. Volume spike
    if vol_ratio >= 1.5:
        if bull >= bear: bull += 2
        else:            bear += 2

    # 5. VWAP
    if price > vwap: bull += 1
    else:            bear += 1

    # 6. MACD histogram slope
    h_now  = sf(row["macd_hist"])
    h_prev = sf(prev["macd_hist"])
    if h_now > 0 and h_now > h_prev:   bull += 1
    elif h_now < 0 and h_now < h_prev: bear += 1

    MIN = 7
    if bull >= MIN and bull >= bear + 2:
        sig  = "BUY"
        sl   = round(price - 1.5*atr, 2)
        t1   = round(price + 2.0*atr, 2)
        t2   = round(price + 3.5*atr, 2)
        conf = min(100, int(bull/12*100))
    elif bear >= MIN and bear >= bull + 2:
        sig  = "SELL_SHORT"
        sl   = round(price + 1.5*atr, 2)
        t1   = round(price - 2.0*atr, 2)
        t2   = round(price - 3.5*atr, 2)
        conf = min(100, int(bear/12*100))
    else:
        sig  = "HOLD"
        sl = t1 = t2 = 0.0
        conf = 0

    qty, cap = size_position(price, sl, MAX_INTRADAY_CAP) if sig != "HOLD" else (0, 0.0)
    return dict(type="INTRADAY", signal=sig, price=price, sl=sl, target1=t1, target2=t2,
                qty=qty, capital_deployed=cap, confidence_pct=conf, rsi=round(rsi,1),
                vwap=round(vwap,2), vol_ratio=vol_ratio, atr=round(atr,2),
                bull_score=bull, bear_score=bear)

# ─── Swing Strategy ───────────────────────────────────────────────────────────

def swing_signal(dfd: pd.DataFrame) -> dict:
    df   = dfd.dropna(subset=["ema21","ema50","rsi","macd","supertrend_dir"]).copy()
    row  = df.iloc[-1]
    prev = df.iloc[-2]

    price = sf(row["close"])
    atr   = sf(row["atr"])
    rsi   = sf(row["rsi"])
    bull  = bear = 0

    # 1. Price vs EMA50
    if price > sf(row["ema50"]): bull += 2
    else:                         bear += 2

    # 2. EMA21 vs EMA50
    if sf(row["ema21"]) > sf(row["ema50"]): bull += 2
    else:                                     bear += 2

    # 3. MACD crossover
    if sf(prev["macd"]) < sf(prev["macd_signal"]) and sf(row["macd"]) > sf(row["macd_signal"]):
        bull += 3
    elif sf(row["macd"]) > sf(row["macd_signal"]):
        bull += 1
    if sf(prev["macd"]) > sf(prev["macd_signal"]) and sf(row["macd"]) < sf(row["macd_signal"]):
        bear += 3
    elif sf(row["macd"]) < sf(row["macd_signal"]):
        bear += 1

    # 4. RSI
    if 50 <= rsi <= 65:  bull += 2
    elif 35 <= rsi < 50: bear += 2
    elif rsi > 70:        bear += 1
    elif rsi < 30:        bull += 1

    # 5. Supertrend
    if row["supertrend_dir"] == 1: bull += 2
    else:                           bear += 2

    # 6. 52-week range
    h52   = df["high"].rolling(252, min_periods=50).max().iloc[-1]
    l52   = df["low"].rolling(252, min_periods=50).min().iloc[-1]
    rng52 = sf(h52 - l52)
    pos52 = (price - sf(l52)) / rng52 if rng52 > 0 else 0.5
    if 0.5 <= pos52 <= 0.85: bull += 1
    elif pos52 < 0.30:        bull += 1

    MIN = 8
    if bull >= MIN and bull >= bear + 2:
        sig  = "BUY"
        sl   = round(price - 2.0*atr, 2)
        t1   = round(price + 3.0*atr, 2)
        t2   = round(price + 5.0*atr, 2)
        conf = min(100, int(bull/14*100))
    elif bear >= MIN and bear >= bull + 2:
        sig  = "SELL_SHORT"
        sl   = round(price + 2.0*atr, 2)
        t1   = round(price - 3.0*atr, 2)
        t2   = round(price - 5.0*atr, 2)
        conf = min(100, int(bear/14*100))
    else:
        sig  = "HOLD"
        sl = t1 = t2 = 0.0
        conf = 0

    hold  = "2-5 days" if abs(bull-bear) <= 4 else "5-15 days"
    qty, cap = size_position(price, sl, MAX_SWING_CAP) if sig != "HOLD" else (0, 0.0)
    return dict(type="SWING", signal=sig, price=price, sl=sl, target1=t1, target2=t2,
                qty=qty, capital_deployed=cap, confidence_pct=conf, rsi=round(rsi,1),
                pos_52w_pct=round(pos52*100,1), atr=round(atr,2), hold_period=hold,
                bull_score=bull, bear_score=bear)

# ─── Telegram ─────────────────────────────────────────────────────────────────

def format_message(intra: dict, swing: dict) -> str:
    ts   = datetime.now(IST).strftime("%d-%b-%Y %H:%M IST")
    emap = {"BUY": "🟢", "SELL_SHORT": "🔴", "HOLD": "⚪"}

    def rr(s):
        risk = abs(s["price"] - s["sl"])
        if risk <= 0: return "-"
        return f"1:{round(abs(s['target1'] - s['price']) / risk, 1)}"

    def block(s):
        e = emap.get(s["signal"], "⚪")
        if s["signal"] == "HOLD":
            return (f"{e} *HOLD / NO TRADE*  |  LTP: Rs{s['price']}\n"
                    f"  RSI: {s['rsi']}  |  Bull: {s['bull_score']}  Bear: {s['bear_score']}")
        lines = [
            f"{e} *{s['signal']}*  |  LTP: Rs{s['price']}",
            f"  SL: Rs{s['sl']}",
            f"  T1: Rs{s['target1']}   T2: Rs{s['target2']}",
            f"  Qty: {s['qty']} shares  |  Capital: Rs{s['capital_deployed']:,}",
            f"  R:R = {rr(s)}  |  Confidence: {s['confidence_pct']}%",
            f"  RSI: {s['rsi']}  |  ATR: Rs{s['atr']}",
        ]
        if s.get("vol_ratio"):
            lines.append(f"  Volume: {s['vol_ratio']}x avg  |  VWAP: Rs{s.get('vwap','-')}")
        if s.get("hold_period"):
            lines.append(f"  Hold: {s['hold_period']}  |  52W pos: {s.get('pos_52w_pct','-')}%")
        return "\n".join(lines)

    return (
        f"NMDC (NSE) Signal Update\n"
        f"Time: {ts}\n"
        f"Capital Rs{CAPITAL:,}  |  Risk 1.5pct per trade\n"
        f"------------------------------------\n\n"
        f"INTRADAY (5-min chart)\n{block(intra)}\n\n"
        f"SWING TRADE (Daily chart)\n{block(swing)}\n\n"
        f"------------------------------------\n"
        f"Educational use only. Not SEBI advice."
    )

def send_telegram(message: str) -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        log.warning("Telegram creds missing - skipping.")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    def _send():
        r = requests.post(url, json={
            "chat_id": TELEGRAM_CHAT, "text": message
        }, timeout=15)
        r.raise_for_status()
    try:
        retry(_send, attempts=3, delay=4)
        return True
    except Exception as e:
        log.error(f"Telegram failed: {e}")
        return False

# ─── CSV Logger ───────────────────────────────────────────────────────────────

FIELDS = ["timestamp","trade_type","signal","price","sl","target1","target2",
          "qty","capital_deployed","confidence_pct","rsi","atr",
          "bull_score","bear_score","extra_info"]

def log_to_csv(intra: dict, swing: dict):
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    ts = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
    write_header = not os.path.exists(CSV_PATH)
    def _write():
        with open(CSV_PATH, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if write_header:
                w.writeheader()
            for sig in [intra, swing]:
                extra = {k: sig[k] for k in sig if k not in FIELDS+["type"]}
                w.writerow({
                    "timestamp": ts, "trade_type": sig["type"],
                    "signal": sig["signal"], "price": sig["price"],
                    "sl": sig["sl"], "target1": sig["target1"], "target2": sig["target2"],
                    "qty": sig["qty"], "capital_deployed": sig["capital_deployed"],
                    "confidence_pct": sig["confidence_pct"], "rsi": sig["rsi"],
                    "atr": sig["atr"], "bull_score": sig["bull_score"],
                    "bear_score": sig["bear_score"], "extra_info": str(extra),
                })
    retry(_write, attempts=3, delay=2)
    log.info(f"CSV updated: {CSV_PATH}")

# ─── Main ─────────────────────────────────────────────────────────────────────

def run():
    log.info("=" * 60)
    log.info(f"NMDC Analyzer - {datetime.now(IST).strftime('%d-%b-%Y %H:%M IST')}")

    if not is_market_open():
        log.info("Market CLOSED. Exiting.")
        return

    log.info("Fetching 5-min intraday data...")
    df5 = fetch_intraday()
    log.info(f"  {len(df5)} bars")

    log.info("Fetching daily data...")
    dfd = fetch_daily()
    log.info(f"  {len(dfd)} bars")

    log.info("Computing indicators...")
    df5 = add_indicators(df5)
    dfd = add_indicators(dfd)

    log.info("Running intraday strategy...")
    intra = intraday_signal(df5)
    log.info(f"  -> {intra['signal']} @ Rs{intra['price']}  conf={intra['confidence_pct']}%")

    log.info("Running swing strategy...")
    swing = swing_signal(dfd)
    log.info(f"  -> {swing['signal']} @ Rs{swing['price']}  conf={swing['confidence_pct']}%")

    log.info("Logging to CSV...")
    log_to_csv(intra, swing)

    log.info("Sending Telegram alert...")
    msg  = format_message(intra, swing)
    sent = send_telegram(msg)
    log.info(f"  -> {'sent' if sent else 'skipped'}")

    log.info("Done.")
    log.info("=" * 60)

if __name__ == "__main__":
    run()
