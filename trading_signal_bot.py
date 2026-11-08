"""
Mobile-friendly trading signal dashboard (research/education only).

Install:
    pip install streamlit yfinance pandas numpy plotly scikit-learn requests

Run:
    streamlit run trading_signal_bot.py

REAL quotes: Yahoo Finance via yfinance (may be delayed/unavailable).
OTC quotes: configure OTC_OHLCV_URL to a broker/data-provider endpoint that
returns JSON rows with timestamp, open, high, low, close, volume. Optional
OTC_API_KEY is sent as a Bearer token. The app deliberately does NOT invent
OTC prices or signals when no genuine feed is configured.

This is decision support, not financial advice. No model can guarantee accuracy.
Paper-test signals before risking money; short-expiry products are especially risky.
"""

from __future__ import annotations

import os
import time
import warnings
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Trading Signal Lab", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

REAL_ASSETS = {
    "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "JPY=X",
    "AUD/USD": "AUDUSD=X", "USD/CAD": "CAD=X", "USD/CHF": "CHF=X",
    "NZD/USD": "NZDUSD=X", "EUR/GBP": "EURGBP=X", "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X", "BTC/USD": "BTC-USD", "ETH/USD": "ETH-USD",
    "S&P 500": "^GSPC", "NASDAQ 100": "^NDX", "Gold": "GC=F",
}
# OTC instruments are provider-specific. These names are selectable only when
# a genuine provider feed is configured; symbols are passed to that provider.
OTC_ASSETS = [
    "EUR/USD OTC", "GBP/USD OTC", "USD/JPY OTC", "AUD/USD OTC", "USD/CAD OTC",
    "USD/CHF OTC", "NZD/USD OTC", "EUR/GBP OTC", "EUR/JPY OTC", "GBP/JPY OTC",
    "BTC/USD OTC", "ETH/USD OTC", "Gold OTC", "Silver OTC", "Brent Oil OTC",
]
EXPIRIES = (1, 2, 5)

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    down = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add 15 common indicators; values are calculated from OHLCV bars."""
    x = df.copy()
    c, h, l, v = x.close, x.high, x.low, x.volume
    x["ema_9"] = ema(c, 9)
    x["ema_21"] = ema(c, 21)
    x["ema_50"] = ema(c, 50)
    x["sma_20"] = c.rolling(20).mean()
    x["rsi_14"] = rsi(c)
    macd = ema(c, 12) - ema(c, 26)
    x["macd"] = macd
    x["macd_signal"] = ema(macd, 9)
    x["macd_hist"] = macd - x.macd_signal
    mid = c.rolling(20).mean()
    sd = c.rolling(20).std()
    x["bb_upper"] = mid + 2 * sd
    x["bb_lower"] = mid - 2 * sd
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    x["atr_14"] = tr.rolling(14).mean()
    up_move, down_move = h.diff(), -l.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    atr_s = tr.ewm(alpha=1/14, adjust=False).mean().replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1/14, adjust=False).mean() / atr_s
    minus_di = 100 * minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr_s
    dx = 100 * (plus_di-minus_di).abs() / (plus_di+minus_di).replace(0, np.nan)
    x["adx_14"] = dx.ewm(alpha=1/14, adjust=False).mean()
    x["stoch_k"] = 100 * (c-l.rolling(14).min()) / (h.rolling(14).max()-l.rolling(14).min()).replace(0, np.nan)
    x["stoch_d"] = x.stoch_k.rolling(3).mean()
    x["cci_20"] = (c-mid) / (0.015 * c.rolling(20).apply(
        lambda a: np.mean(np.abs(a-np.mean(a))), raw=True).replace(0, np.nan))
    x["williams_r"] = -100 * (h.rolling(14).max()-c) / (h.rolling(14).max()-l.rolling(14).min()).replace(0, np.nan)
    x["obv"] = (np.sign(c.diff()).fillna(0) * v.fillna(0)).cumsum()
    x["roc_10"] = c.pct_change(10) * 100
    return x

def add_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Detect 22 basic candle/price-action patterns (heuristic, not guarantees)."""
    x = df.copy()
    o, h, l, c = x.open, x.high, x.low, x.close
    body = (c-o).abs()
    rng = (h-l).replace(0, np.nan)
    upper, lower = h-np.maximum(o,c), np.minimum(o,c)-l
    bull, bear = c > o, c < o
    prev_o, prev_c = o.shift(), c.shift()
    prev_body = (prev_c-prev_o).abs()
    p = {}
    p["bullish_engulfing"] = bull & (prev_c < prev_o) & (o <= prev_c) & (c >= prev_o)
    p["bearish_engulfing"] = bear & (prev_c > prev_o) & (o >= prev_c) & (c <= prev_o)
    p["hammer"] = (lower >= 2*body) & (upper <= body) & (body/rng < .4)
    p["hanging_man"] = p["hammer"]
    p["shooting_star"] = (upper >= 2*body) & (lower <= body) & (body/rng < .4)
    p["inverted_hammer"] = p["shooting_star"]
    p["doji"] = body/rng <= .1
    p["dragonfly_doji"] = (body/rng <= .1) & (upper/rng <= .1)
    p["gravestone_doji"] = (body/rng <= .1) & (lower/rng <= .1)
    p["marubozu_bull"] = bull & (upper/rng < .05) & (lower/rng < .05)
    p["marubozu_bear"] = bear & (upper/rng < .05) & (lower/rng < .05)
    p["piercing_line"] = bull & (prev_c < prev_o) & (c > (prev_o+prev_c)/2) & (o < prev_c)
    p["dark_cloud_cover"] = bear & (prev_c > prev_o) & (c < (prev_o+prev_c)/2) & (o > prev_c)
    p["morning_star"] = (prev_c < prev_o) & (body.shift(1) < prev_body*.5) & bull & (c > (prev_o+prev_c)/2)
    p["evening_star"] = (prev_c > prev_o) & (body.shift(1) < prev_body*.5) & bear & (c < (prev_o+prev_c)/2)
    p["tweezer_bottom"] = (l-l.shift()).abs() <= rng*.05
    p["tweezer_top"] = (h-h.shift()).abs() <= rng*.05
    p["inside_bar"] = (h < h.shift()) & (l > l.shift())
    p["outside_bar"] = (h > h.shift()) & (l < l.shift())
    p["three_white_soldiers"] = bull & bull.shift() & bull.shift(2) & (c > c.shift()) & (c.shift() > c.shift(2))
    p["three_black_crows"] = bear & bear.shift() & bear.shift(2) & (c < c.shift()) & (c.shift() < c.shift(2))
    p["bullish_pin_bar"] = (lower/rng > .6) & (body/rng < .3)
    p["bearish_pin_bar"] = (upper/rng > .6) & (body/rng < .3)
    for name, values in p.items():
        x["pat_" + name] = pd.Series(values, index=x.index).fillna(False).astype(int)
    return x

FEATURES = [
    "ema_9","ema_21","ema_50","sma_20","rsi_14","macd","macd_signal",
    "macd_hist","atr_14","adx_14","stoch_k","stoch_d","cci_20",
    "williams_r","roc_10","obv",
]

def load_real(symbol: str, interval: str = "1m", period: str = "5d") -> pd.DataFrame:
    raw = yf.download(symbol, period=period, interval=interval, progress=False,
                      auto_adjust=True, threads=False)
    if raw is None or raw.empty:
        raise RuntimeError("No real-market bars returned by Yahoo Finance.")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.rename(columns={k: k.lower() for k in raw.columns})
    required = ["open","high","low","close"]
    if not all(k in raw for k in required):
        raise RuntimeError("Price feed did not contain OHLC columns.")
    if "volume" not in raw:
        raw["volume"] = 0
    raw = raw[required + ["volume"]].dropna()
    raw.index = pd.to_datetime(raw.index)
    return raw.tail(1200)

def load_otc(asset: str) -> pd.DataFrame:
    endpoint = os.getenv("OTC_OHLCV_URL", "").strip()
    if not endpoint:
        raise RuntimeError("OTC feed not configured. Set OTC_OHLCV_URL to your licensed provider endpoint.")
    headers = {}
    token = os.getenv("OTC_API_KEY", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = requests.get(endpoint, params={"symbol": asset, "interval": "1m", "limit": 1200},
                            headers=headers, timeout=15)
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("data", payload) if isinstance(payload, dict) else payload
    df = pd.DataFrame(rows)
    df.columns = [str(c).lower() for c in df.columns]
    if "timestamp" not in df:
        raise RuntimeError("OTC API must return timestamp, open, high, low, close, volume.")
    df["timestamp"] = pd.to_datetime(df.timestamp, utc=True, errors="coerce")
    df = df.set_index("timestamp")
    for col in ["open","high","low","close","volume"]:
        if col not in df:
            df[col] = 0 if col == "volume" else np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["open","high","low","close","volume"]].dropna().tail(1200)

def build_ai():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(n_estimators=250, max_depth=5,
                                         min_samples_leaf=8, class_weight="balanced",
                                         random_state=42))
    ])

def ai_signal(df: pd.DataFrame, expiry: int):
    """Chronological holdout evaluation, then fit on available history."""
    x = df[FEATURES].replace([np.inf,-np.inf], np.nan)
    # Direction after N bars; exclude the final N bars from training.
    future = df.close.shift(-expiry)
    y = (future > df.close).astype(int)
    valid = future.notna() & x.notna().sum(axis=1).ge(5)
    indices = np.flatnonzero(valid.to_numpy())
    if len(indices) < 180:
        return "WAIT", 0.0, np.nan
    cut = int(len(indices)*.8)
    train_i, test_i = indices[:cut], indices[cut:]
    model = build_ai()
    model.fit(x.iloc[train_i], y.iloc[train_i])
    accuracy = float(model.score(x.iloc[test_i], y.iloc[test_i])) if len(test_i) else np.nan
    model.fit(x.iloc[indices], y.iloc[indices])
    prob_up = float(model.predict_proba(x.iloc[[-1]])[0][list(model.named_steps["model"].classes_).index(1)])
    # Conservative gate: probabilities are model estimates, NOT calibrated
    # win probabilities. Require 90% for 1-minute setups and trend agreement.
    latest = df.iloc[-1]
    trend_up = bool(latest.ema_9 > latest.ema_21 > latest.ema_50)
    trend_down = bool(latest.ema_9 < latest.ema_21 < latest.ema_50)
    min_confidence = {1: 0.90, 2: 0.82, 5: 0.78}.get(int(expiry), 0.90)
    if prob_up >= min_confidence and trend_up:
        return "CALL", prob_up, accuracy
    if prob_up <= (1-min_confidence) and trend_down:
        return "PUT", 1-prob_up, accuracy
    return "WAIT", max(prob_up, 1-prob_up), accuracy

def rule_score(df: pd.DataFrame):
    r = df.iloc[-1]
    score = 0
    score += 1 if r.ema_9 > r.ema_21 > r.ema_50 else -1 if r.ema_9 < r.ema_21 < r.ema_50 else 0
    score += 1 if r.macd > r.macd_signal else -1
    score += 1 if r.rsi_14 > 55 else -1 if r.rsi_14 < 45 else 0
    score += 1 if r.close > r.sma_20 else -1
    score += 1 if r.stoch_k > r.stoch_d else -1
    pattern_cols = [c for c in df.columns if c.startswith("pat_")]
    bullish = sum(int(r[c]) for c in pattern_cols if any(k in c for k in
                   ["bullish","hammer","morning","piercing","three_white","tweezer_bottom"]))
    bearish = sum(int(r[c]) for c in pattern_cols if any(k in c for k in
                   ["bearish","shooting","evening","dark_cloud","three_black","tweezer_top"]))
    score += int(np.sign(bullish-bearish))
    return score

def make_chart(df: pd.DataFrame, asset: str):
    fig = go.Figure(go.Candlestick(
        x=df.index, open=df.open, high=df.high, low=df.low, close=df.close,
        increasing_line_color="#16a34a", decreasing_line_color="#dc2626",
        increasing_fillcolor="#16a34a", decreasing_fillcolor="#dc2626",
        name=asset))
    fig.add_trace(go.Scatter(x=df.index, y=df.ema_9, name="EMA 9",
                             line=dict(width=1, color="#2563eb")))
    fig.add_trace(go.Scatter(x=df.index, y=df.ema_21, name="EMA 21",
                             line=dict(width=1, color="#f59e0b")))
    fig.update_layout(height=430, margin=dict(l=10,r=10,t=35,b=10),
                      xaxis_rangeslider_visible=False, template="plotly_dark",
                      title=f"{asset} · 1-minute candles")
    return fig

def speak_signal(asset: str, stamp: str, direction: str):
    # Browser speech synthesis speaks the asset, time and Call/Put direction.
    safe = f"{asset}, {stamp}, {direction}"
    st.components.v1.html(f"""
    <script>
    const msg = new SpeechSynthesisUtterance({safe!r});
    msg.rate = 0.92; msg.pitch = 1.0;
    window.speechSynthesis.cancel(); window.speechSynthesis.speak(msg);
    </script>""", height=0)

st.markdown("""
<style>
.block-container {padding-top:1rem; padding-bottom:2rem; max-width:1200px}
div.stButton > button {min-height:3rem; width:100%; border-radius:12px; font-weight:700}
.signal-call {background:#14532d;padding:14px;border-radius:12px;color:white;font-size:1.35rem}
.signal-put {background:#7f1d1d;padding:14px;border-radius:12px;color:white;font-size:1.35rem}
.signal-wait {background:#374151;padding:14px;border-radius:12px;color:white;font-size:1.35rem}
</style>
""", unsafe_allow_html=True)

st.title("📈 Trading Signal Lab")
st.caption("Mobile-friendly research dashboard · real-feed mode and provider-connected OTC mode")

with st.sidebar:
    st.header("Signal controls")
    market_type = st.radio("Market", ["Real assets", "OTC assets"])
    assets = REAL_ASSETS if market_type == "Real assets" else {a:a for a in OTC_ASSETS}
    asset = st.selectbox("Asset", list(assets.keys()))
    expiry = st.select_slider("Signal horizon (minutes)", options=list(EXPIRIES), value=1)
    refresh_seconds = st.selectbox("Auto-refresh", [15, 30, 60], index=1)
    auto_mode = st.toggle("Automatic signal generation", value=False)
    sound_on = st.toggle("Auditory alerts", value=True)
    st.caption("Start/stop controls below. Auto-refresh reruns the analysis; it does not place trades.")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("▶ Start signals", type="primary"):
        st.session_state.running = True
with col2:
    if st.button("■ Stop signals"):
        st.session_state.running = False
with col3:
    if st.button("↻ Analyze now"):
        st.session_state.force_refresh = True

if "running" not in st.session_state:
    st.session_state.running = False
if "last_alert_key" not in st.session_state:
    st.session_state.last_alert_key = None

active = st.session_state.running or auto_mode
if not active:
    st.info("Signal generation is paused. Press **Start signals** or enable automatic generation.")
try:
    with st.spinner(f"Loading {asset} market data…"):
        if market_type == "Real assets":
            raw = load_real(assets[asset])
        else:
            raw = load_otc(asset)
        data = add_patterns(add_indicators(raw)).dropna(subset=["close"])
    if len(data) < 100:
        st.warning("Not enough candles for a stable analysis. Try again later.")
    else:
        direction, confidence, holdout = ai_signal(data, expiry)
        score = rule_score(data)
        # Conservative confirmation: AI direction must agree with technical score.
        if direction == "CALL" and score < 2:
            direction = "WAIT"
        elif direction == "PUT" and score > -2:
            direction = "WAIT"
        stamp = datetime.now().astimezone().strftime("%H:%M:%S")
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("AI direction", direction)
        k2.metric("Model score (not win odds)", f"{confidence*100:.1f}%")
        k3.metric("Technical score", f"{score:+d} / 6")
        k4.metric("Historical holdout accuracy", "N/A" if np.isnan(holdout) else f"{holdout*100:.1f}%")
        css = "signal-call" if direction == "CALL" else "signal-put" if direction == "PUT" else "signal-wait"
        st.markdown(f'<div class="{css}">{asset} · {direction} · {expiry} min · {stamp}</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(make_chart(data.tail(180), asset), use_container_width=True)
        latest = data.iloc[-1]
        with st.expander("15 technical indicators"):
            vals = {name: latest.get(name, np.nan) for name in
                    ["ema_9","ema_21","ema_50","sma_20","rsi_14","macd","macd_signal",
                     "macd_hist","bb_upper","bb_lower","atr_14","adx_14","stoch_k",
                     "stoch_d","cci_20","williams_r","obv","roc_10"]}
            st.dataframe(pd.DataFrame(vals.items(), columns=["Indicator","Latest value"]),
                         use_container_width=True, hide_index=True)
        with st.expander("22 price-action pattern checks"):
            pats = [c for c in data.columns if c.startswith("pat_")]
            active_patterns = [p.replace("pat_","").replace("_"," ").title()
                               for p in pats if int(latest[p]) == 1]
            st.write(", ".join(active_patterns) if active_patterns else "No listed pattern detected on the latest candle.")
        alert_key = (asset, str(data.index[-1]), direction, expiry)
        if active and sound_on and direction in ("CALL","PUT") and alert_key != st.session_state.last_alert_key:
            speak_signal(asset, stamp, direction)
            st.session_state.last_alert_key = alert_key
        st.caption("Signals are probabilistic estimates, not verified trade instructions. Holdout accuracy is a simple historical split and is not a guarantee of future performance.")
except Exception as exc:
    st.error(f"Market data / analysis unavailable: {exc}")
    if market_type == "OTC assets":
        st.info("Connect a licensed OTC OHLCV provider using OTC_OHLCV_URL and, if required, OTC_API_KEY. OTC quotes are broker-specific; this app will not fabricate them.")

st.divider()
st.caption("No broker execution is included. The displayed model score is not a calibrated win probability. Confirm feed quality, candle timestamps, spreads, expiry rules and local regulations before any live use. Never treat a high score as a guarantee.")

# Lightweight timed refresh. Streamlit reruns the app; no background trade loop.
if active:
    time.sleep(refresh_seconds)
    st.rerun()
