"""
Cryptera v3.2 - Rule-based SMC engine + Gemini AI structured JSON narration.
"""

import ccxt.async_support as ccxt
import pandas as pd
import asyncio
import requests
import json
import os
import time
import numpy as np
from collections import OrderedDict
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types

from indicators import calculate_indicators, detect_regime, detect_volatility_regime
from strategies import evaluate_strategies
from smc import build_smc_context
from price_action import build_pa_context, calculate_previous_day, get_value_area


SCHEMA_VERSION = "3.2.3"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
SNAPSHOTS_DIR = os.path.join(ROOT_DIR, "snapshots")
OI_HISTORY_PATH = os.path.join(DATA_DIR, "oi_history.json")
EVENTS_PATH = os.path.join(DATA_DIR, "events.json")
SNAPSHOT_RETENTION = 1000  # per symbol


# ---------------- utilities ---------------- #

def to_native(obj):
    """Recursively convert numpy / pandas types to native Python."""
    if isinstance(obj, (np.bool_, np.integer, np.floating)):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return float(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_native(v) for v in obj]
    return obj


# ---------------- config ---------------- #

TIMEFRAMES = ["4h", "1h", "15m"]
LIMITS = {"4h": 200, "1h": 200, "15m": 300}
WINDOW_SIZE = {"4h": 10, "1h": 20, "15m": 30}
TF_MS = {"4h": 4 * 3600_000, "1h": 3600_000, "15m": 15 * 60_000}  # bar length in ms


def get_window_dict(df: pd.DataFrame, size: int) -> dict:
    """Compressed per-TF statistical summary fed to the LLM."""
    recent = df.tail(size)
    closes = recent["close"].tolist()
    start_p = closes[0] if closes else 1.0
    end_p = closes[-1] if closes else 1.0
    price_change_pct = round(((end_p - start_p) / start_p) * 100, 2) if start_p else 0.0

    def _safe(col, default=0.0):
        if col in recent.columns and not pd.isna(recent[col].iloc[-1]):
            return round(float(recent[col].iloc[-1]), 4)
        return default

    rsi_latest = _safe("RSI", 50.0)
    vol_latest = _safe("volume", 0.0)
    vol_mean = float(recent["volume"].mean()) if "volume" in recent.columns else 1.0
    vol_relative = round(vol_latest / max(vol_mean, 0.1), 2)
    supertrend_latest = int(recent["SUPERTd_10_3.0"].iloc[-1]) if "SUPERTd_10_3.0" in recent.columns and not pd.isna(recent["SUPERTd_10_3.0"].iloc[-1]) else 0
    stochrsi_k = _safe("STOCHRSIk_14_14_3_3", 50.0)
    stochrsi_d = _safe("STOCHRSId_14_14_3_3", 50.0)
    adx_latest = _safe("ADX", 0.0)
    atr_pct_latest = _safe("ATR_Pct", 0.5)

    cvd_change = 0.0
    if "cvd" in recent.columns:
        cvd_vals = recent["cvd"].dropna().tolist()
        if len(cvd_vals) >= 2:
            cvd_change = round(cvd_vals[-1] - cvd_vals[0], 2)
    cvd_real = bool(recent["cvd_real"].iloc[-1]) if "cvd_real" in recent.columns and not recent.empty else False

    recent_5 = df.tail(5)
    close_series = [round(float(x), 2) for x in recent_5["close"].tolist()]
    rsi_series = [round(float(x), 2) for x in recent_5["RSI"].tolist()] if "RSI" in recent_5.columns else []

    return {
        "current_price": round(end_p, 4),
        "window_price_change_pct": price_change_pct,
        "rsi": rsi_latest,
        "relative_volume": vol_relative,
        "supertrend_direction": supertrend_latest,
        "stochrsi_k": stochrsi_k,
        "stochrsi_d": stochrsi_d,
        "cvd_window_delta": cvd_change,
        "cvd_is_real": cvd_real,
        "adx": adx_latest,
        "atr_percentile": atr_pct_latest,
        "close_series_last_5": close_series,
        "rsi_series_last_5": rsi_series,
    }


# ---------------- chart series (for the dashboard candlestick chart) ---------------- #

CHART_BARS = {"4h": 120, "1h": 150, "15m": 180}


def build_chart_series(data: dict) -> dict:
    """
    Compact OHLC arrays per timeframe for the frontend price chart. Attached to
    the snapshot AFTER the Gemini call so it never bloats the LLM prompt. Times
    are UNIX seconds (UTC) as lightweight-charts expects.
    """
    out = {}
    for tf, df in data.items():
        if df is None or df.empty:
            out[tf] = []
            continue
        sub = df.tail(CHART_BARS.get(tf, 150))
        times = (sub["timestamp"].astype("int64") // 10**9).tolist()
        o = sub["open"].astype(float).round(6).tolist()
        h = sub["high"].astype(float).round(6).tolist()
        l = sub["low"].astype(float).round(6).tolist()
        c = sub["close"].astype(float).round(6).tolist()
        out[tf] = [{"time": int(t), "open": oo, "high": hh, "low": ll, "close": cc}
                   for t, oo, hh, ll, cc in zip(times, o, h, l, c)]
    return out


# ---------------- caching ---------------- #

OHLCV_CACHE = OrderedDict()
OHLCV_CACHE_MAX = 50
CACHE_TTL_SECONDS = {"4h": 300, "1h": 120, "15m": 20}


def _cache_get(key):
    if key in OHLCV_CACHE:
        OHLCV_CACHE.move_to_end(key)
        return OHLCV_CACHE[key]
    return None


def _cache_put(key, value):
    OHLCV_CACHE[key] = value
    OHLCV_CACHE.move_to_end(key)
    while len(OHLCV_CACHE) > OHLCV_CACHE_MAX:
        OHLCV_CACHE.popitem(last=False)


# ---------------- market data ---------------- #

def fetch_taker_flow(symbol, tf, limit):
    """
    Fetch raw futures klines to obtain per-bar taker-buy base volume, indexed by
    open timestamp. Enables REAL aggressive CVD (buy - sell) instead of the
    candle-position proxy. Returns a DataFrame[timestamp, taker_buy_base] or None.
    """
    try:
        r = requests.get(
            "https://fapi.binance.com/fapi/v1/klines",
            params={"symbol": symbol.replace("/", ""), "interval": tf, "limit": limit},
            timeout=8,
        ).json()
        if not isinstance(r, list) or not r:
            return None
        rows = []
        for k in r:
            # [0]=openTime ... [5]=volume ... [9]=takerBuyBaseVol
            rows.append((int(k[0]), float(k[9])))
        df = pd.DataFrame(rows, columns=["timestamp", "taker_buy_base"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    except Exception as e:
        print(f"[Warning] Taker-flow fetch failed for {tf}: {e}")
        return None


def _drop_forming_candle(df: pd.DataFrame, tf: str, now_ms: int) -> pd.DataFrame:
    """
    Remove the last row if its bar has not closed yet, so all indicators and
    structure are computed on CLOSED candles only (fixes B3/B4/T1-1).
    """
    if df is None or df.empty:
        return df
    last_open_ms = int(df["timestamp"].iloc[-1].value // 1_000_000)
    if now_ms < last_open_ms + TF_MS[tf]:
        return df.iloc[:-1].reset_index(drop=True)
    return df


async def fetch_ohlcv(symbol):
    """
    Fetch OHLCV + taker flow across timeframes (retry + LRU cache), drop the
    forming candle, attach real-CVD inputs, and compute indicators on closed
    bars. Returns (data: {tf: df}, live_price: float).
    """
    exchange = ccxt.binance({"enableRateLimit": True})
    raw_dfs = {}
    now = time.time()
    now_ms = int(now * 1000)
    live_price = None

    try:
        for tf in TIMEFRAMES:
            cache_key = (symbol, tf)
            cached = _cache_get(cache_key)
            if cached is not None:
                cached_time, cached_df = cached
                if now - cached_time < CACHE_TTL_SECONDS[tf]:
                    raw_dfs[tf] = cached_df.copy()
                    continue

            candles = None
            for attempt in range(3):
                try:
                    candles = await exchange.fetch_ohlcv(symbol, tf, limit=LIMITS[tf])
                    break
                except Exception as e:
                    if attempt == 2:
                        raise e
                    await asyncio.sleep(2 ** attempt)

            if candles is not None:
                df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                # capture the freshest (possibly forming) 15m close as the live price
                if tf == "15m":
                    live_price = float(df["close"].iloc[-1])
                # merge real taker-buy volume (best-effort) before dropping forming bar
                taker = await asyncio.to_thread(fetch_taker_flow, symbol, tf, LIMITS[tf])
                if taker is not None:
                    df = df.merge(taker, on="timestamp", how="left")
                df = _drop_forming_candle(df, tf, now_ms)
                raw_dfs[tf] = df
                _cache_put(cache_key, (now, df.copy()))
            elif tf == "15m" and cache_key in OHLCV_CACHE:
                pass
    finally:
        await exchange.close()

    if live_price is None and "15m" in raw_dfs and not raw_dfs["15m"].empty:
        live_price = float(raw_dfs["15m"]["close"].iloc[-1])

    data = {}
    for tf, df in raw_dfs.items():
        data[tf] = await asyncio.to_thread(calculate_indicators, df, timeframe=tf)
    return data, live_price


def fetch_orderbook(symbol):
    """L2 order book with depth-binned imbalance (±0.25 / 0.5 / 1 / 2%)."""
    default = {"bid_vol": 0.0, "ask_vol": 0.0, "skew": 0.0, "mid_price": 0.0, "depth_bins": []}
    try:
        r = requests.get(
            "https://fapi.binance.com/fapi/v1/depth",
            params={"symbol": symbol.replace("/", ""), "limit": 500},
            timeout=8,
        ).json()
    except Exception as e:
        print(f"[Warning] Orderbook fetch failed: {e}")
        return default

    bids = [(float(b[0]), float(b[1])) for b in r.get("bids", [])]
    asks = [(float(a[0]), float(a[1])) for a in r.get("asks", [])]
    if not bids or not asks:
        return default

    mid = (bids[0][0] + asks[0][0]) / 2
    if mid <= 0:
        return default

    bands = [0.0025, 0.005, 0.01, 0.02]
    depth_bins = []
    for b in bands:
        bid_band = sum(qty for price, qty in bids if (mid - price) / mid <= b)
        ask_band = sum(qty for price, qty in asks if (price - mid) / mid <= b)
        total = bid_band + ask_band
        imbalance = (bid_band - ask_band) / total if total > 0 else 0.0
        depth_bins.append({
            "band_pct": round(b * 100, 3),
            "bid": round(bid_band, 4),
            "ask": round(ask_band, 4),
            "imbalance": round(imbalance, 4),
        })

    total_bid = sum(qty for _, qty in bids)
    total_ask = sum(qty for _, qty in asks)
    skew = (total_bid - total_ask) / max(total_bid + total_ask, 1e-9)

    return {
        "bid_vol": round(total_bid, 4),
        "ask_vol": round(total_ask, 4),
        "skew": round(skew, 4),
        "mid_price": round(mid, 6),
        "depth_bins": depth_bins,
    }


def _load_oi_history():
    if not os.path.exists(OI_HISTORY_PATH):
        return {}
    try:
        with open(OI_HISTORY_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_oi_history(hist):
    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        with open(OI_HISTORY_PATH, "w") as f:
            json.dump(hist, f)
    except Exception as e:
        print(f"[Warning] Failed to persist OI history: {e}")


def fetch_open_interest(symbol):
    """OI now + 1h / 4h deltas + 30-day percentile (history endpoint with local fallback)."""
    sym_clean = symbol.replace("/", "")
    oi_now = 0.0
    try:
        r = requests.get(
            "https://fapi.binance.com/fapi/v1/openInterest",
            params={"symbol": sym_clean},
            timeout=5,
        ).json()
        oi_now = float(r.get("openInterest", 0.0))
    except Exception as e:
        print(f"[Warning] OI fetch failed: {e}")

    oi_change_1h = None
    oi_change_4h = None
    oi_percentile_30d = None

    try:
        r = requests.get(
            "https://fapi.binance.com/futures/data/openInterestHist",
            params={"symbol": sym_clean, "period": "1h", "limit": 30 * 24},
            timeout=8,
        ).json()
        if isinstance(r, list) and len(r) >= 2:
            vals = []
            for x in r:
                try:
                    vals.append((int(x.get("timestamp", 0)), float(x.get("sumOpenInterest", 0))))
                except Exception:
                    continue
            vals.sort(key=lambda t: t[0])
            series = [v for _, v in vals]
            if len(series) >= 2 and series[-2] > 0:
                oi_change_1h = round((oi_now - series[-2]) / series[-2] * 100, 3)
            if len(series) >= 5 and series[-5] > 0:
                oi_change_4h = round((oi_now - series[-5]) / series[-5] * 100, 3)
            if series:
                below = sum(1 for v in series if v <= oi_now)
                oi_percentile_30d = round(below / len(series) * 100, 1)
    except Exception as e:
        print(f"[Warning] OI history fetch failed: {e}")

    # Local fallback / augmentation
    history = _load_oi_history()
    sym_hist = history.get(symbol, [])
    sym_hist.append({"ts": int(time.time()), "oi": oi_now})
    sym_hist = sym_hist[-2000:]
    history[symbol] = sym_hist
    _save_oi_history(history)
    if oi_percentile_30d is None and len(sym_hist) >= 10:
        vals_local = [x["oi"] for x in sym_hist]
        below = sum(1 for v in vals_local if v <= oi_now)
        oi_percentile_30d = round(below / len(vals_local) * 100, 1)

    return {
        "value": oi_now,
        "change_1h_pct": oi_change_1h,
        "change_4h_pct": oi_change_4h,
        "percentile_30d": oi_percentile_30d,
    }


def fetch_funding(symbol):
    """Current funding + 24-period trajectory + percentile-within-window."""
    default = {"current": 0.0, "avg_24h": 0.0, "trend": "unknown",
               "sign_changes": 0, "percentile_window": None, "window_periods": 0}
    try:
        r = requests.get(
            "https://fapi.binance.com/fapi/v1/fundingRate",
            params={"symbol": symbol.replace("/", ""), "limit": 24},
            timeout=5,
        ).json()
    except Exception as e:
        print(f"[Warning] Funding fetch failed: {e}")
        return default

    if not isinstance(r, list) or not r:
        return default

    try:
        rates = [float(x["fundingRate"]) for x in r]
    except Exception:
        return default

    current = rates[-1]
    avg_24h = sum(rates[-3:]) / max(len(rates[-3:]), 1)
    sign_changes = sum(1 for i in range(1, len(rates)) if (rates[i] > 0) != (rates[i - 1] > 0))
    delta = current - avg_24h
    if abs(delta) < 1e-6:
        trend = "flat"
    elif delta > 0:
        trend = "rising"
    else:
        trend = "falling"
    below = sum(1 for v in rates if v <= current)
    percentile = round(below / len(rates) * 100, 1)

    return {
        "current": round(current, 6),
        "avg_24h": round(avg_24h, 6),
        "trend": trend,
        "sign_changes": int(sign_changes),
        "percentile_window": percentile,
        "window_periods": len(rates),
    }


def fetch_sentiment():
    try:
        r = requests.get("https://api.alternative.me/fng/", timeout=5).json()
        return int(r["data"][0]["value"])
    except Exception:
        print("[Warning] Fear & Greed API failed. Defaulting to neutral 50.")
        return 50


def fetch_btc_dominance_proxy(symbol):
    """BTCDOM perp 4H/24H change as a dominance proxy. Skipped when symbol is BTC."""
    if "BTC" in symbol.upper().split("/")[0]:
        return None
    try:
        r = requests.get(
            "https://fapi.binance.com/fapi/v1/klines",
            params={"symbol": "BTCDOMUSDT", "interval": "4h", "limit": 10},
            timeout=8,
        ).json()
        if not isinstance(r, list) or len(r) < 2:
            return None
        closes = [float(x[4]) for x in r]
        change_4h = (closes[-1] - closes[-2]) / closes[-2] * 100
        ref_idx = -6 if len(closes) >= 6 else 0
        change_24h = (closes[-1] - closes[ref_idx]) / closes[ref_idx] * 100
        return {
            "btcdom_close": round(closes[-1], 4),
            "change_4h_pct": round(change_4h, 3),
            "change_24h_pct": round(change_24h, 3),
        }
    except Exception as e:
        print(f"[Warning] BTC.D proxy fetch failed: {e}")
        return None


def fetch_event_window(now_ts: float = None, window_hours: int = 2) -> dict:
    """Active high-impact macro events near current time (data/events.json)."""
    if now_ts is None:
        now_ts = time.time()
    if not os.path.exists(EVENTS_PATH):
        return {"active": False, "events": [], "window_hours": window_hours}
    try:
        with open(EVENTS_PATH) as f:
            events = json.load(f)
    except Exception:
        return {"active": False, "events": [], "window_hours": window_hours}

    active = []
    window_s = window_hours * 3600
    for e in events if isinstance(events, list) else []:
        try:
            dt = datetime.fromisoformat(e["datetime"])
            ev_ts = dt.timestamp()
        except Exception:
            continue
        if abs(ev_ts - now_ts) <= window_s:
            active.append({
                "name": e.get("name", "unknown"),
                "impact": e.get("impact", "medium"),
                "datetime": e["datetime"],
                "minutes_until": round((ev_ts - now_ts) / 60, 1),
            })
    return {"active": len(active) > 0, "events": active, "window_hours": window_hours}


# ---------------- Gemini ---------------- #

GEMINI_SYSTEM_INSTRUCTION = """
ICT/SMC institutional trade engine. Return a JSON object matching the enforced schema. Signals are computed on CLOSED candles; `live_price` is the current price for distance/trigger context only.

Rules
1. Use only numbers present in the snapshot. Never invent levels.
2. `strategies.confluence_score` is authoritative (already includes the order-flow modifier). Echo it; only set `header.score_override` on a specific factual error.
3. TRADE PLAN = SELECT THE BEST REAL CANDIDATE, NEVER INVENT. This is the highest-stakes decision in the output — the score only says *whether*; the candidate you pick decides the actual outcome. Choose deliberately.
   `strategies.engine_trade` gives a deterministic primary (entry/stop_loss/take_profit/rr) AND a candidate menu in `engine_trade.candidates`. Each candidate carries quality metadata — USE IT to rank, do not just take the nearest:
     - `candidates.entries[]` {price, source, tf, hvn, zone, confluence, confluence_count} — choose ONE as `trade_decision.entry`
     - `candidates.stops[]`   {price, source, tf} — choose ONE as `trade_decision.stop_loss`
     - `candidates.targets[]` {price, source, rr, tf, confluence, confluence_count} — choose targets; `rr` is from the primary entry/SL
   RANKING RULES (apply in this order):
     a. Higher `confluence_count` wins — a level where multiple sources stack (e.g. VAH+BSL+POC) is far stronger than an isolated one.
     b. `hvn: true` order blocks > plain OBs; higher `tf` (4h > 1h > 15m) levels are more significant.
     c. ENTRIES: prefer `zone` matching the trade — `discount` for longs, `premium` for shorts.
     d. TARGETS: a target sitting just BEFORE a higher-TF level/resistance is safer than one at/through it.
     e. `rr` is the TIE-BREAKER between otherwise-equal candidates, not the primary filter.
   Every price you output (entry / stop_loss / take_profit / tp1 / tp2 / move_to_breakeven_at) MUST be a value from these candidate lists or the engine primary. Never output a price not in the snapshot.
   Default to the engine primary UNLESS a candidate is clearly higher-quality by the rules above. Set `trade_management.selected_from_engine` = true if you echo the primary, false otherwise, and in `selection_rationale` COMPARE your pick against the primary and the next-best candidate ("chose X (confluence 3, HVN) over primary Y because …").
   `engine_trade.rr` / `rr_passed` is the HONEST structural R:R — the engine never inflates TP to hit 2R. If the chosen final TP gives < 2R, do NOT move it further out; downgrade the action (CONDITIONAL_ENTRY / HOLD). If `engine_trade.valid` is false there is no sound structural trade → HOLD.
   TRADE MANAGEMENT: `tp1` = nearest HIGH-CONFLUENCE target (the first strong stacked level), `tp2` = a farther runner (>= 2R if available); set `tp1_rr`/`tp2_rr` from the candidate's `rr`; `move_to_breakeven_at` = tp1 or a candidate level between entry and tp1; `scale_out` describes the split (e.g. "50% at TP1, move SL to BE, 50% runner to TP2").

Regime-conditional scoring — read `strategies.score_mode`:
 score_mode == "trend" (trending regime): components C1-C8 measure trend continuation.
   C1 trend(25) ADX-tiered on 4H/1H agreement · C2 ob_prox(15) distance to 15m/1H OB·FVG
   C3 sweep(10) HTF sweep+reclaim (6 if 15m-only) · C4 momentum(15) 1H RSI/MACD ±slope
   C5 fvg_magnet(15) · C6 ote(10) · C7 cvd(10) real taker-flow CVD · C8 stoch(5)
 score_mode == "mean_revert" (ranging regime): FADE the value-area edge, do not chase trend.
   M1 edge_distance(25) proximity to VAH/VAL · M2 edge_sweep(15) sweep of the edge
   M3 cvd_absorption(15) · M4 stoch_extreme(15) · M5 rejection candle(10) · M6 range_intact(10)
   Direction = SELL near VAH, BUY near VAL. `setup_direction` carries this.

Order flow: `strategies.order_flow_modifier` (already in the score) reflects L2 depth imbalance,
funding crowding + trajectory, OI build, F&G extremes, and BTC-dominance (alt macro headwind) —
cite `order_flow_notes` in the sentiment reasoning. `price_action.avwap` holds anchored-VWAP
dynamic S/R from the last swing high/low; cite it as a level when relevant.

Thresholds
 score >= 60 → ACTIVE_TRADE   45-59 → CONDITIONAL_ENTRY   < 45 → HOLD

Gates
 volume_gate.HARD_GATE        → force HOLD regardless of score
 volume_gate.LOW_VOL_WARNING  → position_size = "50% low-vol"; require +1 confluence
 event_guard.active == true   → cap action at CONDITIONAL_ENTRY; reduce confidence by 15
 cvd_absorption_warning       → reduce confidence by 10; cite explicitly
 setup_direction == "neutral" → HOLD (no clean setup)

Action-specific filling
 HOLD               → trade_decision entry/stop_loss/take_profit/rr.* and all trade_management price fields = null; selected_from_engine=true, scale_out="n/a". Forward scenario filled.
 CONDITIONAL_ENTRY  → select levels from engine_trade.candidates; entry_trigger states the exact required condition; fill trade_management.
 ACTIVE_TRADE       → select levels from engine_trade.candidates; final R:R must be >= 2 (never invent a 2R TP); fill trade_management with tp1/tp2 + BE + scale_out.

Field formats
 mtf_context.<tf>.nearest_ob / nearest_fvg : "<low>-<high>" or "NONE".
 forward_scenario.key_levels.<tf> : ["<price> — <label>", ...] sourced from snapshot.
 narrative.summary : ≤ 2 sentences.
""".strip()


GEMINI_MODEL_NAME = "gemini-3.5-flash"
# Gemini 3.5 uses string-enum thinking levels instead of numeric budgets.
# minimal | low | medium (default) | high. `medium` is the sweet spot for C1-C8
# verification — high adds ~30% latency for marginal lift on rule-following tasks.
GEMINI_THINKING_LEVEL = "high"

_INT = {"type": "integer"}
_NUM = {"type": "number"}
_STR = {"type": "string"}
_BOOL = {"type": "boolean"}
_NUM_NULL = {"type": "number", "nullable": True}
_STR_ARR = {"type": "array", "items": _STR}


def _tf_schema(extra_props: dict, extra_required: list) -> dict:
    """Common per-timeframe schema with optional extras stacked on top."""
    base_props = {
        "structure": _STR,
        "bos": _STR,
        "cvd_delta": _NUM,
        "price_change_pct": _NUM,
        "pd_zone": {"type": "string", "enum": ["premium", "discount", "neutral"]},
        "in_ote": _BOOL,
        "nearest_ob": _STR,
        "nearest_fvg": _STR,
    }
    base_required = ["structure", "cvd_delta", "price_change_pct", "pd_zone",
                     "in_ote", "nearest_ob", "nearest_fvg"]
    return {
        "type": "object",
        "properties": {**base_props, **extra_props},
        "required": base_required + extra_required,
    }


GEMINI_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "header": {
            "type": "object",
            "properties": {
                "pair": _STR,
                "price": _NUM,
                "bias": {"type": "string", "enum": ["BULLISH", "BEARISH", "NEUTRAL"]},
                "regime": _STR,
                "score": _INT,
                "action": {"type": "string", "enum": ["ACTIVE_TRADE", "CONDITIONAL_ENTRY", "HOLD"]},
                "volume_gate": {"type": "string", "enum": ["HARD_GATE", "LOW_VOL_WARNING", "CLEAR"]},
                "score_breakdown": {
                    "type": "object",
                    "properties": {
                        "c1_trend": _INT, "c2_ob_prox": _INT, "c3_sweep": _INT,
                        "c4_momentum": _INT, "c5_fvg_magnet": _INT, "c6_ote": _INT,
                        "c7_cvd": _INT, "c8_stoch": _INT,
                    },
                    "required": ["c1_trend", "c2_ob_prox", "c3_sweep", "c4_momentum",
                                 "c5_fvg_magnet", "c6_ote", "c7_cvd", "c8_stoch"],
                },
                "score_override": {
                    "type": "object",
                    "nullable": True,
                    "properties": {
                        "from": _INT,
                        "to": _INT,
                        "reason": _STR,
                    },
                },
            },
            "required": ["pair", "price", "bias", "regime", "score", "action",
                         "volume_gate", "score_breakdown"],
        },
        "mtf_context": {
            "type": "object",
            "properties": {
                "h4": _tf_schema(
                    {"adx": _NUM,
                     "adx_tier": {"type": "string", "enum": ["WEAK", "MEDIUM", "STRONG", "VERY_STRONG"]}},
                    ["adx", "adx_tier", "bos"],
                ),
                "h1": _tf_schema(
                    {"adx": _NUM, "rsi": _NUM, "rsi_slope": _NUM, "stochrsi_k": _NUM},
                    ["adx", "rsi", "rsi_slope", "stochrsi_k", "bos"],
                ),
                "m15": _tf_schema(
                    {"rel_volume": _NUM, "candle_pattern": _STR},
                    ["rel_volume", "candle_pattern"],
                ),
                "cross_tf_momentum": _STR,
            },
            "required": ["h4", "h1", "m15", "cross_tf_momentum"],
        },
        "narrative": {
            "type": "object",
            "properties": {
                "summary": _STR,
                "primary_draw": _STR,
            },
            "required": ["summary", "primary_draw"],
        },
        "trade_decision": {
            "type": "object",
            "properties": {
                "primary": {
                    "type": "object",
                    "properties": {
                        "direction": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
                        "probability": _INT,
                    },
                    "required": ["direction", "probability"],
                },
                "alternative": {
                    "type": "object",
                    "properties": {
                        "direction": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
                        "probability": _INT,
                        "trigger": _STR,
                    },
                    "required": ["direction", "probability", "trigger"],
                },
                "entry": _NUM_NULL,
                "entry_source": _STR,
                "stop_loss": _NUM_NULL,
                "stop_loss_source": _STR,
                "take_profit": _NUM_NULL,
                "take_profit_source": _STR,
                "rr": {
                    "type": "object",
                    "properties": {
                        "tp_distance": _NUM_NULL,
                        "sl_distance": _NUM_NULL,
                        "ratio": _NUM_NULL,
                        "passed": _BOOL,
                    },
                    "required": ["tp_distance", "sl_distance", "ratio", "passed"],
                },
                "sl_width_pct": _NUM_NULL,
                "confidence_pct": _INT,
                "position_size": {
                    "type": "string",
                    "enum": ["100%", "50% low-vol", "25% cascade", "reduced - conflict"],
                },
                "entry_trigger": _STR,
                "invalidation": _STR,
                "counter_structure": _BOOL,
                "trade_management": {
                    "type": "object",
                    "properties": {
                        "selected_from_engine": _BOOL,
                        "selection_rationale": _STR,
                        "tp1": _NUM_NULL,
                        "tp1_source": _STR,
                        "tp1_rr": _NUM_NULL,
                        "tp2": _NUM_NULL,
                        "tp2_source": _STR,
                        "tp2_rr": _NUM_NULL,
                        "move_to_breakeven_at": _NUM_NULL,
                        "scale_out": _STR,
                    },
                    "required": ["selected_from_engine", "selection_rationale",
                                 "tp1", "tp1_source", "tp1_rr", "tp2", "tp2_source", "tp2_rr",
                                 "move_to_breakeven_at", "scale_out"],
                },
                "reasoning": {
                    "type": "object",
                    "properties": {
                        "structure": _STR,
                        "liquidity": _STR,
                        "momentum": _STR,
                        "sentiment": _STR,
                    },
                    "required": ["structure", "liquidity", "momentum", "sentiment"],
                },
            },
            "required": ["primary", "alternative", "entry", "entry_source",
                         "stop_loss", "stop_loss_source", "take_profit", "take_profit_source",
                         "rr", "sl_width_pct", "confidence_pct", "position_size",
                         "entry_trigger", "invalidation", "counter_structure",
                         "trade_management", "reasoning"],
        },
        "forward_scenario": {
            "type": "object",
            "properties": {
                "direction": {"type": "string", "enum": ["LONG", "SHORT", "NEUTRAL"]},
                "key_levels": {
                    "type": "object",
                    "properties": {"h4": _STR_ARR, "h1": _STR_ARR, "m15": _STR_ARR},
                    "required": ["h4", "h1", "m15"],
                },
                "trigger": _STR,
                "entry": _NUM_NULL,
                "stop_loss": _NUM_NULL,
                "take_profit": _NUM_NULL,
                "rr": _NUM_NULL,
                "volume_condition": _STR,
                "supporting_confluence": _STR,
            },
            "required": ["direction", "key_levels", "trigger", "entry", "stop_loss",
                         "take_profit", "rr", "volume_condition", "supporting_confluence"],
        },
    },
    "required": ["header", "mtf_context", "narrative", "trade_decision", "forward_scenario"],
}


# Module-level lazy client. `google-genai` clients are reusable across calls,
# so we build one on first use and keep it.
_GENAI_CLIENT: "genai.Client | None" = None


def _get_genai_client() -> "genai.Client":
    global _GENAI_CLIENT
    if _GENAI_CLIENT is not None:
        return _GENAI_CLIENT
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    _GENAI_CLIENT = genai.Client(api_key=api_key)
    return _GENAI_CLIENT


def _build_generate_config(include_schema: bool = True) -> "genai_types.GenerateContentConfig":
    """
    GenerateContentConfig for Gemini 3.5 Flash:
      * `system_instruction` lives inside the config in the new SDK.
      * `response_mime_type` + `response_schema` enforce JSON shape.
      * `thinking_config.thinking_level` replaces the deprecated numeric budget.
      * Sampling knobs (temperature/top_p/top_k/candidate_count) are
        deliberately omitted — 3.x is tuned for its own defaults and the docs
        warn against forced sampling for rule-following tasks.
    """
    kwargs = {
        "system_instruction": GEMINI_SYSTEM_INSTRUCTION,
        "response_mime_type": "application/json",
        "thinking_config": genai_types.ThinkingConfig(thinking_level=GEMINI_THINKING_LEVEL),
    }
    if include_schema:
        kwargs["response_schema"] = GEMINI_RESPONSE_SCHEMA
    return genai_types.GenerateContentConfig(**kwargs)


def _extract_response_dict(response) -> dict | None:
    """
    Prefer `response.parsed` when schema enforcement produced a dict /
    pydantic model; otherwise fall back to JSON-parsing `response.text`,
    with defensive code-fence stripping for the rare case a model still
    wraps JSON in markdown.
    """
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        if isinstance(parsed, dict):
            return parsed
        if hasattr(parsed, "model_dump"):
            return parsed.model_dump()
        try:
            return dict(parsed)
        except Exception:
            pass

    raw = getattr(response, "text", None)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].lstrip()
        try:
            return json.loads(cleaned)
        except Exception:
            return {"error": "JSON parse failed", "raw_text": raw}


def query_gemini(snapshot: dict) -> dict:
    """
    Call Gemini 3.5 Flash via the google-genai SDK with schema-enforced JSON
    output and a thinking level. Soft-falls-back to plain JSON-mode (no
    schema) if the SDK rejects our dict-form schema — keeps the engine
    running while a schema-shape issue is diagnosed.
    """
    try:
        client = _get_genai_client()
    except RuntimeError as e:
        return {"error": str(e)}

    prompt = f"DATA SNAPSHOT:\n{json.dumps(snapshot, indent=2)}"

    for label, include_schema in (("schema", True), ("plain-json", False)):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=prompt,
                config=_build_generate_config(include_schema=include_schema),
            )
        except Exception as e:
            msg = str(e).lower()
            schema_rejection = include_schema and any(
                s in msg for s in ("schema", "unknown field", "invalid_argument")
            )
            if schema_rejection:
                print(f"[Warning] Gemini schema rejected ({e}); retrying without schema.")
                continue
            return {"error": f"Gemini call failed ({label}): {e}"}

        result = _extract_response_dict(response)
        if result is None:
            return {"error": "empty response"}
        return result

    return {"error": "All Gemini config tiers failed"}


# ---------------- snapshot retention ---------------- #

def _enforce_snapshot_retention(symbol: str, retention: int = SNAPSHOT_RETENTION):
    """Keep at most `retention` snapshot files per symbol (oldest deleted first)."""
    if not os.path.isdir(SNAPSHOTS_DIR):
        return
    prefix = f"snapshot_{symbol.replace('/', '_')}_"
    files = []
    for name in os.listdir(SNAPSHOTS_DIR):
        if name.startswith(prefix) and name.endswith(".json"):
            full = os.path.join(SNAPSHOTS_DIR, name)
            try:
                files.append((os.path.getmtime(full), full))
            except OSError:
                continue
    files.sort(key=lambda t: t[0])
    if len(files) <= retention:
        return
    for _, path in files[: len(files) - retention]:
        try:
            os.remove(path)
        except OSError:
            continue


# ---------------- core pipeline ---------------- #

async def run_analysis(symbol: str) -> tuple[dict, dict]:
    """End-to-end pipeline. Returns (snapshot_dict, analysis_dict)."""
    print(f"\nAnalyzing market data for {symbol}...")

    data, live_price = await fetch_ohlcv(symbol)
    if live_price is not None:
        print(f"[DEBUG] Live price for {symbol}: {live_price:.4f} (signals on closed bars)")

    # External micro / macro data fetched in parallel-friendly order (sync REST)
    orderbook = fetch_orderbook(symbol)
    funding = fetch_funding(symbol)
    sentiment = fetch_sentiment()
    open_interest = fetch_open_interest(symbol)
    btc_dominance = fetch_btc_dominance_proxy(symbol)
    event_guard = fetch_event_window()

    # Compute each TF's value area once and reuse (avoids recomputation, B11)
    value_areas = {tf: get_value_area(data[tf]) for tf in data}

    # SMC + PA contexts
    smc_context = build_smc_context(data["4h"], data["1h"], data["15m"], value_areas=value_areas)
    pa_context = build_pa_context(data["1h"], data["15m"], df_4h=data["4h"], value_areas=value_areas)
    pdh, pdl, pdc = calculate_previous_day(data["1h"])

    # Volatility regimes per TF
    vol_regime = {tf: detect_volatility_regime(data[tf]) for tf in data}

    # Global regime (SMC-driven); drives regime-conditional scoring
    regime = detect_regime(data["4h"], data["1h"], data["15m"], smc_context=smc_context)

    # 1H Supertrend pulled out for the dashboard pill
    latest_1h = data["1h"].iloc[-1]
    st_val = float(latest_1h["SUPERT_10_3.0"]) if "SUPERT_10_3.0" in latest_1h and not pd.isna(latest_1h["SUPERT_10_3.0"]) else 0.0
    st_dir_val = latest_1h["SUPERTd_10_3.0"] if "SUPERTd_10_3.0" in latest_1h and not pd.isna(latest_1h["SUPERTd_10_3.0"]) else 0
    supertrend_direction = "BULLISH" if st_dir_val == 1 else ("BEARISH" if st_dir_val == -1 else "NEUTRAL")

    windowed_indicators = {tf: get_window_dict(data[tf], WINDOW_SIZE[tf]) for tf in data}

    strategies = evaluate_strategies(
        data["4h"], data["1h"], data["15m"],
        orderbook, funding, sentiment, open_interest,
        smc_context=smc_context,
        windowed_indicators=windowed_indicators,
        pa_context=pa_context,
        market_regime=regime,
        btc_dominance=btc_dominance,
    )

    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "symbol": symbol,
        "live_price": live_price,
        "market_regime": regime,
        "volatility_regime": vol_regime,
        "event_guard": event_guard,
        "previous_day": {"pdh": pdh, "pdl": pdl, "pdc": pdc},
        "orderbook": orderbook,
        "funding": funding,
        "fear_greed_index": sentiment,
        "open_interest": open_interest,
        "btc_dominance_proxy": btc_dominance,
        "supertrend": {"direction": supertrend_direction, "level": st_val},
        "smc_context": smc_context,
        "price_action": pa_context,
        "strategies": strategies,
        "engine_trade": strategies.get("engine_trade"),
        "windowed_indicators": windowed_indicators,
    }
    snapshot_native = to_native(snapshot)

    print("Generating structured trade plan from Gemini...")
    analysis = query_gemini(snapshot_native)
    snapshot_native["analysis"] = analysis

    # Attach chart OHLC AFTER the LLM call so it never inflates the prompt.
    snapshot_native["chart_series"] = build_chart_series(data)

    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_filename = f"snapshot_{symbol.replace('/', '_')}_{timestamp_str}.json"
    snapshot_path = os.path.join(SNAPSHOTS_DIR, snapshot_filename)
    with open(snapshot_path, "w") as f:
        json.dump(snapshot_native, f, indent=2)
    print(f"Snapshot saved to snapshots/{snapshot_filename}")

    _enforce_snapshot_retention(symbol)
    return snapshot_native, analysis


# ---------------- CLI ---------------- #

async def main():
    import sys
    symbol = "SOL/USDT"

    if len(sys.argv) > 1:
        arg = sys.argv[1].upper().strip()
        if arg in ("SOL", "SOL/USDT", "SOLUSDT"):
            symbol = "SOL/USDT"
        elif arg in ("BTC", "BTC/USDT", "BTCUSDT"):
            symbol = "BTC/USDT"
        elif arg in ("ETH", "ETH/USDT", "ETHUSDT"):
            symbol = "ETH/USDT"
        elif "/" in arg:
            symbol = arg
        else:
            print(f"Unknown argument '{arg}'. Defaulting to SOL/USDT.")
    else:
        print("\n=== Cryptera v3.2 Core Engine ===")
        print("1. SOL/USDT (default)")
        print("2. BTC/USDT")
        print("3. ETH/USDT")
        print("4. Custom symbol")
        try:
            choice = input("Select coin choice (1-4): ").strip()
            if choice == "2":
                symbol = "BTC/USDT"
            elif choice == "3":
                symbol = "ETH/USDT"
            elif choice == "4":
                custom = input("Enter custom symbol (e.g. LINK/USDT): ").strip().upper()
                if "/" in custom:
                    symbol = custom
                else:
                    print("Invalid format. Defaulting to SOL/USDT.")
        except Exception:
            print("Invalid input. Defaulting to SOL/USDT.")

    snapshot_native, analysis = await run_analysis(symbol)
    print("\n" + "=" * 60)
    print(json.dumps(analysis, indent=2))
    print("=" * 60)

    import gc, sys as _sys
    gc.collect()
    _sys.exit(0)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except SystemExit:
        pass
