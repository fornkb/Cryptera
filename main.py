"""
Cryptera v3.1 - Rule-based SMC engine + Gemini AI structured JSON narration.
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
import google.generativeai as genai

from indicators import calculate_indicators, detect_regime, detect_volatility_regime
from strategies import evaluate_strategies
from smc import build_smc_context
from price_action import build_pa_context, calculate_previous_day


SCHEMA_VERSION = "3.1.0"
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
        "adx": adx_latest,
        "atr_percentile": atr_pct_latest,
        "close_series_last_5": close_series,
        "rsi_series_last_5": rsi_series,
    }


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

async def fetch_ohlcv(symbol):
    """Fetch and index OHLCV across the configured timeframes with retry + LRU cache."""
    exchange = ccxt.binance({"enableRateLimit": True})
    raw_dfs = {}
    now = time.time()

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
                raw_dfs[tf] = df
                _cache_put(cache_key, (now, df.copy()))
    finally:
        await exchange.close()

    data = {}
    for tf, df in raw_dfs.items():
        data[tf] = await asyncio.to_thread(calculate_indicators, df, timeframe=tf)
    return data


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
You are an Elite Institutional Trading System (ICT/SMC). Output a STRICTLY VALID JSON trade decision derived from the supplied market snapshot. No preamble, no markdown, no comments outside the JSON.

ABSOLUTE GROUNDING RULES
1. Use ONLY exact numbers present in the snapshot. Never invent price levels, swing points, OBs, FVGs or VA bounds.
2. The snapshot's `strategies.confluence_breakdown` already mirrors the C1-C8 rubric below. Treat it as the authoritative score. Only override if a specific factual error exists; otherwise echo it.
3. Risk-reward: enforce min 1:2. Verify mathematically and report distances.
4. SL within 0.5%-1.5% of entry; entry close to current 15m structure (OB / FVG edge / current price). No far-away macro entries.
5. Output MUST be a single JSON object matching the schema below. No leading text.

SCORING RUBRIC (already pre-computed — verify, do not recompute unless wrong):
 C1 trend_alignment   max 25 — ADX-tiered on agreement of 4H/1H structure; +10 base on conflict
 C2 ob_proximity      max 15 — tiered 0/5/10/15 by distance to 15m or 1H OB/FVG edge
 C3 liquidity_sweep   max 10 — binary
 C4 momentum          max 15 — base 10 if 1H RSI/MACD aligns with bias, +/-5 RSI slope
 C5 fvg_magnet        max 15 — unfilled draw-on-liquidity FVG within 3% on 1H or 4H
 C6 ote_bonus         max 10 — in_ote on 1H or 4H
 C7 cvd_alignment     max 10 — 2+ TF CVD signs match bias; 0 if 15m diverges from 4H (absorption)
 C8 stochrsi          max  5 — overbought (sell) / oversold (buy) on 1H

THRESHOLDS
  score >= 60        ACTIVE_TRADE  (still subject to volume gate)
  score 45 - 59      CONDITIONAL_ENTRY
  score < 45         HOLD

VOLUME GATE (from strategies.volume_gate.state)
  HARD_GATE          execution suspended regardless of score; action = HOLD
  LOW_VOL_WARNING    reduce position size to 50%; require +1 confluence factor
  CLEAR              no penalty

EVENT GUARD
  If event_guard.active == true: cap action at CONDITIONAL_ENTRY and reduce confidence by 15.

CONFLICT HANDLING
  4H/1H conflict      follow 4H direction; cite conflict as a risk factor
  15m diverges 1H     flag counter_structure=true; require 15m CHoCH/BOS trigger before entry
  CVD absorption      (strategies.cvd_absorption_warning == true) reduce confidence by 10, note explicitly

OUTPUT JSON SCHEMA (every field required, fill with snapshot values; use null only where explicitly allowed):
{
  "header": {
    "pair": "<symbol>",
    "price": <number>,
    "bias": "BULLISH" | "BEARISH" | "NEUTRAL",
    "regime": "<market_regime>",
    "score": <int 0-100>,
    "action": "ACTIVE_TRADE" | "CONDITIONAL_ENTRY" | "HOLD",
    "volume_gate": "HARD_GATE" | "LOW_VOL_WARNING" | "CLEAR",
    "score_breakdown": {
      "c1_trend": <int>, "c2_ob_prox": <int>, "c3_sweep": <int>,
      "c4_momentum": <int>, "c5_fvg_magnet": <int>, "c6_ote": <int>,
      "c7_cvd": <int>, "c8_stoch": <int>
    },
    "score_override": null
  },
  "mtf_context": {
    "h4":  {"structure": "<str>", "bos": "<str|NONE>", "adx": <num>, "adx_tier": "WEAK|MEDIUM|STRONG|VERY_STRONG", "cvd_delta": <num>, "price_change_pct": <num>, "pd_zone": "premium|discount|neutral", "in_ote": <bool>, "nearest_ob": "<str|NONE>", "nearest_fvg": "<str|NONE>"},
    "h1":  {"structure": "<str>", "bos": "<str|NONE>", "adx": <num>, "rsi": <num>, "rsi_slope": <num>, "stochrsi_k": <num>, "cvd_delta": <num>, "price_change_pct": <num>, "pd_zone": "premium|discount|neutral", "in_ote": <bool>, "nearest_ob": "<str|NONE>", "nearest_fvg": "<str|NONE>"},
    "m15": {"structure": "<str>", "rel_volume": <num>, "candle_pattern": "<str>", "cvd_delta": <num>, "price_change_pct": <num>, "pd_zone": "premium|discount|neutral", "in_ote": <bool>, "nearest_ob": "<str|NONE>", "nearest_fvg": "<str|NONE>"},
    "cross_tf_momentum": "<one short sentence comparing 4h/1h/15m price-change directions>"
  },
  "narrative": {
    "summary": "<<=2 sentences MTF flow + VA positioning + CVD alignment>",
    "primary_draw": "<which liquidity pool or FVG is the next draw on price>"
  },
  "trade_decision": {
    "primary":     {"direction": "BUY"|"SELL"|"HOLD", "probability": <int 0-100>},
    "alternative": {"direction": "BUY"|"SELL"|"HOLD", "probability": <int 0-100>, "trigger": "<exact condition>"},
    "entry": <number|null>,
    "entry_source": "<e.g. '15m bear FVG top'>",
    "stop_loss": <number|null>,
    "stop_loss_source": "<e.g. '15m swing high'>",
    "take_profit": <number|null>,
    "take_profit_source": "<e.g. 'SSL @ 66193'>",
    "rr": {"tp_distance": <num>, "sl_distance": <num>, "ratio": <num>, "passed": <bool>},
    "sl_width_pct": <num>,
    "confidence_pct": <int 0-100>,
    "position_size": "<'100%' | '50% low-vol' | '25% cascade' | 'reduced — conflict'>",
    "entry_trigger": "<exact condition required for entry>",
    "invalidation": "<exact condition that voids the setup>",
    "counter_structure": <bool>,
    "reasoning": {
      "structure": "<one sentence ADX tier + multi-TF structure>",
      "liquidity": "<one sentence VAH/VAL/swept pools/rel_vol/OI>",
      "momentum":  "<one sentence RSI+slope, StochRSI, CVD all 3 TFs>",
      "sentiment": "<one sentence orderbook depth bins + funding trajectory + F&G>"
    }
  },
  "forward_scenario": {
    "direction": "LONG" | "SHORT" | "NEUTRAL",
    "key_levels": {
      "h4":  ["<level + label>", ...],
      "h1":  ["<level + label>", ...],
      "m15": ["<level + label>", ...]
    },
    "trigger":           "<exact contingent event>",
    "entry":             <number|null>,
    "stop_loss":         <number|null>,
    "take_profit":       <number|null>,
    "rr":                <num>,
    "volume_condition":  "<rel_vol gate requirement>",
    "supporting_confluence": "<<=2 sentences citing exact snapshot values>"
  }
}

If action == "HOLD", set entry/stop_loss/take_profit/rr to null in trade_decision but still fill forward_scenario completely.
""".strip()


def query_gemini(snapshot: dict) -> dict:
    """Call Gemini with response_mime_type=application/json and return parsed dict."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {"error": "GEMINI_API_KEY not set"}

    genai.configure(api_key=api_key, transport="rest")
    generation_config = {
        "response_mime_type": "application/json",
        "temperature": 0.2,
    }
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=GEMINI_SYSTEM_INSTRUCTION,
        generation_config=generation_config,
    )
    prompt = f"DATA SNAPSHOT:\n{json.dumps(snapshot, indent=2)}"
    try:
        raw = model.generate_content(prompt).text
    except Exception as e:
        return {"error": f"Gemini call failed: {e}"}
    if not raw:
        return {"error": "empty response"}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Some models prefix or suffix with code fences despite mime; strip and retry.
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].lstrip()
        try:
            return json.loads(cleaned)
        except Exception as e:
            return {"error": f"JSON parse failed: {e}", "raw_text": raw}


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

    data = await fetch_ohlcv(symbol)
    if "15m" in data and not data["15m"].empty:
        current_price = data["15m"]["close"].iloc[-1]
        print(f"[DEBUG] Current price for {symbol}: {current_price:.4f}")

    # External micro / macro data fetched in parallel-friendly order (sync REST)
    orderbook = fetch_orderbook(symbol)
    funding = fetch_funding(symbol)
    sentiment = fetch_sentiment()
    open_interest = fetch_open_interest(symbol)
    btc_dominance = fetch_btc_dominance_proxy(symbol)
    event_guard = fetch_event_window()

    # SMC + PA contexts
    smc_context = build_smc_context(data["4h"], data["1h"], data["15m"])
    pa_context = build_pa_context(data["1h"], data["15m"], df_4h=data["4h"])
    pdh, pdl, pdc = calculate_previous_day(data["1h"])

    # Volatility regimes per TF
    vol_regime = {tf: detect_volatility_regime(data[tf]) for tf in data}

    # Single trend-bias source (SMC-driven) used inside detect_regime
    regime = detect_regime(data["4h"], data["1h"], data["15m"], smc_context=smc_context)

    # 1H Supertrend pulled out for the dashboard pill
    latest_1h = data["1h"].iloc[-1]
    st_val = float(latest_1h["SUPERT_10_3.0"]) if "SUPERT_10_3.0" in latest_1h and not pd.isna(latest_1h["SUPERT_10_3.0"]) else 0.0
    st_dir_val = latest_1h["SUPERTd_10_3.0"] if "SUPERTd_10_3.0" in latest_1h and not pd.isna(latest_1h["SUPERTd_10_3.0"]) else 0
    supertrend_direction = "BULLISH" if st_dir_val == 1 else ("BEARISH" if st_dir_val == -1 else "NEUTRAL")

    windowed_indicators = {tf: get_window_dict(data[tf], WINDOW_SIZE[tf]) for tf in data}

    strategies = evaluate_strategies(
        data["4h"], data["1h"], data["15m"],
        orderbook, funding, sentiment,
        smc_context=smc_context,
        windowed_indicators=windowed_indicators,
    )

    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "symbol": symbol,
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
        "windowed_indicators": windowed_indicators,
    }
    snapshot_native = to_native(snapshot)

    print("Generating structured trade plan from Gemini...")
    analysis = query_gemini(snapshot_native)
    snapshot_native["analysis"] = analysis

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
        print("\n=== Cryptera v3.1 Core Engine ===")
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
