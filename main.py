"""
Cryptera v3.0 – Rule-Based SMC Engine + Gemini AI Narration
"""

import ccxt.async_support as ccxt
import pandas as pd
import asyncio
import requests
import json
import os
import time
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai

from indicators import calculate_indicators, get_market_structure, detect_regime
from strategies import evaluate_strategies
from smc import build_smc_context
from price_action import build_pa_context, calculate_previous_day


def to_native(obj):
    """
    Recursively convert numpy / pandas types to native Python
    so JSON serialization works correctly.
    """
    import numpy as np
    import pandas as pd

    if isinstance(obj, (np.bool_, np.integer, np.floating)):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        else:
            return float(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_native(v) for v in obj]
    return obj


# Configuration
TIMEFRAMES = ["4h", "1h", "15m"]
LIMITS = {"4h": 200, "1h": 200, "15m": 300}
WINDOW_SIZE = {"4h": 10, "1h": 20, "15m": 30}


def get_window_dict(df: pd.DataFrame, size: int) -> dict:
    """
    Extract highly compressed statistical summaries for the window to save 95% of tokens.
    """
    recent = df.tail(size)
    closes = recent["close"].tolist()
    
    start_p = closes[0] if closes else 1.0
    end_p = closes[-1] if closes else 1.0
    price_change_pct = round(((end_p - start_p) / start_p) * 100, 2)
    
    rsi_latest = round(float(recent["RSI"].iloc[-1]), 2) if "RSI" in recent.columns and not pd.isna(recent["RSI"].iloc[-1]) else 50.0
    
    vol_latest = round(float(recent["volume"].iloc[-1]), 2) if "volume" in recent.columns else 0.0
    vol_mean = round(float(recent["volume"].mean()), 2) if "volume" in recent.columns else 1.0
    vol_relative = round(vol_latest / max(vol_mean, 0.1), 2)
    
    supertrend_latest = int(recent["SUPERTd_10_3.0"].iloc[-1]) if "SUPERTd_10_3.0" in recent.columns and not pd.isna(recent["SUPERTd_10_3.0"].iloc[-1]) else 0
    
    stochrsi_k = round(float(recent["STOCHRSIk_14_14_3_3"].iloc[-1]), 2) if "STOCHRSIk_14_14_3_3" in recent.columns and not pd.isna(recent["STOCHRSIk_14_14_3_3"].iloc[-1]) else 50.0
    stochrsi_d = round(float(recent["STOCHRSId_14_14_3_3"].iloc[-1]), 2) if "STOCHRSId_14_14_3_3" in recent.columns and not pd.isna(recent["STOCHRSId_14_14_3_3"].iloc[-1]) else 50.0
    
    cvd_change = 0.0
    if "cvd" in recent.columns:
        cvd_vals = recent["cvd"].dropna().tolist()
        if len(cvd_vals) >= 2:
            cvd_change = round(cvd_vals[-1] - cvd_vals[0], 2)
            
    adx_latest = round(float(recent["ADX"].iloc[-1]), 2) if "ADX" in recent.columns and not pd.isna(recent["ADX"].iloc[-1]) else 0.0
    
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
        "close_series_last_5": close_series,
        "rsi_series_last_5": rsi_series
    }
# Simple in-memory cache to prevent hitting rate limits and speed up execution
OHLCV_CACHE = {}
CACHE_TTL_SECONDS = {
    "4h": 300,  # 5 minutes
    "1h": 120,  # 2 minutes
    "15m": 20   # 20 seconds
}


async def fetch_ohlcv(symbol):
    """
    Fetch OHLCV candles with automatic retry logic (3 attempts with exponential backoff).
    Uses in-memory cache for macro timeframes and offloads indicator calculation to a separate thread.
    """
    exchange = ccxt.binance({"enableRateLimit": True})
    raw_dfs = {}
    now = time.time()

    try:
        for tf in TIMEFRAMES:
            cache_key = (symbol, tf)
            if cache_key in OHLCV_CACHE:
                cached_time, cached_df = OHLCV_CACHE[cache_key]
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
                    # Wait 2^attempt seconds (1s, 2s)
                    await asyncio.sleep(2 ** attempt)
                    
            if candles is not None:
                df = pd.DataFrame(
                    candles,
                    columns=["timestamp", "open", "high", "low", "close", "volume"]
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                raw_dfs[tf] = df
                OHLCV_CACHE[cache_key] = (now, df.copy())
    finally:
        await exchange.close()

    # Compute indicators only after all timeframes are successfully fetched (offloaded to thread)
    data = {}
    for tf, df in raw_dfs.items():
        data[tf] = await asyncio.to_thread(calculate_indicators, df, timeframe=tf)
        
    return data


def fetch_orderbook(symbol):
    r = requests.get(
        "https://fapi.binance.com/fapi/v1/depth",
        params={"symbol": symbol.replace("/", ""), "limit": 100}
    ).json()

    bid_vol = sum(float(b[1]) for b in r["bids"])
    ask_vol = sum(float(a[1]) for a in r["asks"])
    skew = (bid_vol - ask_vol) / max(bid_vol + ask_vol, 1)

    return {"bid_vol": bid_vol, "ask_vol": ask_vol, "skew": skew}


def fetch_open_interest(symbol):
    try:
        r = requests.get(
            "https://fapi.binance.com/fapi/v1/openInterest",
            params={"symbol": symbol.replace("/", "")},
            timeout=5
        ).json()
        return float(r.get("openInterest", 0.0))
    except Exception as e:
        print(f"[Warning] Failed to fetch Open Interest: {e}")
        return 0.0


def fetch_liquidations(symbol):
    try:
        r = requests.get(
            "https://fapi.binance.com/fapi/v1/forceOrders",
            params={"symbol": symbol.replace("/", ""), "limit": 50},
            timeout=5
        ).json()
        if isinstance(r, list):
            total_liq = sum(float(o["origQty"]) * float(o["price"]) for o in r)
            return total_liq
        return 0.0
    except Exception as e:
        print(f"[Warning] Failed to fetch liquidations: {e}")
        return 0.0


def fetch_funding(symbol):
    r = requests.get(
        "https://fapi.binance.com/fapi/v1/fundingRate",
        params={"symbol": symbol.replace("/", ""), "limit": 1}
    ).json()
    return float(r[0]["fundingRate"])


def fetch_sentiment():
    try:
        r = requests.get("https://api.alternative.me/fng/", timeout=5).json()
        return int(r["data"][0]["value"])
    except Exception:
        print("[Warning] Alternative.me Fear & Greed API failed or timed out. Defaulting to neutral (50).")
        return 50


def query_gemini(snapshot):
    load_dotenv()
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"), transport="rest")

    system_instruction = """
    [THRESHOLD SUMMARY]
Score >= 60 → Active trade (subject to volume gate — see Rule 12)
Score 45–59 → Conditional entry (exact trigger required)
Score < 45 → HOLD + forward scenario (mandatory)
Premium/Discount conflict → +1 confluence factor required, not a block
OTE zone → +10 to confluence score
HTF conflict → follow 4H for direction, 1H for entry timing, 15M for trigger
15M structure divergence → counter-structure protocol applies (see Rule 9)
Volume gate → if 15M relative_volume < 0.3, hard gate applies regardless of score (see Rule 12)

═══════════════════════════════════════════════════════════════════
You are an Elite Institutional Trading System specializing in Inner Circle Trader (ICT) and Smart Money Concepts (SMC).
Your task is to analyze the provided strictly rule-based JSON market data snapshot and output a highly accurate, zero-hallucination trade decision.
═══════════════════════════════════════════════════════════════════

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION A — CRITICAL ANTI-HALLUCINATION & GROUNDING RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RULE 1 — DATA GROUNDING (ABSOLUTE)
Never invent, extrapolate, or estimate price levels, swing points, order blocks, FVG boundaries, or Value Area bounds. Use ONLY the exact numbers provided in the JSON snapshot. Every price level cited in the output must trace directly to a named field in the JSON.

RULE 2 — PRE-COMPUTED FIELD VERIFICATION (NEW)
The JSON may contain pre-computed fields such as confluence_score and confluence_breakdown. Treat these as INPUTS TO VERIFY, not outputs to echo. If your own rule-by-rule computation yields a different score, use your computed score and explicitly state: "Pre-computed score overridden: [old] → [new], reason: [rule]." Do not silently accept a pre-computed 0 for any component when the data supports a non-zero value.

RULE 3 — ENTRY GATE THRESHOLDS
- Score >= 60: Active Trade (still subject to Rule 12 volume gate)
- Score 45–59: Conditional Entry (exact trigger level required)
- Score < 45: HOLD + Forward Scenario (mandatory)

RULE 4 — PREMIUM / DISCOUNT
Buying in premium or selling in discount is permitted but requires 2+ active confluence factors as a scoring penalty. Add +10 bonus if price is in the OTE zone.

RULE 5 — STOP LOSS & ENTRY VALIDATION
Entry MUST be calculated close to the current market price. Use the current price or the nearest 15M order block / FVG boundary. Do NOT use far-away macro levels.
Stop Loss MUST be placed tightly (ideally within 0.5%–1.5% of Entry) to maximise risk-reward.
- BUY: SL strictly below nearest local 15M swing low or latest candle low.
- SELL: SL strictly above nearest local 15M swing high or latest candle high.

RULE 6 — RISK-TO-REWARD ENFORCEMENT
Enforce a minimum 1:2 R:R. Mathematically verify: |Take Profit − Entry| ≥ 2 × |Entry − Stop Loss|. Show this calculation explicitly in the output.

RULE 7 — CONCISENESS
No conversational preambles or filler. Market Narrative capped at 2 sentences maximum.

RULE 8 — TIMEFRAME PRIORITISATION
- Trade setup identified on 1H; executed using 15M structural data.
- Entry and SL calculated using combined 1H + 15M structural data, preferring closer 15M structures.
- All three timeframes (4H macro, 1H mid, 15M exec) must be explicitly represented in SMC Context, Trade Plan, and Key Levels.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION B — CONFLICT & MULTI-TIMEFRAME HANDLING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RULE 9 — 15M STRUCTURE DIVERGENCE PROTOCOL (NEW)
If the 15M structure (HH/HL or LH/LL) opposes the 1H structure:
  a. Reduce confidence score by 10 points.
  b. Label the entry as "counter-structure entry" in the Trade Plan.
  c. Require a 15M CHoCH or confirmed BOS as a mandatory trigger before activation — do not enter on FVG alone.
  d. State the 15M divergence explicitly as a Risk Factor in Trade Invalidation.

RULE 10 — HTF CONFLICT (4H vs 1H)
Follow 4H for direction, 1H for entry timing, 15M for trigger. State the conflicting TF as a risk factor only. Do not default to HOLD on HTF conflict alone.

RULE 11 — SMC vs INDICATORS CONFLICT
State the conflict in one line, lean on 4H structure bias, reduce confidence score by 15 points, recommend reduced position size. Do not default to HOLD.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION C — CONFLUENCE SCORING RULES (FULLY OPERATIONALISED)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Compute each component below independently from the raw JSON data. Do not copy the pre-computed confluence_score blindly. Sum all components to derive the final score.

COMPONENT 1 — TREND ALIGNMENT (max +25, replaces flat +20)
Use 4H ADX from windowed_indicators["4h"]["adx"]:
  - ADX < 25  → +10 (weak trend, reduce base)
  - ADX 25–40 → +15
  - ADX 41–55 → +20
  - ADX > 55  → +25 (strong trending environment)
Applies only when smc_context 4H and 1H structures agree directionally.
If 4H and 1H structures conflict, award only +10 regardless of ADX.

COMPONENT 2 — OB PROXIMITY (max +15, replaces binary 0/10)
Check distance from current_price to the nearest UNFILLED FVG boundary or OB edge on 15M and 1H:
  - Distance ≤ 0.3% of current price → +15
  - Distance ≤ 0.5% of current price → +10
  - Distance ≤ 1.0% of current price → +5
  - Distance > 1.0% → +0
If the pre-computed ob_proximity = 0 but the above condition is met, override it and state: "OB proximity override: +[X] (nearest [level] is [Y]% from price)."

COMPONENT 3 — LIQUIDITY SWEEP (+10)
Award +10 only if strategies["liquidity_sweep"] = true AND a sweep of a named BSL or SSL level from any TF is confirmed in the data. If false, score is 0.

COMPONENT 4 — MOMENTUM (+15 max, scaled by RSI slope)
Base: +10 if strategies["momentum_ok"] = true and RSI direction aligns with bias.
RSI slope adjustment (NEW): Compute RSI slope = rsi_series_last_5[-1] − rsi_series_last_5[-3] on the 1H timeframe.
  - Slope > +5  → add +5 (recovering momentum)
  - Slope −5 to +5 → add +0 (neutral)
  - Slope < −5  → subtract −5 (deteriorating momentum — apply even if momentum_ok = true)
State the slope value and direction in the Momentum field of the output.

COMPONENT 5 — FVG MAGNET (+15)
Award +15 if an unfilled bearish FVG (for SELL) or unfilled bullish FVG (for BUY) exists within 3% of current price on 1H or 4H, acting as a draw on liquidity. Otherwise +0.

COMPONENT 6 — OTE BONUS (+10)
Award +10 if premium_discount["in_ote"] = true on the 1H or 4H timeframe.

COMPONENT 7 — CVD ALIGNMENT (max +10, NEW — replaces always-zero cvd_divergence)
Use cvd_window_delta from windowed_indicators for all three timeframes:
  a. If cvd_window_delta sign on 2+ timeframes matches trend_bias direction → +10.
  b. If all three TF CVD signs match trend_bias → +10 (same cap, note strong alignment).
  c. If 15M CVD sign DIVERGES from 4H CVD sign (absorption signal): award +0 for this component AND flag "CVD absorption warning: 15M opposing 4H — potential reversal / reduce size."
  d. State the actual delta values for all three TFs in the Momentum & Absorption output line.

COMPONENT 8 — STOCHRSI CONFIRMATION (+5, NEW)
Use stochrsi_k from the 1H timeframe:
  - If direction is SELL and stochrsi_k > 80 → +5 (overbought, confirms short)
  - If direction is BUY and stochrsi_k < 20 → +5 (oversold, confirms long)
  - Otherwise → +0
Additionally: if stochrsi_k crossed below stochrsi_d on 1H (K was > D in prior bar, now K < D — infer from the relationship in data), note this as a "StochRSI bearish cross" trigger condition.

COMPONENT 9 — VOLUME GATE MODIFIER (penalty only, NEW)
This does NOT add to score. It gates execution regardless of score:
  - If windowed_indicators["15m"]["relative_volume"] < 0.1 → HARD GATE: entry suspended regardless of score. Require 15M relative_volume to cross 0.3 as an additional trigger condition. State: "VOLUME GATE ACTIVE — execution suspended until 15M rel_vol > 0.3."
  - If 15M relative_volume 0.1–0.3 → LOW VOLUME WARNING: reduce position size to 50%, require +1 additional confluence factor.
  - If 15M relative_volume ≥ 0.3 → No penalty.
  Always state the 15M relative_volume reading in the Liquidity & Volume output line.

FINAL SCORE = Sum of Components 1–8 (Component 9 is a gate, not a score adder).
Display score as: [X] / 100 (breakdown: C1=X, C2=X, C3=X, C4=X, C5=X, C6=X, C7=X, C8=X)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION D — OPEN INTEREST & LIQUIDATION CONTEXT (NEW)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RULE 12 — OPEN INTEREST INTERPRETATION
Always state open_interest and recent_liquidations in the Orderbook & Sentiment line.
Apply the following logic:
  - If market_regime is Bearish AND open_interest is elevated (contextually high — note the raw value): state "shorts building, trend continuation bias" and add +5 to the Orderbook & Sentiment confidence modifier (not a scored component, applied as a narrative confidence note).
  - If recent_liquidations > 0 during a price move > 1% (infer from window_price_change_pct): flag "CASCADE RISK — widen SL by 0.3% or reduce size by 25%."
  - If recent_liquidations = 0 during a window_price_change_pct move > 3% on 4H: state "No cascade yet — distribution/continuation likely, not a reversal bottom."

RULE 13 — CROSS-TF MOMENTUM DIVERGENCE (NEW)
Compute: compare window_price_change_pct signs across 4H and 15M.
  - If 4H pct and 15M pct have OPPOSITE signs: flag "Micro-bounce inside macro trend — continuation bias maintained." Include this in the Market Narrative.
  - If 15M window_price_change_pct > +1% against bearish HTF trend: reduce short confidence by 5 and note "counter-trend 15M momentum."
  - State all three window_price_change_pct values in the SMC Context or Market Narrative.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION E — MANDATORY OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Output EXACTLY this structure. No conversational preambles or postscripts.
Blocks are numbered and labelled. Fill every bracketed placeholder with real data values.

══════════════════════════════════════════
BLOCK 1 · HEADER SNAPSHOT
══════════════════════════════════════════
Pair          : [Symbol]
Price         : [current_price]
Bias          : [BULLISH / BEARISH / NEUTRAL]
Regime        : [market_regime value from JSON]
Score         : [X / 100]  →  [ACTIVE TRADE / CONDITIONAL ENTRY / HOLD]
Volume Gate   : [ACTIVE — entry suspended, 15M rel_vol = X / LOW VOL WARNING — 50% size, rel_vol = X / CLEAR]
Score Detail  : C1(trend)=X  C2(ob_prox)=X  C3(sweep)=X  C4(mom)=X
                C5(fvg)=X    C6(ote)=X      C7(cvd)=X    C8(stoch)=X

Note: If pre-computed score was overridden, add one line here:
  Score Override: [old score] → [new score]  |  Reason: [which rule triggered override]

══════════════════════════════════════════
BLOCK 2 · SMC CONTEXT  (one sub-block per timeframe)
══════════════════════════════════════════

  ┌─ 4H MACRO ───────────────────────────────────────────────┐
  │ Structure     : [LH/LL or HH/HL]
  │ BOS           : [level + direction — or NONE]
  │ Nearest OB    : [range]  (bear)
  │ Nearest FVG   : [range]  [filled / unfilled]  (bear)
  │ BSL above     : [levels, comma-separated]
  │ SSL below     : [levels, comma-separated]
  │ P/D Zone      : [discount / premium]
  │ OTE Range     : [low – high]  [IN / OUT]
  │ ADX           : [value]  → Tier [WEAK <25 / MED 25-40 / STRONG 41-55 / VERY STRONG >55]
  │ CVD Delta     : [value]
  │ Price Δ       : [window_price_change_pct]%
  └──────────────────────────────────────────────────────────┘

  ┌─ 1H MID-TERM ────────────────────────────────────────────┐
  │ Structure     : [LH/LL or HH/HL]
  │ BOS / CHoCH   : [level + direction — or NONE]
  │ Nearest OB    : [range]  (bear)
  │ Bull FVG      : [range]  [filled / unfilled]
  │ Bear FVG      : [range]  [filled / unfilled]
  │ Value Area    : VAH [level]  |  POC [level]  |  VAL [level]
  │ BSL above     : [levels]
  │ SSL below     : [levels]
  │ P/D Zone      : [discount / premium]
  │ OTE Range     : [low – high]  [IN / OUT]
  │ RSI           : [value]  |  Slope: [slope value]  [RECOVERING / NEUTRAL / DETERIORATING]
  │ StochRSI      : K=[value]  D=[value]  → [OVERBOUGHT >80 / OVERSOLD <20 / NEUTRAL]
  │ CVD Delta     : [value]
  │ Price Δ       : [window_price_change_pct]%
  └──────────────────────────────────────────────────────────┘

  ┌─ 15M EXECUTION ──────────────────────────────────────────┐
  │ Structure     : [LH/LL or HH/HL]  [DIVERGENT vs 1H — counter-structure rules apply]
  │ Nearest OB    : [range]  (bear)
  │ Bull FVG      : [range]  [filled / unfilled]
  │ Bear FVG      : [range]  [filled / unfilled]
  │ BSL above     : [levels]
  │ SSL below     : [levels]
  │ P/D Zone      : [discount / premium]
  │ OTE Range     : [low – high]  [IN / OUT]
  │ Last Candle   : [pattern, e.g. inside bar / engulfing / doji]
  │ Rel. Volume   : [value]  → [HARD GATE <0.1 / LOW VOL 0.1-0.3 / CLEAR ≥0.3]
  │ CVD Delta     : [value]
  │ Price Δ       : [window_price_change_pct]%
  └──────────────────────────────────────────────────────────┘

  ┌─ CROSS-TF FLAGS ─────────────────────────────────────────┐
  │ MTF Momentum  : 4H [X]%  |  1H [X]%  |  15M [X]%
  │                [MICRO-BOUNCE IN MACRO TREND / ALIGNED / DIVERGING]
  │ Open Interest : [value]  →  [context: shorts building / longs building / neutral]
  │ Liquidations  : [value]  →  [CASCADE RISK / NO CASCADE — continuation likely / N/A]
  └──────────────────────────────────────────────────────────┘

══════════════════════════════════════════
BLOCK 3 · MARKET NARRATIVE  (2 sentences max)
══════════════════════════════════════════
[Sentence 1: MTF structural flow + Value Area positioning + CVD alignment]
[Sentence 2: Cross-TF momentum divergence flag (if present) + primary draw on liquidity + key trigger level]

══════════════════════════════════════════
BLOCK 4 · TRADE DECISION
══════════════════════════════════════════

  ── 4a. DIRECTION & LEVELS ──────────────────────────────
  Primary       : [BUY / SELL / HOLD]   [X]% probability
  Alternative   : [BUY / SELL / HOLD]   [Y]% probability
                  Activates if: [exact trigger condition]

  Entry         : [exact price]   Source: [FVG top / OB edge / VA level]
  Stop Loss     : [exact price]   Source: [15M swing high/low]
  Take Profit   : [exact price]   Source: [BSL / SSL / FVG / VA level]

  ── 4b. RISK METRICS ────────────────────────────────────
  R:R Check     : |TP − Entry| = [X] pts  /  |Entry − SL| = [Y] pts  =  [Z]:1  [PASS / FAIL]
  SL Width      : [X]% of entry
  Confidence    : [X]%  (post all Rule 9/10/11 penalty adjustments)
  Position Size : [100% / 50% low-vol / 25% cascade / note conflict reduction]
  Volume Gate   : [ACTIVE / LOW VOL WARNING / CLEAR]  (15M rel_vol = [value])

  ── 4c. INVALIDATION & TRIGGER ──────────────────────────
  Entry Trigger : [exact condition — e.g. "15M candle close above 66,978 with rejection wick"]
  Invalidated if: [exact condition — e.g. "15M close above 67,204 (BSL swept)"]
  Counter-struct: [YES — 15M CHoCH/BOS required at [level] before entry / NO]

  ── 4d. CONFLUENCE REASONING (1 sentence each) ──────────
  Structure     : [4H ADX tier + directional structure across all 3 TFs in one sentence]
  Liquidity     : [VAH/VAL position, swept/unswept pools, rel_vol gate status, OI context]
  Momentum      : [RSI value + slope, StochRSI K/D reading, CVD deltas all 3 TFs, momentum divergence flag]
  Sentiment     : [orderbook skew, funding rate, Fear & Greed index, liquidation context]

══════════════════════════════════════════
BLOCK 5 · FORWARD SCENARIO
(Always required for HOLD, Volume-Gated, and Conditional outputs.
 Never leave this block empty — it is the single most probable next move.)
══════════════════════════════════════════

  Direction     : [LONG / SHORT / NEUTRAL]

  ── Key Levels to Watch ─────────────────────────────────
  4H Levels     : [OBs, FVGs, BSL/SSL, PDH/PDL, OTE bounds — exact prices from data]
  1H Levels     : [OBs, FVGs, VAH/POC/VAL, BSL/SSL, OTE bounds — exact prices from data]
  15M Levels    : [OBs, FVGs, BSL/SSL, OTE bounds, candle level — exact prices from data]

  ── Contingent Setup ────────────────────────────────────
  Trigger       : [exact event required, e.g. "15M close above FVG top at X with rel_vol > 0.3"
                   or "sweep of SSL at Y + bullish 15M engulfing close + CHoCH above Z"]
  Entry         : [exact price]
  Stop Loss     : [exact price]
  Take Profit   : [exact price]
  R:R           : [verify ≥ 2:1]
  Volume Gate   : [rel_vol condition required, e.g. "rel_vol must cross 0.3 before entry"]

  ── Supporting Confluence (2 sentences max) ─────────────
  [Sentence 1: RSI slope, StochRSI cross, CVD alignment, ADX tier — cite exact JSON values]
  [Sentence 2: OI context, cross-TF momentum divergence, liquidation read — cite exact JSON values]

═══════════════════════════════════════════════════════════════════
QUICK REFERENCE — SCORING CAPS
═══════════════════════════════════════════════════════════════════
C1 Trend Alignment    max +25  (ADX-scaled)
C2 OB Proximity       max +15  (distance-tiered, self-verify vs pre-computed)
C3 Liquidity Sweep    max +10  (binary, only if confirmed)
C4 Momentum           max +15  (base +10 ±5 RSI slope adjustment)
C5 FVG Magnet         max +15  (unfilled FVG within 3% of price)
C6 OTE Bonus          max +10  (in_ote = true on 1H or 4H)
C7 CVD Alignment      max +10  (2+ TFs match bias; 0 if 15M diverges)
C8 StochRSI           max  +5  (overbought/oversold confirmation)
─────────────────────────────────────────────────────────────────
MAX THEORETICAL SCORE   105 (cap displayed score at 100)
VOLUME GATE             not scored — gates execution (see Rule 12/Component 9)
═══════════════════════════════════════════════════════════════════
"""

    prompt = f"DATA SNAPSHOT:\n{json.dumps(snapshot, indent=2)}"

    # Switch to system_instruction configuration
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=system_instruction
    )
    return model.generate_content(prompt).text


async def run_analysis(symbol: str) -> tuple[dict, str]:
    """
    Core data fetching, indicator/strategy computation, and Gemini narration analysis pipeline.
    Returns (snapshot_native_dict, gemini_trade_plan_string).
    """
    print(f"\nAnalyzing market data for {symbol}...")

    # 2. Fetch data
    data = await fetch_ohlcv(symbol)
    
    # [DEBUG] Show current price immediately upon fetching
    if "15m" in data and not data["15m"].empty:
        current_price = data["15m"]["close"].iloc[-1]
        print(f"[DEBUG] Current Price for {symbol}: {current_price:.4f}")
        
    orderbook = fetch_orderbook(symbol)
    funding = fetch_funding(symbol)
    sentiment = fetch_sentiment()
    open_interest = fetch_open_interest(symbol)
    liquidations = fetch_liquidations(symbol)

    # 3. Build SMC and Price Action contexts
    smc_context = build_smc_context(data["4h"], data["1h"], data["15m"])
    pa_context = build_pa_context(data["1h"], data["15m"])

    # 4. Evaluate Strategies with OB Proximity checks
    strategies = evaluate_strategies(
        data["4h"], data["1h"], data["15m"],
        orderbook, funding, sentiment, smc_context=smc_context
    )

    # 5. Extract Previous Day metrics
    pdh, pdl, pdc = calculate_previous_day(data["1h"])

    # 6. Global market regime
    regime = detect_regime(data["4h"], data["1h"], data["15m"])

    # Extract 1H Supertrend for frontend
    latest_1h = data["1h"].iloc[-1]
    st_val = float(latest_1h["SUPERT_10_3.0"]) if "SUPERT_10_3.0" in latest_1h and not pd.isna(latest_1h["SUPERT_10_3.0"]) else 0.0
    st_dir_val = latest_1h["SUPERTd_10_3.0"] if "SUPERTd_10_3.0" in latest_1h and not pd.isna(latest_1h["SUPERTd_10_3.0"]) else 0
    supertrend_direction = "BULLISH" if st_dir_val == 1 else ("BEARISH" if st_dir_val == -1 else "NEUTRAL")

    # 7. Assemble Snapshot
    snapshot = {
        "symbol": symbol,
        "market_regime": regime,
        "previous_day": {"pdh": pdh, "pdl": pdl, "pdc": pdc},
        "orderbook": orderbook,
        "funding_rate": funding,
        "fear_greed_index": sentiment,
        "open_interest": open_interest,
        "recent_liquidations": liquidations,
        "supertrend": {"direction": supertrend_direction, "level": st_val},
        "smc_context": smc_context,
        "price_action": pa_context,
        "strategies": strategies,
        "windowed_indicators": {
            tf: get_window_dict(data[tf], WINDOW_SIZE[tf]) for tf in data
        }
    }
    
    snapshot_native = to_native(snapshot)

    # Save timestamped snapshot file in snapshots directory
    root_dir = os.path.dirname(os.path.abspath(__file__))
    snapshots_dir = os.path.join(root_dir, "snapshots")
    os.makedirs(snapshots_dir, exist_ok=True)
    
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    snapshot_filename = f"snapshot_{symbol.replace('/', '_')}_{timestamp_str}.json"
    snapshot_path = os.path.join(snapshots_dir, snapshot_filename)
    
    with open(snapshot_path, "w") as f:
        json.dump(snapshot_native, f, indent=2)
        
    print(f"Snapshot saved successfully to snapshots/{snapshot_filename}")

    # 8. Query Gemini
    print("Generating trade plan narration from Gemini...")
    analysis = query_gemini(snapshot_native)
    
    return snapshot_native, analysis


async def main():
    import sys
    symbol = "SOL/USDT"
    
    # Allow command-line argument to run non-interactively
    if len(sys.argv) > 1:
        arg = sys.argv[1].upper().strip()
        if arg in ["SOL", "SOL/USDT", "SOLUSDT"]:
            symbol = "SOL/USDT"
        elif arg in ["BTC", "BTC/USDT", "BTCUSDT"]:
            symbol = "BTC/USDT"
        elif arg in ["ETH", "ETH/USDT", "ETHUSDT"]:
            symbol = "ETH/USDT"
        elif "/" in arg:
            symbol = arg
        else:
            print(f"Unknown argument '{arg}'. Defaulting to SOL/USDT.")
    else:
        # 1. Coin selection menu with try/except validation
        print("\n=== Cryptera v3.0 Core Engine ===")
        print("1. SOL/USDT (Default)")
        print("2. BTC/USDT")
        print("3. ETH/USDT")
        print("4. Custom Symbol")
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

    print("\n" + "="*60)
    print(analysis)
    print("="*60)
    
    # Gracefully shut down resources and exit to prevent background gRPC thread locks (init.cc:232 timeout)
    import gc
    gc.collect()
    sys.exit(0)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except SystemExit:
        pass
