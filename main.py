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


async def fetch_ohlcv(symbol):
    """
    Fetch OHLCV candles with automatic retry logic (3 attempts with exponential backoff).
    Fetch all timeframes first, then compute indicators to avoid partial state.
    """
    exchange = ccxt.binance({"enableRateLimit": True})
    raw_dfs = {}

    try:
        for tf in TIMEFRAMES:
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
    finally:
        await exchange.close()

    # Compute indicators only after all timeframes are successfully fetched
    data = {}
    for tf, df in raw_dfs.items():
        data[tf] = calculate_indicators(df, timeframe=tf)
        
    return data


def fetch_orderbook(symbol):
    r = requests.get(
        "https://fapi.binance.com/fapi/v1/depth",
        params={"symbol": symbol.replace("/", ""), "limit": 5}
    ).json()

    bid_vol = sum(float(b[1]) for b in r["bids"])
    ask_vol = sum(float(a[1]) for a in r["asks"])
    skew = (bid_vol - ask_vol) / max(bid_vol + ask_vol, 1)

    return {"bid_vol": bid_vol, "ask_vol": ask_vol, "skew": skew}


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
Score >= 60 → Active trade
Score 45–59 → Conditional entry (exact trigger required)
Score < 45 → HOLD + forward scenario (mandatory)
Premium/Discount conflict → +1 confluence factor required, not a block
OTE zone → +10 to confluence score
HTF conflict → follow 4H, note risk, reduce size

You are an Elite Institutional Trading System specializing in Inner Circle Trader (ICT) and Smart Money Concepts (SMC).
Your task is to analyze the provided strictly rule-based JSON market data snapshot and output a highly accurate, zero-hallucination trade decision.

CRITICAL RULES TO ELIMINATE HALLUCINATION & ENSURE ACCURACY:
1. GROUNDING: Never invent, extrapolate, or estimate price levels, swing points, order blocks, FVG boundaries, or Value Area bounds. Use ONLY the exact numbers provided in the JSON snapshot.
2. ENTRY GATE THRESHOLDS:
   - Score >= 60: Active Trade.
   - Score 45–59: Conditional Entry (Exact trigger level required).
   - Score < 45: HOLD + Forward Scenario (Mandatory).
3. PREMIUM/DISCOUNT RULE:
   - Buying (BUY/Long) in premium or selling (SELL/Short) in discount is permitted but requires at least 2+ active confluence factors as a scoring penalty.
   - Add +10 bonus to the confluence score if price is in the OTE zone.
4. STOP LOSS VALIDATION:
   - For a BUY: Stop Loss MUST be placed strictly below the nearest key swing low / sell-side liquidity level provided in the data.
   - For a SELL: Stop Loss MUST be placed strictly above the nearest key swing high / buy-side liquidity level provided in the data.
5. RISK-TO-REWARD (R:R): Enforce a minimum 1:2 R:R. You must mathematically verify that: |Take Profit - Entry| >= 2 * |Entry - Stop Loss|.
6. CONCISENESS & TOKEN OPTIMIZATION: Do not include polite conversational preambles, introductory filler, or generic explanations. Keep the narrative highly compact. Market Narrative is strictly capped at 2 sentences max.
7. TIMEFRAME PRIORITIZATION & CALCULATIONS:
   - Prioritize identifying trades on the 1H timeframe as the primary trade setup and execution timeframe.
   - Entry and Stop Loss (SL) MUST be calculated using a combined analysis of 1H and 15m structural data (e.g. entry inside a 1H/15m OB or FVG, SL placed strictly below/above a 1H/15m swing low/high).
   - Sourced Key Levels (support/resistance boundaries, order blocks, Fair Value Gaps, and BSL/SSL liquidity pools) MUST be explicitly listed and detailed from ALL three timeframes (4H macro levels, 1H mid-term structures, and 15m low-timeframe zones) concurrently in your output sections (specifically inside the SMC Context, Trade Plan, and Key Levels to Watch).

CONFLICT & MULTI-TIMEFRAME HANDLING:
- HTF Conflicts: Replace mandatory probability justifications with: follow 4H for direction, 1H for entry timing, 15M for trigger. State the conflicting TF as a risk factor only.
- SMC vs Indicators Conflict: State the conflict in one line, lean on 4H structure bias, reduce confidence score by 15 points, recommend reduced position size, and do not default to HOLD.
- Premium/Discount vs Structure Conflict (e.g. discount zone + bearish 4H): Name it as an explicit risk factor and require an extra (+1) confluence factor to authorize a setup.

OUTPUT FORMAT:
Ensure you output EXACTLY this structure with NO conversational preambles or postscripts:

PAIR: [Symbol]
CURRENT BIAS: [BULLISH / BEARISH / NEUTRAL]

SMC Context: Structure [HH/HL, etc.], Nearest Bull OB [range], Nearest FVG [range], Liquidity above [levels], Premium/Discount zone [premium/discount], 1H Value Area [VAH to VAL]

MARKET NARRATIVE:
[Max 2 extremely concise, high-density technical sentences outlining MTF flow, Value Area positioning, and Liquidity Sweep / Reclaim signals]

TRADE PLAN:
- Primary Scenario: [BUY / SELL / HOLD with X% probability]
- Alternative Scenario: [Y% probability]
- Current Bias: [BULLISH / BEARISH / NEUTRAL]
- Entry: [Exact price from OB/FVG/Value Area or N/A if HOLD]
- Stop Loss: [Exact price based on swing low/high or N/A if HOLD]
- Take Profit: [Exact price targeting BSL/SSL or N/A if HOLD]
- Confidence: [Score 1-100%]
- Trade Invalidation/Trigger: [Condition or trigger level]
- Structure: [1 concise sentence reasoning trend & structure]
- Liquidity & Volume: [1 concise sentence reasoning VAH/VAL or swept pools]
- Momentum & Absorption: [1 concise sentence reasoning RSI/MACD or Sweep & Reclaim]
- Orderbook & Sentiment: [1 concise sentence reasoning orderbook skew or funding]

HYPOTHETICAL IF-THEN SETUP (FOR UNCERTAIN/HOLD MARKETS OR CONTINUATION):
[Note: HOLD must never be a terminal answer — always provide this section with the single most probable next move and its exact trigger level]
- Most Possible Price Direction: [LONG / SHORT / NEUTRAL]
- Key Levels to Watch: [List exact support/resistance levels, VAH/VAL bounds, order blocks, FVG boundaries, PDH/PDL, or volume POC from the data]
- Trade Setup & Trigger Condition: [Outline the exact setup and price event that must occur, e.g., 'A 15m candle close above the bearish FVG top at X' or 'A sweep of equal lows at Y followed by a bullish engulfing close']
- Contingent Entry Level: [Exact price from data]
- Contingent Invalidation (SL): [Exact price from data]
- Contingent Target (TP): [Exact price from data]
- Supporting Data & Confluence: [1-2 sentences detailing the precise technical indicators, volume profile, orderbook skew, or structural confluence backing this potential setup]
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

    # 7. Assemble Snapshot
    snapshot = {
        "symbol": symbol,
        "market_regime": regime,
        "previous_day": {"pdh": pdh, "pdl": pdl, "pdc": pdc},
        "orderbook": orderbook,
        "funding_rate": funding,
        "fear_greed_index": sentiment,
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
