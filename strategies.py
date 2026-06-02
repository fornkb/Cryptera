"""
Rule-Based ICT / SMC Strategy Engine
"""

import pandas as pd
import numpy as np
from smc import find_swings

def check_ob_proximity(current_price, smc_context, ob_type, tolerance=0.005):
    """
    Verify if the current price is within tolerance (0.5%) of a valid unmitigated Order Block.
    ob_type should be 'bullish' or 'bearish'.
    """
    if not smc_context or "15m" not in smc_context:
        return False
        
    obs = smc_context["15m"].get("order_blocks", {})
    target_obs = obs.get(ob_type, [])
    
    if not target_obs:
        return False
        
    for ob in target_obs:
        top = ob["top"]
        bottom = ob["bottom"]
        if (bottom * (1 - tolerance)) <= current_price <= (top * (1 + tolerance)):
            return True
            
    return False

def check_fvg_proximity(current_price, smc_context, fvg_type, tolerance=0.01):
    """
    Verify if the current price is within tolerance (1.0%) of a valid unfilled FVG.
    fvg_type should be 'bullish' or 'bearish'.
    """
    if not smc_context or "15m" not in smc_context:
        return False
        
    fvgs = smc_context["15m"].get("fvg", {})
    if fvg_type == "bullish":
        nearest_bull = fvgs.get("nearest_bullish_fvg")
        if nearest_bull:
            dist = abs(current_price - nearest_bull["top"]) / current_price
            if dist <= tolerance:
                return True
    else:
        nearest_bear = fvgs.get("nearest_bearish_fvg")
        if nearest_bear:
            dist = abs(nearest_bear["bottom"] - current_price) / current_price
            if dist <= tolerance:
                return True
                
    return False

def detect_reversal(df_15m: pd.DataFrame, smc_context=None) -> dict:
    """
    Reversal Strategy:
    - Bullish Reversal: RSI < 30 and price is within 0.5% of a valid Bullish OB.
    - Bearish Reversal: RSI > 70 and price is within 0.5% of a valid Bearish OB.
    """
    latest = df_15m.iloc[-1]
    rsi = float(latest["RSI"])
    current_price = float(latest["close"])
    
    bullish_trigger = rsi < 30
    bearish_trigger = rsi > 70
    
    # Enforce OB proximity check if smc_context is provided
    if smc_context is not None:
        bullish_trigger = bullish_trigger and check_ob_proximity(current_price, smc_context, "bullish")
        bearish_trigger = bearish_trigger and check_ob_proximity(current_price, smc_context, "bearish")
        
    return {
        "bullish": bool(bullish_trigger),
        "bearish": bool(bearish_trigger)
    }

def detect_pullback(df_15m: pd.DataFrame, trend_bias="neutral", smc_context=None) -> dict:
    """
    Pullback Strategy:
    - Bullish Pullback: Trend is bullish, RSI is between 40 and 60, and price is near Bullish OB.
    - Bearish Pullback: Trend is bearish, RSI is between 40 and 60, and price is near Bearish OB.
    """
    latest = df_15m.iloc[-1]
    rsi = float(latest["RSI"])
    current_price = float(latest["close"])
    
    rsi_ok = 40 < rsi < 60
    
    bullish_trigger = rsi_ok and trend_bias == "bullish"
    bearish_trigger = rsi_ok and trend_bias == "bearish"
    
    if smc_context is not None:
        bullish_trigger = bullish_trigger and check_ob_proximity(current_price, smc_context, "bullish")
        bearish_trigger = bearish_trigger and check_ob_proximity(current_price, smc_context, "bearish")
        
    return {
        "bullish": bool(bullish_trigger),
        "bearish": bool(bearish_trigger)
    }

def detect_breakout(df_15m: pd.DataFrame, smc_context=None) -> dict:
    """
    Breakout Strategy:
    - Bullish Breakout: Close price breaks above last swing high, BB squeeze check met.
    - Bearish Breakout: Close price breaks below last swing low, BB squeeze check met.
    - BB squeeze check: BBB (bandwidth) was below 0.02 in the last 5 candles before current breakout.
    """
    if len(df_15m) < 10:
        return {"bullish": False, "bearish": False}
        
    latest = df_15m.iloc[-1]
    current_price = float(latest["close"])
    
    # Get swing highs/lows using smc logic (avoid lookahead)
    sh_list, sl_list = find_swings(df_15m, left=5, right=5)
    if not sh_list or not sl_list:
        return {"bullish": False, "bearish": False}
        
    last_sh = sh_list[-1][1]
    last_sl = sl_list[-1][1]
    
    # ATR Squeeze check: ATR_Z was < -0.5 in last 5 candles before current
    squeeze_ok = False
    if "ATR_Z" in df_15m.columns:
        # Check index -6 to -2 (the 5 candles before index -1)
        prev_atr_z = df_15m["ATR_Z"].iloc[-6:-1]
        squeeze_ok = (prev_atr_z < -0.5).any()
        
    bullish_trigger = current_price > last_sh and squeeze_ok
    bearish_trigger = current_price < last_sl and squeeze_ok
    
    if smc_context is not None:
        bullish_trigger = bullish_trigger and check_fvg_proximity(current_price, smc_context, "bullish")
        bearish_trigger = bearish_trigger and check_fvg_proximity(current_price, smc_context, "bearish")
        
    return {
        "bullish": bool(bullish_trigger),
        "bearish": bool(bearish_trigger)
    }

def detect_divergence(df_15m: pd.DataFrame, smc_context=None) -> dict:
    """
    Divergence Strategy:
    - Bullish Divergence: Price Lower Low, RSI Higher Low over last 10 candles (using swing points).
    - Bearish Divergence: Price Higher High, RSI Lower High over last 10 candles (using swing points).
    """
    sh_list, sl_list = find_swings(df_15m, left=5, right=5)
    if len(df_15m) < 10 or not sh_list or not sl_list:
        return {"bullish": False, "bearish": False}
        
    recent_10 = df_15m.tail(10)
    recent_timestamps = set(recent_10['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S').tolist())
    
    recent_sh = [s for s in sh_list if s[0][:19] in [t[:19] for t in recent_timestamps]]
    recent_sl = [s for s in sl_list if s[0][:19] in [t[:19] for t in recent_timestamps]]
    
    bullish_div = False
    bearish_div = False
    current_price = float(df_15m["close"].iloc[-1])
    
    # Bullish Divergence: Price Lower Low, RSI Higher Low
    if len(recent_sl) >= 2:
        recent_sl = sorted(recent_sl, key=lambda x: x[0])
        p1, p2 = recent_sl[-2][1], recent_sl[-1][1]
        ts1, ts2 = recent_sl[-2][0], recent_sl[-1][0]
        
        r1_rows = df_15m[df_15m['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S').str.startswith(ts1[:19])]
        r2_rows = df_15m[df_15m['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S').str.startswith(ts2[:19])]
        
        if not r1_rows.empty and not r2_rows.empty:
            rsi1 = float(r1_rows['RSI'].iloc[0])
            rsi2 = float(r2_rows['RSI'].iloc[0])
            if p2 < p1 and rsi2 > rsi1:
                bullish_div = True
                
    # Bearish Divergence: Price Higher High, RSI Lower High
    if len(recent_sh) >= 2:
        recent_sh = sorted(recent_sh, key=lambda x: x[0])
        p1, p2 = recent_sh[-2][1], recent_sh[-1][1]
        ts1, ts2 = recent_sh[-2][0], recent_sh[-1][0]
        
        r1_rows = df_15m[df_15m['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S').str.startswith(ts1[:19])]
        r2_rows = df_15m[df_15m['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S').str.startswith(ts2[:19])]
        
        if not r1_rows.empty and not r2_rows.empty:
            rsi1 = float(r1_rows['RSI'].iloc[0])
            rsi2 = float(r2_rows['RSI'].iloc[0])
            if p2 > p1 and rsi2 < rsi1:
                bearish_div = True
                
    if smc_context is not None:
        bullish_div = bullish_div and check_ob_proximity(current_price, smc_context, "bullish")
        bearish_div = bearish_div and check_ob_proximity(current_price, smc_context, "bearish")
        
    return {
        "bullish": bool(bullish_div),
        "bearish": bool(bearish_div)
    }

def evaluate_strategies(df_4h, df_1h, df_15m, orderbook, funding, sentiment, smc_context=None):
    """
    Main evaluation entry point. Combines all rules to see if we have valid confluences.
    """
    latest_15m = df_15m.iloc[-1]
    latest_4h = df_4h.iloc[-1]
    current_price = float(latest_15m["close"])

    # 4H Structure-based trend bias
    struct_4h = smc_context.get("4h", {}).get("structure", "NEUTRAL") if smc_context else "NEUTRAL"
    if "HH" in struct_4h or "HL" in struct_4h:
        trend = "bullish"
    elif "LH" in struct_4h or "LL" in struct_4h:
        trend = "bearish"
    else:
        trend = "bullish" if latest_4h["EMA_50"] > latest_4h["EMA_200"] else "bearish"

    # === STATEFUL LIQUIDITY SWEEP ===
    sweep = False
    for i in range(-5, 0):
        candle = df_15m.iloc[i]
        scan_before = df_15m.iloc[:i-1].tail(30)
        
        if trend == "bullish":
            lows = scan_before[scan_before["swing_low"]]["low"]
            if not lows.empty:
                key = lows.min()
                if candle["low"] < key and candle["close"] > key:
                    sweep = True
        else:
            highs = scan_before[scan_before["swing_high"]]["high"]
            if not highs.empty:
                key = highs.max()
                if candle["high"] > key and candle["close"] < key:
                    sweep = True

    # === DISPLACEMENT ===
    displacement = (
        latest_15m["VOL_Z"] > 1.0
        and latest_15m["ATR_Z"] > -0.2
        and (
            (trend == "bullish" and latest_15m["fvg_bull"])
            or (trend == "bearish" and latest_15m["fvg_bear"])
        )
    )

    # === PULLBACK, REVERSAL, BREAKOUT, DIVERGENCE SUB-STRATEGIES ===
    rev = detect_reversal(df_15m, smc_context)
    pb = detect_pullback(df_15m, trend, smc_context)
    bo = detect_breakout(df_15m, smc_context)
    div = detect_divergence(df_15m, smc_context)

    # Momentum, orderbook, sentiment and funding
    momentum_ok = (
        (trend == "bullish" and latest_15m["RSI"] < 65 and latest_15m["MACD"] > latest_15m["MACD_SIGNAL"])
        or (trend == "bearish" and latest_15m["RSI"] > 35 and latest_15m["MACD"] < latest_15m["MACD_SIGNAL"])
    )
    book_ok = orderbook["skew"] > 0 if trend == "bullish" else orderbook["skew"] < 0
    sentiment_ok = sentiment < 75 if trend == "bullish" else sentiment > 25
    funding_ok = abs(funding) < 0.01

    # Check if any sub-strategy triggers a valid bias direction matching trend
    strategy_bullish = rev["bullish"] or pb["bullish"] or bo["bullish"] or div["bullish"]
    strategy_bearish = rev["bearish"] or pb["bearish"] or bo["bearish"] or div["bearish"]
    
    strategy_triggered = (trend == "bullish" and strategy_bullish) or (trend == "bearish" and strategy_bearish)

    # Final setup is ready if we have a sweep + displacement confluences OR a strong sub-strategy trigger
    final_setup_ready = (
        (sweep and displacement and momentum_ok and book_ok and sentiment_ok and funding_ok)
        or (strategy_triggered and book_ok and sentiment_ok and funding_ok)
    )

    # === CONFLUENCE SCORE MATRIX (0-100) ===
    score_breakdown = {
        "trend_alignment": 0,
        "ob_proximity": 0,
        "liquidity_sweep": 0,
        "momentum": 0,
        "fvg_magnet": 0,
        "orderbook_funding": 0,
        "ote_bonus": 0
    }
    
    # 1. Trend Alignment (Max 20)
    # Only award 20 points if Trend AND Structure AND EMA agree; 0 points if they conflict.
    ema_50_15m = float(latest_15m["EMA_50"]) if "EMA_50" in latest_15m else current_price
    struct_1h = smc_context.get("1h", {}).get("structure", "NEUTRAL") if smc_context else "NEUTRAL"
    
    struct_bullish = "HH" in struct_1h
    struct_bearish = "LH" in struct_1h
    
    ema_bullish = current_price >= ema_50_15m
    ema_bearish = current_price <= ema_50_15m
    
    agree_bullish = (trend == "bullish") and struct_bullish and ema_bullish
    agree_bearish = (trend == "bearish") and struct_bearish and ema_bearish
    
    if agree_bullish or agree_bearish:
        score_breakdown["trend_alignment"] = 20
    else:
        score_breakdown["trend_alignment"] = 0
            
    # 2. OB Proximity (Max 20)
    has_ob_prox = check_ob_proximity(current_price, smc_context, "bullish" if trend == "bullish" else "bearish")
    if has_ob_prox:
        score_breakdown["ob_proximity"] = 20
        
    # 3. Liquidity Sweep & Reclaim (Max 20)
    if sweep:
        score_breakdown["liquidity_sweep"] = 20
        
    # 4. Momentum (Max 15)
    mom_points = 0
    rsi_val = float(latest_15m["RSI"]) if "RSI" in latest_15m else 50.0
    if trend == "bullish":
        if rsi_val < 65:
            mom_points += 7
        if "MACD" in latest_15m and "MACD_SIGNAL" in latest_15m and latest_15m["MACD"] > latest_15m["MACD_SIGNAL"]:
            mom_points += 8
    else:
        if rsi_val > 35:
            mom_points += 7
        if "MACD" in latest_15m and "MACD_SIGNAL" in latest_15m and latest_15m["MACD"] < latest_15m["MACD_SIGNAL"]:
            mom_points += 8
    score_breakdown["momentum"] = mom_points
    
    # 5. FVG Magnet (Max 15)
    has_fvg_magnet = False
    if smc_context and "15m" in smc_context:
        fvgs = smc_context["15m"].get("fvg", {})
        if trend == "bullish":
            nearest_bull = fvgs.get("nearest_bullish_fvg")
            if nearest_bull:
                dist = (current_price - nearest_bull["top"]) / current_price
                if 0 <= dist <= 0.01:
                    has_fvg_magnet = True
        else:
            nearest_bear = fvgs.get("nearest_bearish_fvg")
            if nearest_bear:
                dist = (nearest_bear["bottom"] - current_price) / current_price
                if 0 <= dist <= 0.01:
                    has_fvg_magnet = True
    if has_fvg_magnet:
        score_breakdown["fvg_magnet"] = 20
        
    # 6. Orderbook & Funding (Max 10)
    ob_fund_points = 0
    if book_ok:
        ob_fund_points += 5
    if funding_ok:
        ob_fund_points += 5
    score_breakdown["orderbook_funding"] = ob_fund_points
    
    # 7. OTE Zone Bonus (Max 10)
    in_ote = False
    if smc_context and "15m" in smc_context:
        in_ote = smc_context["15m"].get("premium_discount", {}).get("in_ote", False)
    if in_ote:
        score_breakdown["ote_bonus"] = 10
        
    total_score = min(100, sum(score_breakdown.values()))

    return {
        "trend_bias": trend,
        "liquidity_sweep": sweep,
        "displacement": displacement,
        "momentum_ok": momentum_ok,
        "orderbook_ok": book_ok,
        "sentiment_ok": sentiment_ok,
        "funding_ok": funding_ok,
        "strategies": {
            "reversal": rev,
            "pullback": pb,
            "breakout": bo,
            "divergence": div
        },
        "final_setup_ready": bool(final_setup_ready),
        "confluence_score": int(total_score),
        "confluence_breakdown": score_breakdown
    }
