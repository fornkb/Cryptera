"""
Cryptera v3.0 - Smart Money Concepts (SMC) Engine
"""

import pandas as pd
import numpy as np

def find_swings(df: pd.DataFrame, left=5, right=5):
    """
    Find swing highs and swing lows without lookahead lag on the latest closed candle.
    Returns the last 5 swing highs and swing lows as list of (timestamp, price) tuples.
    """
    swing_highs = []
    swing_lows = []
    n = len(df)
    
    # We must scan up to n - right to avoid lookahead on unconfirmed candles
    for i in range(left, n - right):
        val_high = df['high'].iloc[i]
        val_low = df['low'].iloc[i]
        
        # Check Swing High
        is_high = True
        for j in range(i - left, i + right + 1):
            if df['high'].iloc[j] > val_high:
                is_high = False
                break
        if is_high:
            ts = df['timestamp'].iloc[i]
            ts_str = ts.isoformat() if hasattr(ts, 'isoformat') else str(ts)
            swing_highs.append((ts_str, float(val_high)))
            
        # Check Swing Low
        is_low = True
        for j in range(i - left, i + right + 1):
            if df['low'].iloc[j] < val_low:
                is_low = False
                break
        if is_low:
            ts = df['timestamp'].iloc[i]
            ts_str = ts.isoformat() if hasattr(ts, 'isoformat') else str(ts)
            swing_lows.append((ts_str, float(val_low)))
            
    return swing_highs[-5:], swing_lows[-5:]

def get_market_structure(swing_highs, swing_lows, df=None):
    """
    Compare consecutive swings to determine structure and return trend, BOS, and CHoCH.
    Enforces body closes for structural breaks to distinguish from liquidity sweeps.
    """
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return "NEUTRAL", {"level": None, "direction": None}, {"detected": False, "direction": None, "level": None}
    
    ts_h1, sh1 = swing_highs[-2]
    ts_h2, sh2 = swing_highs[-1]
    ts_l1, sl1 = swing_lows[-2]
    ts_l2, sl2 = swing_lows[-1]
    
    high_state = "LH"
    if sh2 > sh1:
        high_state = "HH"
        if df is not None and not df.empty:
            recent_closes = df['close'].tail(100)
            if not (recent_closes > sh1).any():
                high_state = "LH"  # Swept but failed to close
                
    low_state = "HL" if sl2 > sl1 else "LL"
    if sl2 < sl1:
        if df is not None and not df.empty:
            recent_closes = df['close'].tail(100)
            if not (recent_closes < sl1).any():
                low_state = "HL"  # Swept but failed to close
                
    structure = f"{high_state}/{low_state}"
    
    bos_level = None
    bos_dir = None
    
    if df is not None and not df.empty:
        current_close = float(df['close'].iloc[-1])
        # Bullish BOS: price close breaks previous swing high
        if current_close > sh1:
            bos_level = sh1
            bos_dir = "BULLISH"
        # Bearish BOS: price close breaks previous swing low
        elif current_close < sl1:
            bos_level = sl1
            bos_dir = "BEARISH"
    else:
        if sh2 > sh1:
            bos_level = sh1
            bos_dir = "BULLISH"
        elif sl2 < sl1:
            bos_level = sl1
            bos_dir = "BEARISH"
            
    # CHoCH Detection (Enforces close)
    choch = {"detected": False, "direction": None, "level": None}
    if df is not None and not df.empty:
        current_close = float(df['close'].iloc[-1])
        if structure == "HH/HL" and current_close < sl2:
            choch = {"detected": True, "direction": "BEARISH", "level": sl2}
        elif structure == "LH/LL" and current_close > sh2:
            choch = {"detected": True, "direction": "BULLISH", "level": sh2}
            
    return structure, {"level": bos_level, "direction": bos_dir}, choch

def find_order_blocks(df: pd.DataFrame, swing_highs, swing_lows, lookback=50):
    """
    Find unmitigated Order Blocks (OBs):
    - Bearish OB: Last green candle before a bearish impulse (>1.5% drop in <=3 candles)
    - Bullish OB: Last red candle before a bullish impulse (>1.5% rise in <=3 candles)
    - Filter out mitigated ones (where price has closed through them)
    - Flags OBs near High Volume Nodes (VAH/VAL/POC)
    """
    from price_action import get_value_area
    va = get_value_area(df, lookback=lookback)
    vah, val, poc = va['vah'], va['val'], va['poc']
    
    bullish_obs = []
    bearish_obs = []
    
    n = len(df)
    start_idx = max(3, n - lookback)
    
    for i in range(start_idx, n):
        # 1. Check for Bullish Impulse (upward expansion based on ATR or 1.5% fallback in <= 3 candles)
        bullish_impulse = False
        impulse_start_idx = -1
        
        for k in [1, 2, 3]:
            prev_close = df['close'].iloc[i-k]
            curr_close = df['close'].iloc[i]
            
            # Dynamic volatility normalization using ATR
            curr_atr = df['ATR'].iloc[i-k] if 'ATR' in df.columns else np.nan
            if pd.isna(curr_atr) or curr_atr <= 0:
                # Fallback to standard 1.5% relative move
                threshold = prev_close * 0.015
            else:
                # Dynamic threshold based on ATR
                threshold = 2.0 * curr_atr
                
            if (curr_close - prev_close) >= threshold:
                bullish_impulse = True
                impulse_start_idx = i - k
                break
                
        if bullish_impulse:
            # Last red candle before the impulse start
            for j in range(impulse_start_idx - 1, max(0, impulse_start_idx - 10), -1):
                if df['close'].iloc[j] < df['open'].iloc[j]:
                    ob_high = float(df['high'].iloc[j])
                    ob_low = float(df['low'].iloc[j])
                    ob_ts = df['timestamp'].iloc[j].isoformat() if hasattr(df['timestamp'].iloc[j], 'isoformat') else str(df['timestamp'].iloc[j])
                    
                    exists = any(ob['timestamp'] == ob_ts for ob in bullish_obs)
                    if not exists:
                        # Check if OB aligns with Value Area High, Low, or POC (within 0.5%)
                        is_hvn = False
                        for level in [vah, val, poc]:
                            if abs((ob_high + ob_low)/2 - level) / level <= 0.005:
                                is_hvn = True
                                break
                                
                        bullish_obs.append({
                            "timestamp": ob_ts,
                            "top": ob_high,
                            "bottom": ob_low,
                            "created_at_idx": j,
                            "high_volume_node": is_hvn
                        })
                    break
                    
        # 2. Check for Bearish Impulse (downward expansion based on ATR or 1.5% fallback in <= 3 candles)
        bearish_impulse = False
        impulse_start_idx = -1
        
        for k in [1, 2, 3]:
            prev_close = df['close'].iloc[i-k]
            curr_close = df['close'].iloc[i]
            
            curr_atr = df['ATR'].iloc[i-k] if 'ATR' in df.columns else np.nan
            if pd.isna(curr_atr) or curr_atr <= 0:
                threshold = prev_close * 0.015
            else:
                threshold = 2.0 * curr_atr
                
            if (prev_close - curr_close) >= threshold:
                bearish_impulse = True
                impulse_start_idx = i - k
                break
                
        if bearish_impulse:
            # Last green candle before the impulse start
            for j in range(impulse_start_idx - 1, max(0, impulse_start_idx - 10), -1):
                if df['close'].iloc[j] > df['open'].iloc[j]:
                    ob_high = float(df['high'].iloc[j])
                    ob_low = float(df['low'].iloc[j])
                    ob_ts = df['timestamp'].iloc[j].isoformat() if hasattr(df['timestamp'].iloc[j], 'isoformat') else str(df['timestamp'].iloc[j])
                    
                    exists = any(ob['timestamp'] == ob_ts for ob in bearish_obs)
                    if not exists:
                        # Check if OB aligns with Value Area High, Low, or POC (within 0.5%)
                        is_hvn = False
                        for level in [vah, val, poc]:
                            if abs((ob_high + ob_low)/2 - level) / level <= 0.005:
                                is_hvn = True
                                break
                                
                        bearish_obs.append({
                            "timestamp": ob_ts,
                            "top": ob_high,
                            "bottom": ob_low,
                            "created_at_idx": j,
                            "high_volume_node": is_hvn
                        })
                    break
                    
    # Filter for unmitigated Order Blocks
    # Mitigated if subsequent candle closes break the OB high/low
    unmitigated_bullish = []
    for ob in bullish_obs:
        ob_bottom = ob['bottom']
        created_idx = ob['created_at_idx']
        subsequent_closes = df['close'].iloc[created_idx + 1:]
        
        if subsequent_closes.empty or subsequent_closes.min() >= ob_bottom:
            unmitigated_bullish.append({
                "timestamp": ob['timestamp'],
                "top": ob['top'],
                "bottom": ob['bottom'],
                "high_volume_node": ob['high_volume_node']
            })
            
    unmitigated_bearish = []
    for ob in bearish_obs:
        ob_top = ob['top']
        created_idx = ob['created_at_idx']
        subsequent_closes = df['close'].iloc[created_idx + 1:]
        
        if subsequent_closes.empty or subsequent_closes.max() <= ob_top:
            unmitigated_bearish.append({
                "timestamp": ob['timestamp'],
                "top": ob['top'],
                "bottom": ob['bottom'],
                "high_volume_node": ob['high_volume_node']
            })
            
    return {
        "bullish": unmitigated_bullish[-3:] if unmitigated_bullish else [],
        "bearish": unmitigated_bearish[-3:] if unmitigated_bearish else []
    }

def find_fvg(df: pd.DataFrame, lookback=50):
    """
    Find 3-candle Fair Value Gaps (FVG) and return the nearest unfilled FVG above and below price.
    - Bullish FVG: df['high'].iloc[i-2] < df['low'].iloc[i]
    - Bearish FVG: df['low'].iloc[i-2] > df['high'].iloc[i]
    """
    bullish_fvgs = []
    bearish_fvgs = []
    n = len(df)
    start_idx = max(2, n - lookback)
    current_price = float(df['close'].iloc[-1])
    
    for i in range(start_idx, n):
        # Bullish FVG
        high_2 = float(df['high'].iloc[i-2])
        low_0 = float(df['low'].iloc[i])
        if low_0 > high_2:
            # Check if subsequently filled (mitigated) up to n-2
            subsequent = df.iloc[i+1 : n-1]
            filled = False
            if not subsequent.empty:
                if subsequent['low'].min() <= high_2:
                    filled = True
            bullish_fvgs.append({
                "top": low_0,
                "bottom": high_2,
                "timestamp": df['timestamp'].iloc[i].isoformat() if hasattr(df['timestamp'].iloc[i], 'isoformat') else str(df['timestamp'].iloc[i]),
                "filled": filled
            })
                
        # Bearish FVG
        low_2 = float(df['low'].iloc[i-2])
        high_0 = float(df['high'].iloc[i])
        if low_2 > high_0:
            # Check if subsequently filled up to n-2
            subsequent = df.iloc[i+1 : n-1]
            filled = False
            if not subsequent.empty:
                if subsequent['high'].max() >= low_2:
                    filled = True
            bearish_fvgs.append({
                "top": low_2,
                "bottom": high_0,
                "timestamp": df['timestamp'].iloc[i].isoformat() if hasattr(df['timestamp'].iloc[i], 'isoformat') else str(df['timestamp'].iloc[i]),
                "filled": filled
            })
                
    # Find nearest unfilled FVG below current price (Bullish)
    nearest_bull = None
    below_fvgs = [f for f in bullish_fvgs if not f['filled'] and f['top'] <= current_price]
    if below_fvgs:
        nearest_bull = max(below_fvgs, key=lambda x: x['top'])
        
    # Find nearest unfilled FVG above current price (Bearish)
    nearest_bear = None
    above_fvgs = [f for f in bearish_fvgs if not f['filled'] and f['bottom'] >= current_price]
    if above_fvgs:
        nearest_bear = min(above_fvgs, key=lambda x: x['bottom'])
        
    return {
        "nearest_bullish_fvg": nearest_bull,
        "nearest_bearish_fvg": nearest_bear,
        "bullish_fvgs": bullish_fvgs[-5:] if bullish_fvgs else [],
        "bearish_fvgs": bearish_fvgs[-5:] if bearish_fvgs else []
    }

def find_liquidity_levels(swing_highs, swing_lows, tolerance=0.002):
    """
    Cluster swing highs and lows within tolerance (0.2%) to find equal highs/lows.
    Returns top 3 buy-side (highs) and sell-side (lows) liquidity levels.
    """
    high_prices = sorted([s[1] for s in swing_highs])
    low_prices = sorted([s[1] for s in swing_lows])
    
    # Cluster Highs (Buy-Side Liquidity)
    bsl_levels = []
    used_highs = set()
    for i, h1 in enumerate(high_prices):
        if i in used_highs:
            continue
        cluster = [h1]
        for j, h2 in enumerate(high_prices[i+1:]):
            idx = i + 1 + j
            if idx in used_highs:
                continue
            if (h2 - h1) / h1 <= tolerance:
                cluster.append(h2)
                used_highs.add(idx)
        bsl_levels.append(max(cluster))
        
    # Cluster Lows (Sell-Side Liquidity)
    ssl_levels = []
    used_lows = set()
    for i, l1 in enumerate(low_prices):
        if i in used_lows:
            continue
        cluster = [l1]
        for j, l2 in enumerate(low_prices[i+1:]):
            idx = i + 1 + j
            if idx in used_lows:
                continue
            if (l2 - l1) / l1 <= tolerance:
                cluster.append(l2)
                used_lows.add(idx)
        ssl_levels.append(min(cluster))
        
    return {
        "buy_side": sorted(bsl_levels, reverse=True)[:3],
        "sell_side": sorted(ssl_levels)[:3]
    }

def get_premium_discount(swing_high, swing_low, current_price):
    """
    Evaluate premium/discount zone and check if in OTE zone (61.8% to 78.6% retracement).
    """
    if swing_high <= swing_low:
        return {
            "zone": "neutral",
            "ote_zone": {"low": 0, "high": 0},
            "in_ote": False
        }
        
    range_size = swing_high - swing_low
    midpoint = swing_low + 0.5 * range_size
    
    zone = "premium" if current_price > midpoint else "discount"
    
    # OTE Retracement levels
    ote_low = swing_high - 0.786 * range_size
    ote_high = swing_high - 0.618 * range_size
    
    in_ote = ote_low <= current_price <= ote_high
    
    return {
        "zone": zone,
        "ote_zone": {"low": round(ote_low, 4), "high": round(ote_high, 4)},
        "in_ote": in_ote
    }

def build_smc_context(df_4h, df_1h, df_15m):
    """
    Build multi-timeframe SMC context for snapshot.
    """
    context = {}
    for tf, df in [("4h", df_4h), ("1h", df_1h), ("15m", df_15m)]:
        if df is None or df.empty:
            continue
            
        current_price = float(df['close'].iloc[-1])
        # Dynamic swing parameter selection to reduce latency on lower timeframes
        left_p, right_p = 5, 5
        if tf == "1h":
            left_p, right_p = 4, 3
        elif tf == "15m":
            left_p, right_p = 3, 2
            
        sh, sl = find_swings(df, left=left_p, right=right_p)
        struct, bos, choch = get_market_structure(sh, sl, df)
        obs = find_order_blocks(df, sh, sl)
        fvgs = find_fvg(df)
        liq = find_liquidity_levels(sh, sl)
        
        # Range of last 5 swings
        if sh and sl:
            max_h = max([x[1] for x in sh])
            min_l = min([x[1] for x in sl])
            pd_zone = get_premium_discount(max_h, min_l, current_price)
        else:
            pd_zone = {"zone": "neutral", "in_ote": False, "ote_zone": {"low": 0, "high": 0}}
            
        context[tf] = {
            "current_price": current_price,
            "structure": struct,
            "bos": bos,
            "choch": choch,
            "order_blocks": obs,
            "fvg": fvgs,
            "liquidity_levels": liq,
            "premium_discount": pd_zone
        }
        
    return context
