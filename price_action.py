"""
Cryptera v3.0 - Price Action (PA) Engine
"""

import pandas as pd
import numpy as np
from smc import find_swings

def detect_candle_pattern(df: pd.DataFrame) -> str:
    """
    Check the last 3 candles for manual patterns (no ta-lib C dependency).
    Returns pattern name or "none".
    """
    if len(df) < 3:
        return "none"
        
    # candles index: -3 (oldest), -2 (middle), -1 (latest)
    c2 = df.iloc[-3]
    c1 = df.iloc[-2]
    c0 = df.iloc[-1]
    
    # helper properties
    def get_candle_props(c):
        body_size = abs(c['close'] - c['open'])
        candle_range = c['high'] - c['low'] if c['high'] - c['low'] > 0 else 0.001
        is_green = c['close'] > c['open']
        is_red = c['close'] < c['open']
        lower_wick = min(c['open'], c['close']) - c['low']
        upper_wick = c['high'] - max(c['open'], c['close'])
        return body_size, candle_range, is_green, is_red, lower_wick, upper_wick
        
    b0, r0, g0, r_0, lw0, uw0 = get_candle_props(c0)
    b1, r1, g1, r_1, lw1, uw1 = get_candle_props(c1)
    b2, r2, g2, r_2, lw2, uw2 = get_candle_props(c2)
    
    # 1. Doji
    if b0 / r0 <= 0.1:
        return "doji"
        
    # 2. Inside Bar
    if c0['high'] < c1['high'] and c0['low'] > c1['low']:
        return "inside_bar"
        
    # 3. Pin Bar Bullish
    # Body is in upper 30% of range, lower wick is at least 60% of range, small body
    if (lw0 / r0 >= 0.6) and (b0 / r0 <= 0.3):
        return "pin_bar_bull"
        
    # 4. Pin Bar Bearish
    # Body is in lower 30% of range, upper wick is at least 60% of range, small body
    if (uw0 / r0 >= 0.6) and (b0 / r0 <= 0.3):
        return "pin_bar_bear"
        
    # 5. Bullish Engulfing
    # c1 was red, c0 is green, c0 body completely covers c1 body
    if r_1 and g0 and (c0['close'] >= c1['open']) and (c0['open'] <= c1['close']):
        return "bullish_engulfing"
        
    # 6. Bearish Engulfing
    # c1 was green, c0 is red, c0 body completely covers c1 body
    if g1 and r_0 and (c0['close'] <= c1['open']) and (c0['open'] >= c1['close']):
        return "bearish_engulfing"
        
    # 7. Morning Star
    # c2 large red, c1 small body, c0 green closing >= 50% of c2 body
    if r_2 and (b1 / r2 <= 0.3) and g0 and (c0['close'] >= (c2['open'] + c2['close'])/2):
        return "morning_star"
        
    # 8. Evening Star
    # c2 large green, c1 small body, c0 red closing <= 50% of c2 body
    if g2 and (b1 / r2 <= 0.3) and r_0 and (c0['close'] <= (c2['open'] + c2['close'])/2):
        return "evening_star"
        
    return "none"

def get_poc(df: pd.DataFrame, lookback=50, buckets=100) -> float:
    """
    Compute Point of Control (POC) by bucketing close prices weighted by volume.
    """
    recent = df.tail(lookback)
    closes = recent['close'].values
    vols = recent['volume'].values
    
    min_price = closes.min()
    max_price = closes.max()
    
    if max_price == min_price:
        return float(min_price)
        
    hist, bin_edges = np.histogram(closes, bins=buckets, weights=vols)
    max_idx = np.argmax(hist)
    
    poc = (bin_edges[max_idx] + bin_edges[max_idx + 1]) / 2
    return float(poc)

def get_value_area(df: pd.DataFrame, lookback=50, buckets=100) -> dict:
    """
    Compute Value Area High (VAH) and Value Area Low (VAL) covering 70% of the volume profile.
    """
    recent = df.tail(lookback)
    closes = recent['close'].values
    vols = recent['volume'].values
    
    min_price = closes.min()
    max_price = closes.max()
    
    if max_price == min_price:
        return {
            "vah": float(max_price),
            "val": float(min_price),
            "poc": float(min_price)
        }
        
    hist, bin_edges = np.histogram(closes, bins=buckets, weights=vols)
    max_idx = np.argmax(hist)
    poc = (bin_edges[max_idx] + bin_edges[max_idx + 1]) / 2
    
    total_vol = hist.sum()
    if total_vol <= 0:
        return {
            "vah": float(poc),
            "val": float(poc),
            "poc": float(poc)
        }
        
    target_vol = total_vol * 0.70
    selected_bins = {max_idx}
    current_vol = hist[max_idx]
    
    left_idx = max_idx - 1
    right_idx = max_idx + 1
    
    while current_vol < target_vol:
        left_vol = hist[left_idx] if left_idx >= 0 else 0
        right_vol = hist[right_idx] if right_idx < buckets else 0
        
        if left_vol == 0 and right_vol == 0:
            break
            
        if left_vol >= right_vol:
            selected_bins.add(left_idx)
            current_vol += left_vol
            left_idx -= 1
        else:
            selected_bins.add(right_idx)
            current_vol += right_vol
            right_idx += 1
            
    min_bin = min(selected_bins)
    max_bin = max(selected_bins)
    
    val = bin_edges[min_bin]
    vah = bin_edges[max_bin + 1]
    
    return {
        "vah": float(vah),
        "val": float(val),
        "poc": float(poc)
    }

def get_sr_levels(swing_highs, swing_lows, current_price, value_area=None, tolerance=0.003) -> dict:
    """
    Cluster all swing points within tolerance (0.3%) to find key S/R zones.
    Returns nearest 3 Support and 3 Resistance zones with their touch density counts.
    Adds a high_confidence flag if the zone aligns with a Value Area node.
    """
    prices = sorted([s[1] for s in swing_highs] + [s[1] for s in swing_lows])
    if not prices:
        return {"support": [], "resistance": []}
        
    clusters = []
    current_cluster = []
    
    for p in prices:
        if not current_cluster:
            current_cluster.append(p)
        else:
            base = current_cluster[0]
            if (p - base) / base <= tolerance:
                current_cluster.append(p)
            else:
                clusters.append(current_cluster)
                current_cluster = [p]
    if current_cluster:
        clusters.append(current_cluster)
        
    sr_zones = []
    for c in clusters:
        avg_price = sum(c) / len(c)
        touch_count = len(c)
        
        is_high_conf = False
        if value_area:
            for level in [value_area['vah'], value_area['val'], value_area['poc']]:
                if abs(avg_price - level) / level <= 0.005:
                    is_high_conf = True
                    break
                    
        sr_zones.append({
            "price": float(avg_price),
            "touches": int(touch_count),
            "high_confidence": is_high_conf
        })
        
    supports = sorted([z for z in sr_zones if z["price"] < current_price], key=lambda x: x["price"], reverse=True)[:3]
    resistances = sorted([z for z in sr_zones if z["price"] > current_price], key=lambda x: x["price"])[:3]
    
    return {
        "support": supports,
        "resistance": resistances
    }

def calculate_previous_day(df: pd.DataFrame):
    """
    Extract PDH, PDL, PDC from DataFrame by daily grouping.
    """
    try:
        dates = df['timestamp'].dt.date
        daily = df.groupby(dates).agg({'high': 'max', 'low': 'min', 'close': 'last'}).sort_index()
        if len(daily) >= 2:
            pdh = float(daily['high'].iloc[-2])
            pdl = float(daily['low'].iloc[-2])
            pdc = float(daily['close'].iloc[-2])
        else:
            pdh = float(df['high'].max())
            pdl = float(df['low'].min())
            pdc = float(df['close'].iloc[-1])
        return pdh, pdl, pdc
    except Exception:
        return float(df['high'].max()), float(df['low'].min()), float(df['close'].iloc[-1])

def build_pa_context(df_1h, df_15m) -> dict:
    """
    Build Price Action context for snapshot.
    """
    if df_15m is None or df_15m.empty:
        return {}
        
    current_price = float(df_15m['close'].iloc[-1])
    last_pattern = detect_candle_pattern(df_15m)
    poc_1h = get_poc(df_1h) if df_1h is not None and not df_1h.empty else current_price
    
    # Calculate Value Area High / Low on 1H (represents intermediate session value)
    va_1h = get_value_area(df_1h) if df_1h is not None and not df_1h.empty else {"vah": current_price, "val": current_price, "poc": current_price}
    
    # Calculate support/resistance using 15m swings with 15m low-lag settings (left=3, right=2)
    sh, sl = find_swings(df_15m, left=3, right=2)
    sr_levels = get_sr_levels(sh, sl, current_price, value_area=va_1h)
    
    # Calculate Previous Day metrics from 1h
    pdh, pdl, pdc = calculate_previous_day(df_1h) if df_1h is not None and not df_1h.empty else (current_price, current_price, current_price)
    
    return {
        "last_candle_pattern_15m": last_pattern,
        "poc_1h": poc_1h,
        "value_area_1h": va_1h,
        "sr_levels": sr_levels,
        "pdh": pdh,
        "pdl": pdl,
        "pdc": pdc
    }
