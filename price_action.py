"""
Cryptera v3.1 - Price Action (PA) Engine

ATR-normalised S/R clustering, multi-TF volume profile, plus untested HTF POC
tracking (useful as magnet targets).
"""

import pandas as pd
import numpy as np
from smc import find_swings


def _atr_pct(df: pd.DataFrame, fallback: float = 0.005) -> float:
    if df is None or df.empty or "ATR" not in df.columns:
        return fallback
    atr = df["ATR"].iloc[-1]
    price = float(df["close"].iloc[-1])
    if pd.isna(atr) or price <= 0:
        return fallback
    return max(float(atr) / price, 0.0005)


def detect_candle_pattern(df: pd.DataFrame) -> str:
    """Manual classifier over the last 3 candles. Returns pattern name or 'none'."""
    if len(df) < 3:
        return "none"

    c2 = df.iloc[-3]
    c1 = df.iloc[-2]
    c0 = df.iloc[-1]

    def props(c):
        body = abs(c["close"] - c["open"])
        rng = c["high"] - c["low"] if c["high"] - c["low"] > 0 else 0.001
        is_green = c["close"] > c["open"]
        is_red = c["close"] < c["open"]
        lower_wick = min(c["open"], c["close"]) - c["low"]
        upper_wick = c["high"] - max(c["open"], c["close"])
        return body, rng, is_green, is_red, lower_wick, upper_wick

    b0, r0, g0, r_0, lw0, uw0 = props(c0)
    b1, r1, g1, r_1, lw1, uw1 = props(c1)
    b2, r2, g2, r_2, lw2, uw2 = props(c2)

    if r0 > 0 and b0 / r0 <= 0.1:
        return "doji"
    if c0["high"] < c1["high"] and c0["low"] > c1["low"]:
        return "inside_bar"
    if r0 > 0 and (lw0 / r0 >= 0.6) and (b0 / r0 <= 0.3):
        return "pin_bar_bull"
    if r0 > 0 and (uw0 / r0 >= 0.6) and (b0 / r0 <= 0.3):
        return "pin_bar_bear"
    if r_1 and g0 and (c0["close"] >= c1["open"]) and (c0["open"] <= c1["close"]):
        return "bullish_engulfing"
    if g1 and r_0 and (c0["close"] <= c1["open"]) and (c0["open"] >= c1["close"]):
        return "bearish_engulfing"
    if r_2 and r2 > 0 and (b1 / r2 <= 0.3) and g0 and (c0["close"] >= (c2["open"] + c2["close"]) / 2):
        return "morning_star"
    if g2 and r2 > 0 and (b1 / r2 <= 0.3) and r_0 and (c0["close"] <= (c2["open"] + c2["close"]) / 2):
        return "evening_star"
    return "none"


def get_poc(df: pd.DataFrame, lookback=50, buckets=100) -> float:
    recent = df.tail(lookback)
    closes = recent["close"].values
    vols = recent["volume"].values
    min_price = closes.min()
    max_price = closes.max()
    if max_price == min_price:
        return float(min_price)
    hist, bin_edges = np.histogram(closes, bins=buckets, weights=vols)
    max_idx = np.argmax(hist)
    return float((bin_edges[max_idx] + bin_edges[max_idx + 1]) / 2)


def get_value_area(df: pd.DataFrame, lookback=50, buckets=100) -> dict:
    recent = df.tail(lookback)
    closes = recent["close"].values
    vols = recent["volume"].values
    min_price = closes.min()
    max_price = closes.max()
    if max_price == min_price:
        return {"vah": float(max_price), "val": float(min_price), "poc": float(min_price)}

    hist, bin_edges = np.histogram(closes, bins=buckets, weights=vols)
    max_idx = np.argmax(hist)
    poc = (bin_edges[max_idx] + bin_edges[max_idx + 1]) / 2

    total_vol = hist.sum()
    if total_vol <= 0:
        return {"vah": float(poc), "val": float(poc), "poc": float(poc)}

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

    val = bin_edges[min(selected_bins)]
    vah = bin_edges[max(selected_bins) + 1]
    return {"vah": float(vah), "val": float(val), "poc": float(poc)}


def get_sr_levels(swing_highs, swing_lows, current_price, value_area=None,
                  atr_pct: float = 0.003) -> dict:
    """
    Cluster swings within an ATR-scaled tolerance into S/R zones. Each zone
    carries a touch count and a high-confidence flag if it aligns with VAH/POC/VAL.
    """
    tolerance = max(0.0015, min(0.01, atr_pct * 0.6))
    hvn_tol = max(0.002, min(0.01, atr_pct * 0.7))

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
            if base > 0 and (p - base) / base <= tolerance:
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
            for level in (value_area["vah"], value_area["val"], value_area["poc"]):
                if level > 0 and abs(avg_price - level) / level <= hvn_tol:
                    is_high_conf = True
                    break
        sr_zones.append({
            "price": float(avg_price),
            "touches": int(touch_count),
            "high_confidence": is_high_conf,
        })

    supports = sorted([z for z in sr_zones if z["price"] < current_price],
                      key=lambda x: x["price"], reverse=True)[:3]
    resistances = sorted([z for z in sr_zones if z["price"] > current_price],
                         key=lambda x: x["price"])[:3]
    return {"support": supports, "resistance": resistances}


def calculate_previous_day(df: pd.DataFrame):
    try:
        dates = df["timestamp"].dt.date
        daily = df.groupby(dates).agg({"high": "max", "low": "min", "close": "last"}).sort_index()
        if len(daily) >= 2:
            return float(daily["high"].iloc[-2]), float(daily["low"].iloc[-2]), float(daily["close"].iloc[-2])
        return float(df["high"].max()), float(df["low"].min()), float(df["close"].iloc[-1])
    except Exception:
        return float(df["high"].max()), float(df["low"].min()), float(df["close"].iloc[-1])


def find_untested_pocs(df: pd.DataFrame, sessions: int = 5, lookback_per_session: int = 50) -> list:
    """
    Build a list of session POCs that have not been re-tested since formation.
    Strong magnet targets for trades anchored to volume.
    """
    if df is None or df.empty:
        return []
    out = []
    n = len(df)
    if n < lookback_per_session:
        return []
    current_price = float(df["close"].iloc[-1])
    step = max(1, lookback_per_session // 2)
    for s in range(sessions):
        end = n - s * step
        start = end - lookback_per_session
        if start < 0 or end <= start:
            break
        slice_df = df.iloc[start:end]
        if slice_df.empty:
            continue
        poc = get_poc(slice_df)
        # untested if price has not returned to within 0.5% of POC since end-of-session
        post = df.iloc[end:]
        if post.empty:
            tested = False
        else:
            band = max(poc * 0.005, 1e-6)
            tested = ((post["high"] >= poc - band) & (post["low"] <= poc + band)).any()
        if not tested:
            out.append({
                "poc": float(poc),
                "session_end_ts": slice_df["timestamp"].iloc[-1].isoformat() if hasattr(slice_df["timestamp"].iloc[-1], "isoformat") else str(slice_df["timestamp"].iloc[-1]),
                "distance_pct": round(abs(current_price - poc) / current_price * 100, 3) if current_price > 0 else None,
            })
    return out[:5]


def build_pa_context(df_1h, df_15m, df_4h=None) -> dict:
    if df_15m is None or df_15m.empty:
        return {}

    current_price = float(df_15m["close"].iloc[-1])
    last_pattern = detect_candle_pattern(df_15m)

    va_15m = get_value_area(df_15m) if df_15m is not None and not df_15m.empty else {"vah": current_price, "val": current_price, "poc": current_price}
    va_1h = get_value_area(df_1h) if df_1h is not None and not df_1h.empty else {"vah": current_price, "val": current_price, "poc": current_price}
    va_4h = get_value_area(df_4h) if df_4h is not None and not df_4h.empty else {"vah": current_price, "val": current_price, "poc": current_price}

    sh, sl = find_swings(df_15m, left=3, right=2)
    sr_levels = get_sr_levels(sh, sl, current_price, value_area=va_1h, atr_pct=_atr_pct(df_15m))

    pdh, pdl, pdc = calculate_previous_day(df_1h) if df_1h is not None and not df_1h.empty else (current_price, current_price, current_price)

    untested_pocs_4h = find_untested_pocs(df_4h, sessions=5, lookback_per_session=50) if df_4h is not None and not df_4h.empty else []

    return {
        "last_candle_pattern_15m": last_pattern,
        "value_area_15m": va_15m,
        "value_area_1h": va_1h,
        "value_area_4h": va_4h,
        "poc_1h": va_1h.get("poc", current_price),
        "sr_levels": sr_levels,
        "pdh": pdh,
        "pdl": pdl,
        "pdc": pdc,
        "untested_pocs_4h": untested_pocs_4h,
    }
