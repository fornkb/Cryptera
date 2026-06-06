"""
Cryptera v3.2 - Price Action (PA) Engine

Volume profile is range-distributed (each candle's volume spread across its
high-low range), not close-only. Session POCs use true non-overlapping windows.
Candle-pattern detection is limited to confirmation-grade reversal patterns.
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
    """
    Confirmation-grade pattern over the last 1-3 candles. Limited to patterns
    with genuine reversal/continuation signal at a level: pin bars, engulfing,
    morning/evening star. Pure-noise single-bar patterns (doji, inside bar) are
    intentionally excluded.
    """
    if len(df) < 3:
        return "none"
    c2, c1, c0 = df.iloc[-3], df.iloc[-2], df.iloc[-1]

    def props(c):
        body = abs(c["close"] - c["open"])
        rng = c["high"] - c["low"] if c["high"] - c["low"] > 0 else 0.001
        return body, rng, c["close"] > c["open"], c["close"] < c["open"], \
            min(c["open"], c["close"]) - c["low"], c["high"] - max(c["open"], c["close"])

    b0, r0, g0, r_0, lw0, uw0 = props(c0)
    b1, r1, g1, r_1, lw1, uw1 = props(c1)
    b2, r2, g2, r_2, lw2, uw2 = props(c2)

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


# ---------------- range-distributed volume profile ---------------- #

def _build_profile(df: pd.DataFrame, lookback: int, buckets: int):
    """
    Distribute each candle's volume uniformly across its [low, high] range into a
    price histogram. Returns (hist, bin_edges) or (None, None) on degenerate input.
    """
    recent = df.tail(lookback)
    lows = recent["low"].to_numpy(dtype=float)
    highs = recent["high"].to_numpy(dtype=float)
    vols = recent["volume"].to_numpy(dtype=float)

    p_min = float(np.min(lows))
    p_max = float(np.max(highs))
    if not np.isfinite(p_min) or not np.isfinite(p_max) or p_max <= p_min:
        return None, None

    edges = np.linspace(p_min, p_max, buckets + 1)
    hist = np.zeros(buckets, dtype=float)
    bin_w = (p_max - p_min) / buckets

    for lo, hi, vol in zip(lows, highs, vols):
        if vol <= 0:
            continue
        if hi <= lo:
            # zero-range bar: dump into its single bin
            b = min(int((lo - p_min) / bin_w), buckets - 1)
            hist[b] += vol
            continue
        lo_b = int((lo - p_min) / bin_w)
        hi_b = min(int((hi - p_min) / bin_w), buckets - 1)
        lo_b = max(lo_b, 0)
        span = hi_b - lo_b + 1
        # spread this candle's volume evenly across the bins it covers
        hist[lo_b:hi_b + 1] += vol / span

    return hist, edges


def get_poc(df: pd.DataFrame, lookback=50, buckets=100) -> float:
    hist, edges = _build_profile(df, lookback, buckets)
    if hist is None:
        return float(df["close"].iloc[-1])
    idx = int(np.argmax(hist))
    return float((edges[idx] + edges[idx + 1]) / 2)


def get_value_area(df: pd.DataFrame, lookback=50, buckets=100) -> dict:
    hist, edges = _build_profile(df, lookback, buckets)
    if hist is None:
        price = float(df["close"].iloc[-1])
        return {"vah": price, "val": price, "poc": price}

    max_idx = int(np.argmax(hist))
    poc = (edges[max_idx] + edges[max_idx + 1]) / 2
    total = hist.sum()
    if total <= 0:
        return {"vah": float(poc), "val": float(poc), "poc": float(poc)}

    target = total * 0.70
    selected = {max_idx}
    cur = hist[max_idx]
    left, right = max_idx - 1, max_idx + 1
    while cur < target:
        lv = hist[left] if left >= 0 else 0
        rv = hist[right] if right < buckets else 0
        if lv == 0 and rv == 0:
            break
        if lv >= rv:
            selected.add(left)
            cur += lv
            left -= 1
        else:
            selected.add(right)
            cur += rv
            right += 1

    val = edges[min(selected)]
    vah = edges[max(selected) + 1]
    return {"vah": float(vah), "val": float(val), "poc": float(poc)}


def get_sr_levels(swing_highs, swing_lows, current_price, value_area=None, atr_pct: float = 0.003) -> dict:
    """Cluster swings into S/R zones with touch counts and VA-aligned confidence flags."""
    tolerance = max(0.0015, min(0.01, atr_pct * 0.6))
    hvn_tol = max(0.002, min(0.01, atr_pct * 0.7))

    prices = sorted([s[1] for s in swing_highs] + [s[1] for s in swing_lows])
    if not prices:
        return {"support": [], "resistance": []}

    clusters, current = [], []
    for p in prices:
        if not current:
            current.append(p)
        else:
            base = current[0]
            if base > 0 and (p - base) / base <= tolerance:
                current.append(p)
            else:
                clusters.append(current)
                current = [p]
    if current:
        clusters.append(current)

    zones = []
    for c in clusters:
        avg = sum(c) / len(c)
        high_conf = False
        if value_area:
            for level in (value_area["vah"], value_area["val"], value_area["poc"]):
                if level > 0 and abs(avg - level) / level <= hvn_tol:
                    high_conf = True
                    break
        zones.append({"price": float(avg), "touches": len(c), "high_confidence": high_conf})

    supports = sorted([z for z in zones if z["price"] < current_price], key=lambda x: x["price"], reverse=True)[:3]
    resistances = sorted([z for z in zones if z["price"] > current_price], key=lambda x: x["price"])[:3]
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


def find_untested_pocs(df: pd.DataFrame, n_sessions: int = 4, session_bars: int = 50) -> list:
    """
    True non-overlapping session POCs that have not been re-tested since they
    formed. The most-recent (still-developing) window is skipped because it has
    no post-formation data to test against (fixes B2). Returns up to 5 magnets
    sorted by proximity to current price.
    """
    if df is None or df.empty:
        return []
    n = len(df)
    current_price = float(df["close"].iloc[-1])
    out = []
    # session s spans [n - (s+1)*bars, n - s*bars); s=0 is the developing window -> skip
    for s in range(1, n_sessions + 1):
        end = n - s * session_bars
        start = end - session_bars
        if start < 0 or end <= start:
            break
        window = df.iloc[start:end]
        if window.empty:
            continue
        poc = get_poc(window, lookback=session_bars)
        post = df.iloc[end:]
        band = max(poc * 0.005, 1e-6)
        tested = (not post.empty) and ((post["high"] >= poc - band) & (post["low"] <= poc + band)).any()
        if not tested:
            ts = window["timestamp"].iloc[-1]
            out.append({
                "poc": float(poc),
                "session_end_ts": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                "distance_pct": round(abs(current_price - poc) / current_price * 100, 3) if current_price > 0 else None,
            })
    # de-dup near-identical POCs and sort by proximity
    deduped = []
    for p in sorted(out, key=lambda x: (x["distance_pct"] if x["distance_pct"] is not None else 1e9)):
        if all(abs(p["poc"] - q["poc"]) / max(q["poc"], 1e-9) > 0.002 for q in deduped):
            deduped.append(p)
    return deduped[:5]


def build_pa_context(df_1h, df_15m, df_4h=None, value_areas: dict = None) -> dict:
    """value_areas: optional pre-computed {tf:{vah,val,poc}} to avoid recompute (B11)."""
    if df_15m is None or df_15m.empty:
        return {}

    value_areas = value_areas or {}
    current_price = float(df_15m["close"].iloc[-1])
    last_pattern = detect_candle_pattern(df_15m)

    va_15m = value_areas.get("15m") or get_value_area(df_15m)
    va_1h = value_areas.get("1h") or (get_value_area(df_1h) if df_1h is not None and not df_1h.empty
                                      else {"vah": current_price, "val": current_price, "poc": current_price})
    va_4h = value_areas.get("4h") or (get_value_area(df_4h) if df_4h is not None and not df_4h.empty
                                      else {"vah": current_price, "val": current_price, "poc": current_price})

    sh, sl = find_swings(df_15m, left=3, right=2)
    sr_levels = get_sr_levels(sh, sl, current_price, value_area=va_1h, atr_pct=_atr_pct(df_15m))

    pdh, pdl, pdc = calculate_previous_day(df_1h) if df_1h is not None and not df_1h.empty \
        else (current_price, current_price, current_price)

    untested_pocs_4h = find_untested_pocs(df_4h) if df_4h is not None and not df_4h.empty else []

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
