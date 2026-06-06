"""
Cryptera v3.2 - Smart Money Concepts (SMC) Engine

Single fractal swing definition shared with the indicator layer (no more dual
swing systems). Developing swings are a separate early-warning channel and are
never allowed to redefine structure. BOS / CHoCH are detected as the first
close beyond the reference swing (event semantics), with optional displacement.
Adaptive thresholds scale with ATR-as-a-fraction-of-price.
"""

import pandas as pd
import numpy as np


# Per-timeframe fractal half-widths (left, right) used everywhere swings are needed.
SWING_WIDTHS = {"4h": (5, 5), "1h": (4, 3), "15m": (3, 2)}
DEFAULT_WIDTH = (3, 3)


def _atr_pct(df: pd.DataFrame, fallback: float = 0.005) -> float:
    """ATR as a fraction of current close. Used to scale tolerances per asset/TF."""
    if df is None or df.empty or "ATR" not in df.columns:
        return fallback
    atr = df["ATR"].iloc[-1]
    price = float(df["close"].iloc[-1])
    if pd.isna(atr) or price <= 0:
        return fallback
    return max(float(atr) / price, 0.0005)


# ---------------- unified fractal swing detection ---------------- #

def fractal_swing_mask(high: pd.Series, low: pd.Series, left: int, right: int):
    """
    Vectorised fractal pivots. A bar is a swing high if its high is the maximum
    over [i-left, i+right]; swing low symmetrically. The last `right` bars are
    NaN/False (no look-ahead). Returns (is_high: bool Series, is_low: bool Series).
    """
    window = left + right + 1
    # rolling(window).max() at q covers [q-window+1, q]; shift(-right) re-centres
    roll_high = high.rolling(window).max().shift(-right)
    roll_low = low.rolling(window).min().shift(-right)
    is_high = (high == roll_high) & roll_high.notna()
    is_low = (low == roll_low) & roll_low.notna()
    return is_high, is_low


def _confirmed_swings(df: pd.DataFrame, left: int, right: int):
    """Return confirmed swings as ordered lists of dicts {idx, ts, price}."""
    if df is None or len(df) < (left + right + 1):
        return [], []
    is_high, is_low = fractal_swing_mask(df["high"], df["low"], left, right)

    def _collect(mask, col):
        out = []
        for idx in np.flatnonzero(mask.to_numpy()):
            ts = df["timestamp"].iloc[idx]
            ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            out.append({"idx": int(idx), "ts": ts_str, "price": float(df[col].iloc[idx])})
        return out

    return _collect(is_high, "high"), _collect(is_low, "low")


def find_developing_swings(df: pd.DataFrame):
    """
    Early-warning swings on the unconfirmed tail (volume + wick rejection).
    Returned as a SEPARATE channel; never fed into structure / BOS / CHoCH.
    """
    if df is None or df.empty:
        return [], []
    n = len(df)
    right = 3
    atr = df["ATR"].iloc[-1] if "ATR" in df.columns and not pd.isna(df["ATR"].iloc[-1]) else (df["high"] - df["low"]).mean()
    vol_mean = df["volume"].rolling(20).mean().iloc[-1] if n >= 20 else df["volume"].mean()
    dev_highs, dev_lows = [], []
    for idx in range(max(0, n - right), n):
        c = df.iloc[idx]
        high, low, open_p, close = float(c["high"]), float(c["low"]), float(c["open"]), float(c["close"])
        vol = float(c["volume"])
        body = abs(close - open_p)
        upper_wick = high - max(open_p, close)
        lower_wick = min(open_p, close) - low
        ts = c["timestamp"]
        ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
        if upper_wick > body * 1.2 and upper_wick > atr * 0.4 and vol > vol_mean * 1.1:
            dev_highs.append({"idx": idx, "ts": ts_str, "price": high})
        if lower_wick > body * 1.2 and lower_wick > atr * 0.4 and vol > vol_mean * 1.1:
            dev_lows.append({"idx": idx, "ts": ts_str, "price": low})
    return dev_highs, dev_lows


def find_swings(df: pd.DataFrame, left=5, right=5):
    """
    Back-compatible accessor: returns the last 5 CONFIRMED swings as
    (ts, price) tuples for highs and lows. Structure-grade only.
    """
    highs, lows = _confirmed_swings(df, left, right)
    sh = [(h["ts"], h["price"]) for h in highs][-5:]
    sl = [(l["ts"], l["price"]) for l in lows][-5:]
    return sh, sl


# ---------------- structure / BOS / CHoCH ---------------- #

def _has_displacement_fvg(df: pd.DataFrame, start_idx: int, end_idx: int, direction: str) -> bool:
    """True if a same-direction 3-candle FVG exists within (start_idx, end_idx]."""
    lo = max(2, start_idx)
    hi = min(len(df) - 1, end_idx)
    for i in range(lo, hi + 1):
        if direction == "BULLISH":
            if float(df["low"].iloc[i]) > float(df["high"].iloc[i - 2]):
                return True
        else:
            if float(df["high"].iloc[i]) < float(df["low"].iloc[i - 2]):
                return True
    return False


def get_market_structure(confirmed_highs, confirmed_lows, df=None, recent_bars=12):
    """
    Determine structure label plus event-based BOS and CHoCH.

    confirmed_highs / confirmed_lows: ordered lists of {idx, ts, price}.
    BOS  = first close beyond the most recent confirmed swing in the trend
           direction (continuation), flagged 'fresh' if within recent_bars.
    CHoCH = first counter-trend close beyond the last protected swing.
    """
    neutral = (
        "NEUTRAL",
        {"level": None, "direction": None, "fresh": False, "displacement": False},
        {"detected": False, "direction": None, "level": None, "fresh": False},
    )
    if len(confirmed_highs) < 2 or len(confirmed_lows) < 2:
        return neutral

    sh1, sh2 = confirmed_highs[-2]["price"], confirmed_highs[-1]["price"]
    sl1, sl2 = confirmed_lows[-2]["price"], confirmed_lows[-1]["price"]

    high_state = "HH" if sh2 > sh1 else "LH"
    low_state = "HL" if sl2 > sl1 else "LL"
    structure = f"{high_state}/{low_state}"

    bos = {"level": None, "direction": None, "fresh": False, "displacement": False}
    choch = {"detected": False, "direction": None, "level": None, "fresh": False}

    if df is None or df.empty:
        return structure, bos, choch

    closes = df["close"].to_numpy()
    n = len(df)

    last_high = confirmed_highs[-1]
    last_low = confirmed_lows[-1]

    # --- BOS: first close beyond the most recent confirmed swing (continuation) ---
    # Bullish BOS: first close above last confirmed swing high after it formed.
    after_h = closes[last_high["idx"] + 1:]
    break_pos_up = None
    for k, c in enumerate(after_h):
        if c > last_high["price"]:
            break_pos_up = last_high["idx"] + 1 + k
            break

    after_l = closes[last_low["idx"] + 1:]
    break_pos_dn = None
    for k, c in enumerate(after_l):
        if c < last_low["price"]:
            break_pos_dn = last_low["idx"] + 1 + k
            break

    # Choose the more recent break as the active BOS
    if break_pos_up is not None and (break_pos_dn is None or break_pos_up >= break_pos_dn):
        bos = {
            "level": last_high["price"],
            "direction": "BULLISH",
            "fresh": (n - 1 - break_pos_up) <= recent_bars,
            "displacement": _has_displacement_fvg(df, last_high["idx"], break_pos_up, "BULLISH"),
        }
    elif break_pos_dn is not None:
        bos = {
            "level": last_low["price"],
            "direction": "BEARISH",
            "fresh": (n - 1 - break_pos_dn) <= recent_bars,
            "displacement": _has_displacement_fvg(df, last_low["idx"], break_pos_dn, "BEARISH"),
        }

    # --- CHoCH: counter-trend break of the last protected swing ---
    # In an up-leg (HH/HL) a close below the last higher-low = bearish CHoCH.
    if structure == "HH/HL":
        after = closes[last_low["idx"] + 1:]
        for k, c in enumerate(after):
            if c < last_low["price"]:
                pos = last_low["idx"] + 1 + k
                choch = {"detected": True, "direction": "BEARISH", "level": last_low["price"],
                         "fresh": (n - 1 - pos) <= recent_bars}
                break
    elif structure == "LH/LL":
        after = closes[last_high["idx"] + 1:]
        for k, c in enumerate(after):
            if c > last_high["price"]:
                pos = last_high["idx"] + 1 + k
                choch = {"detected": True, "direction": "BULLISH", "level": last_high["price"],
                         "fresh": (n - 1 - pos) <= recent_bars}
                break

    return structure, bos, choch


# ---------------- order blocks ---------------- #

def find_order_blocks(df: pd.DataFrame, value_area: dict, lookback=50):
    """
    Unmitigated Order Blocks. Impulse threshold scales with ATR. HVN tag uses the
    pre-computed value_area for this TF (passed in — no recompute, fixes B11).
    """
    vah, val, poc = value_area["vah"], value_area["val"], value_area["poc"]
    atr_pct = _atr_pct(df)
    hvn_tol = max(0.003, min(0.01, atr_pct * 0.6))

    bullish_obs, bearish_obs = [], []
    n = len(df)
    start_idx = max(3, n - lookback)

    def _tag_hvn(mid):
        for level in (vah, val, poc):
            if level > 0 and abs(mid - level) / level <= hvn_tol:
                return True
        return False

    for i in range(start_idx, n):
        # bullish impulse
        bull, b_start = False, -1
        for k in (1, 2, 3):
            prev_c, curr_c = df["close"].iloc[i - k], df["close"].iloc[i]
            atr_i = df["ATR"].iloc[i - k] if "ATR" in df.columns else np.nan
            thr = (prev_c * max(atr_pct, 0.005) * 1.5) if (pd.isna(atr_i) or atr_i <= 0) else 2.0 * atr_i
            if (curr_c - prev_c) >= thr:
                bull, b_start = True, i - k
                break
        if bull:
            for j in range(b_start - 1, max(0, b_start - 10), -1):
                if df["close"].iloc[j] < df["open"].iloc[j]:
                    hi, lo = float(df["high"].iloc[j]), float(df["low"].iloc[j])
                    ts = df["timestamp"].iloc[j]
                    ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
                    if not any(o["timestamp"] == ts_str for o in bullish_obs):
                        bullish_obs.append({"timestamp": ts_str, "top": hi, "bottom": lo,
                                            "created_at_idx": j, "high_volume_node": _tag_hvn((hi + lo) / 2)})
                    break

        # bearish impulse
        bear, s_start = False, -1
        for k in (1, 2, 3):
            prev_c, curr_c = df["close"].iloc[i - k], df["close"].iloc[i]
            atr_i = df["ATR"].iloc[i - k] if "ATR" in df.columns else np.nan
            thr = (prev_c * max(atr_pct, 0.005) * 1.5) if (pd.isna(atr_i) or atr_i <= 0) else 2.0 * atr_i
            if (prev_c - curr_c) >= thr:
                bear, s_start = True, i - k
                break
        if bear:
            for j in range(s_start - 1, max(0, s_start - 10), -1):
                if df["close"].iloc[j] > df["open"].iloc[j]:
                    hi, lo = float(df["high"].iloc[j]), float(df["low"].iloc[j])
                    ts = df["timestamp"].iloc[j]
                    ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
                    if not any(o["timestamp"] == ts_str for o in bearish_obs):
                        bearish_obs.append({"timestamp": ts_str, "top": hi, "bottom": lo,
                                            "created_at_idx": j, "high_volume_node": _tag_hvn((hi + lo) / 2)})
                    break

    unmit_bull = []
    for ob in bullish_obs:
        sub = df["close"].iloc[ob["created_at_idx"] + 1:]
        if sub.empty or sub.min() >= ob["bottom"]:
            unmit_bull.append({k: v for k, v in ob.items() if k != "created_at_idx"})

    unmit_bear = []
    for ob in bearish_obs:
        sub = df["close"].iloc[ob["created_at_idx"] + 1:]
        if sub.empty or sub.max() <= ob["top"]:
            unmit_bear.append({k: v for k, v in ob.items() if k != "created_at_idx"})

    return {"bullish": unmit_bull[-3:], "bearish": unmit_bear[-3:]}


# ---------------- fair value gaps ---------------- #

def find_fvg(df: pd.DataFrame, lookback=50):
    """3-candle FVGs; nearest unfilled above/below current price."""
    bullish, bearish = [], []
    n = len(df)
    start_idx = max(2, n - lookback)
    current_price = float(df["close"].iloc[-1])

    for i in range(start_idx, n):
        high_2, low_0 = float(df["high"].iloc[i - 2]), float(df["low"].iloc[i])
        if low_0 > high_2:
            sub = df.iloc[i + 1:]
            filled = (not sub.empty) and (sub["low"].min() <= high_2)
            ts = df["timestamp"].iloc[i]
            bullish.append({"top": low_0, "bottom": high_2,
                            "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                            "filled": bool(filled)})
        low_2, high_0 = float(df["low"].iloc[i - 2]), float(df["high"].iloc[i])
        if low_2 > high_0:
            sub = df.iloc[i + 1:]
            filled = (not sub.empty) and (sub["high"].max() >= low_2)
            ts = df["timestamp"].iloc[i]
            bearish.append({"top": low_2, "bottom": high_0,
                            "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                            "filled": bool(filled)})

    nearest_bull = None
    below = [f for f in bullish if not f["filled"] and f["top"] <= current_price]
    if below:
        nearest_bull = max(below, key=lambda x: x["top"])
    nearest_bear = None
    above = [f for f in bearish if not f["filled"] and f["bottom"] >= current_price]
    if above:
        nearest_bear = min(above, key=lambda x: x["bottom"])

    return {
        "nearest_bullish_fvg": nearest_bull,
        "nearest_bearish_fvg": nearest_bear,
        "bullish_fvgs": bullish[-5:],
        "bearish_fvgs": bearish[-5:],
    }


# ---------------- liquidity pools ---------------- #

def find_liquidity_levels(swing_highs, swing_lows, atr_pct: float = 0.002):
    """Cluster swings within an ATR-scaled tolerance into BSL / SSL pools (top 3 each)."""
    tolerance = max(0.001, min(0.01, atr_pct * 0.5))
    high_prices = sorted([s[1] for s in swing_highs])
    low_prices = sorted([s[1] for s in swing_lows])

    def _cluster(prices, take_max):
        levels, used = [], set()
        for i, p1 in enumerate(prices):
            if i in used:
                continue
            cl = [p1]
            for j, p2 in enumerate(prices[i + 1:]):
                idx = i + 1 + j
                if idx in used:
                    continue
                if p1 > 0 and (p2 - p1) / p1 <= tolerance:
                    cl.append(p2)
                    used.add(idx)
            levels.append(max(cl) if take_max else min(cl))
        return levels

    bsl = _cluster(high_prices, take_max=True)
    ssl = _cluster(low_prices, take_max=False)
    return {"buy_side": sorted(bsl, reverse=True)[:3], "sell_side": sorted(ssl)[:3]}


def get_premium_discount(swing_high, swing_low, current_price):
    """Premium/discount + OTE (61.8%-78.6%) of a single dealing range."""
    if swing_high <= swing_low:
        return {"zone": "neutral", "ote_zone": {"low": 0, "high": 0}, "in_ote": False}
    rng = swing_high - swing_low
    midpoint = swing_low + 0.5 * rng
    zone = "premium" if current_price > midpoint else "discount"
    ote_low = swing_high - 0.786 * rng
    ote_high = swing_high - 0.618 * rng
    return {
        "zone": zone,
        "ote_zone": {"low": round(ote_low, 4), "high": round(ote_high, 4)},
        "in_ote": ote_low <= current_price <= ote_high,
    }


def _current_dealing_range(confirmed_highs, confirmed_lows):
    """
    Most recent dealing range = the latest confirmed swing high and the latest
    confirmed swing low that bracket current price (single leg), not max/min of a
    mixed window (fixes B9).
    """
    if not confirmed_highs or not confirmed_lows:
        return None, None
    last_high = confirmed_highs[-1]
    last_low = confirmed_lows[-1]
    # Pair the most recent high with the most recent low; order them as a range.
    hi = last_high["price"]
    lo = last_low["price"]
    if hi <= lo:
        # Degenerate ordering — widen using the prior swing on the deficient side.
        if len(confirmed_highs) >= 2:
            hi = max(hi, confirmed_highs[-2]["price"])
        if len(confirmed_lows) >= 2:
            lo = min(lo, confirmed_lows[-2]["price"])
    return hi, lo


# ---------------- context assembly ---------------- #

def build_smc_context(df_4h, df_1h, df_15m, value_areas: dict = None):
    """
    Build multi-timeframe SMC context. `value_areas` is an optional
    {tf: {vah,val,poc}} dict computed once upstream to avoid recomputation (B11).
    """
    from price_action import get_value_area

    context = {}
    value_areas = value_areas or {}
    for tf, df in (("4h", df_4h), ("1h", df_1h), ("15m", df_15m)):
        if df is None or df.empty:
            continue

        current_price = float(df["close"].iloc[-1])
        left_p, right_p = SWING_WIDTHS.get(tf, DEFAULT_WIDTH)

        confirmed_highs, confirmed_lows = _confirmed_swings(df, left_p, right_p)
        sh = [(h["ts"], h["price"]) for h in confirmed_highs][-5:]
        sl = [(l["ts"], l["price"]) for l in confirmed_lows][-5:]

        struct, bos, choch = get_market_structure(confirmed_highs, confirmed_lows, df)

        va = value_areas.get(tf) or get_value_area(df)
        obs = find_order_blocks(df, va)
        fvgs = find_fvg(df)
        liq = find_liquidity_levels(sh, sl, atr_pct=_atr_pct(df))

        dealing_high, dealing_low = _current_dealing_range(confirmed_highs, confirmed_lows)
        if dealing_high is not None and dealing_low is not None:
            pd_zone = get_premium_discount(dealing_high, dealing_low, current_price)
        else:
            pd_zone = {"zone": "neutral", "in_ote": False, "ote_zone": {"low": 0, "high": 0}}

        dev_h, dev_l = find_developing_swings(df)

        context[tf] = {
            "current_price": current_price,
            "structure": struct,
            "bos": bos,
            "choch": choch,
            "order_blocks": obs,
            "fvg": fvgs,
            "liquidity_levels": liq,
            "premium_discount": pd_zone,
            "dealing_range": {"high": dealing_high, "low": dealing_low},
            "developing_swings": {
                "highs": [d["price"] for d in dev_h],
                "lows": [d["price"] for d in dev_l],
            },
        }

    return context
