"""
Cryptera v3.1 - Smart Money Concepts (SMC) Engine

Adaptive thresholds: all proximity / cluster / impulse tests are now scaled
by ATR-as-a-fraction-of-price so the engine behaves consistently across
high- and low-volatility assets.
"""

import pandas as pd
import numpy as np


def _atr_pct(df: pd.DataFrame, fallback: float = 0.005) -> float:
    """ATR as a fraction of current close. Used to scale tolerances per asset/TF."""
    if df is None or df.empty or "ATR" not in df.columns:
        return fallback
    atr = df["ATR"].iloc[-1]
    price = float(df["close"].iloc[-1])
    if pd.isna(atr) or price <= 0:
        return fallback
    return max(float(atr) / price, 0.0005)


def find_swings(df: pd.DataFrame, left=5, right=5):
    """
    Confirmed swing highs / lows by left/right comparison, augmented with
    'developing' swings on unconfirmed candles via volume + wick rejection
    so the engine doesn't lag the last `right` bars.
    """
    swing_highs = []
    swing_lows = []
    n = len(df)
    if n < 10:
        return [], []

    for i in range(left, n - right):
        val_high = df["high"].iloc[i]
        val_low = df["low"].iloc[i]

        is_high = True
        for j in range(i - left, i + right + 1):
            if df["high"].iloc[j] > val_high:
                is_high = False
                break
        if is_high:
            ts = df["timestamp"].iloc[i]
            ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            swing_highs.append((ts_str, float(val_high)))

        is_low = True
        for j in range(i - left, i + right + 1):
            if df["low"].iloc[j] < val_low:
                is_low = False
                break
        if is_low:
            ts = df["timestamp"].iloc[i]
            ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            swing_lows.append((ts_str, float(val_low)))

    # Developing swings on the unconfirmed tail
    atr = df["ATR"].iloc[-1] if "ATR" in df.columns else (df["high"] - df["low"]).mean()
    vol_mean = df["volume"].rolling(20).mean().iloc[-1] if len(df) >= 20 else df["volume"].mean()

    for idx in range(n - right, n):
        if idx < 0 or idx >= n:
            continue
        c = df.iloc[idx]
        high, low, open_p, close = float(c["high"]), float(c["low"]), float(c["open"]), float(c["close"])
        vol = float(c["volume"])
        body = abs(close - open_p)
        upper_wick = high - max(open_p, close)
        lower_wick = min(open_p, close) - low
        ts_str = c["timestamp"].isoformat() if hasattr(c["timestamp"], "isoformat") else str(c["timestamp"])

        if upper_wick > body * 1.2 and upper_wick > atr * 0.4 and vol > vol_mean * 1.1:
            if not swing_highs or swing_highs[-1][0] != ts_str:
                swing_highs.append((ts_str, high))

        if lower_wick > body * 1.2 and lower_wick > atr * 0.4 and vol > vol_mean * 1.1:
            if not swing_lows or swing_lows[-1][0] != ts_str:
                swing_lows.append((ts_str, low))

    return swing_highs[-5:], swing_lows[-5:]


def get_market_structure(swing_highs, swing_lows, df=None):
    """Compare consecutive swings to determine structure, BOS and CHoCH (close-confirmed)."""
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
            recent_closes = df["close"].tail(100)
            if not (recent_closes > sh1).any():
                high_state = "LH"  # swept but failed to close

    low_state = "HL" if sl2 > sl1 else "LL"
    if sl2 < sl1:
        if df is not None and not df.empty:
            recent_closes = df["close"].tail(100)
            if not (recent_closes < sl1).any():
                low_state = "HL"

    structure = f"{high_state}/{low_state}"

    bos_level = None
    bos_dir = None
    if df is not None and not df.empty:
        current_close = float(df["close"].iloc[-1])
        if current_close > sh1:
            bos_level = sh1
            bos_dir = "BULLISH"
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

    choch = {"detected": False, "direction": None, "level": None}
    if df is not None and not df.empty:
        current_close = float(df["close"].iloc[-1])
        if structure == "HH/HL" and current_close < sl2:
            choch = {"detected": True, "direction": "BEARISH", "level": sl2}
        elif structure == "LH/LL" and current_close > sh2:
            choch = {"detected": True, "direction": "BULLISH", "level": sh2}

    return structure, {"level": bos_level, "direction": bos_dir}, choch


def find_order_blocks(df: pd.DataFrame, swing_highs, swing_lows, lookback=50):
    """
    Unmitigated Order Blocks. Impulse threshold scales with ATR; HVN tag based on
    proximity to this dataframe's own value area (computed inline to avoid
    cross-TF leakage).
    """
    from price_action import get_value_area

    va = get_value_area(df, lookback=lookback)
    vah, val, poc = va["vah"], va["val"], va["poc"]
    atr_pct = _atr_pct(df)
    # HVN proximity threshold scales with ATR but clamped to a sensible band
    hvn_tol = max(0.003, min(0.01, atr_pct * 0.6))

    bullish_obs = []
    bearish_obs = []
    n = len(df)
    start_idx = max(3, n - lookback)

    for i in range(start_idx, n):
        # --- bullish impulse ---
        bullish_impulse = False
        impulse_start_idx = -1
        for k in (1, 2, 3):
            prev_close = df["close"].iloc[i - k]
            curr_close = df["close"].iloc[i]
            curr_atr = df["ATR"].iloc[i - k] if "ATR" in df.columns else np.nan
            if pd.isna(curr_atr) or curr_atr <= 0:
                threshold = prev_close * max(atr_pct, 0.005) * 1.5
            else:
                threshold = 2.0 * curr_atr
            if (curr_close - prev_close) >= threshold:
                bullish_impulse = True
                impulse_start_idx = i - k
                break

        if bullish_impulse:
            for j in range(impulse_start_idx - 1, max(0, impulse_start_idx - 10), -1):
                if df["close"].iloc[j] < df["open"].iloc[j]:
                    ob_high = float(df["high"].iloc[j])
                    ob_low = float(df["low"].iloc[j])
                    ob_ts = df["timestamp"].iloc[j].isoformat() if hasattr(df["timestamp"].iloc[j], "isoformat") else str(df["timestamp"].iloc[j])
                    if any(ob["timestamp"] == ob_ts for ob in bullish_obs):
                        break
                    is_hvn = False
                    for level in (vah, val, poc):
                        if level > 0 and abs((ob_high + ob_low) / 2 - level) / level <= hvn_tol:
                            is_hvn = True
                            break
                    bullish_obs.append({
                        "timestamp": ob_ts,
                        "top": ob_high,
                        "bottom": ob_low,
                        "created_at_idx": j,
                        "high_volume_node": is_hvn,
                    })
                    break

        # --- bearish impulse ---
        bearish_impulse = False
        impulse_start_idx = -1
        for k in (1, 2, 3):
            prev_close = df["close"].iloc[i - k]
            curr_close = df["close"].iloc[i]
            curr_atr = df["ATR"].iloc[i - k] if "ATR" in df.columns else np.nan
            if pd.isna(curr_atr) or curr_atr <= 0:
                threshold = prev_close * max(atr_pct, 0.005) * 1.5
            else:
                threshold = 2.0 * curr_atr
            if (prev_close - curr_close) >= threshold:
                bearish_impulse = True
                impulse_start_idx = i - k
                break

        if bearish_impulse:
            for j in range(impulse_start_idx - 1, max(0, impulse_start_idx - 10), -1):
                if df["close"].iloc[j] > df["open"].iloc[j]:
                    ob_high = float(df["high"].iloc[j])
                    ob_low = float(df["low"].iloc[j])
                    ob_ts = df["timestamp"].iloc[j].isoformat() if hasattr(df["timestamp"].iloc[j], "isoformat") else str(df["timestamp"].iloc[j])
                    if any(ob["timestamp"] == ob_ts for ob in bearish_obs):
                        break
                    is_hvn = False
                    for level in (vah, val, poc):
                        if level > 0 and abs((ob_high + ob_low) / 2 - level) / level <= hvn_tol:
                            is_hvn = True
                            break
                    bearish_obs.append({
                        "timestamp": ob_ts,
                        "top": ob_high,
                        "bottom": ob_low,
                        "created_at_idx": j,
                        "high_volume_node": is_hvn,
                    })
                    break

    # Unmitigated filter
    unmitigated_bullish = []
    for ob in bullish_obs:
        subsequent_closes = df["close"].iloc[ob["created_at_idx"] + 1:]
        if subsequent_closes.empty or subsequent_closes.min() >= ob["bottom"]:
            unmitigated_bullish.append({k: v for k, v in ob.items() if k != "created_at_idx"})

    unmitigated_bearish = []
    for ob in bearish_obs:
        subsequent_closes = df["close"].iloc[ob["created_at_idx"] + 1:]
        if subsequent_closes.empty or subsequent_closes.max() <= ob["top"]:
            unmitigated_bearish.append({k: v for k, v in ob.items() if k != "created_at_idx"})

    return {
        "bullish": unmitigated_bullish[-3:],
        "bearish": unmitigated_bearish[-3:],
    }


def find_fvg(df: pd.DataFrame, lookback=50):
    """
    3-candle Fair Value Gaps. Mitigation check uses the full subsequent series
    (including the most recent candle) rather than excluding the last bar.
    """
    bullish_fvgs = []
    bearish_fvgs = []
    n = len(df)
    start_idx = max(2, n - lookback)
    current_price = float(df["close"].iloc[-1])

    for i in range(start_idx, n):
        high_2 = float(df["high"].iloc[i - 2])
        low_0 = float(df["low"].iloc[i])
        if low_0 > high_2:
            subsequent = df.iloc[i + 1:]
            filled = (not subsequent.empty) and (subsequent["low"].min() <= high_2)
            bullish_fvgs.append({
                "top": low_0,
                "bottom": high_2,
                "timestamp": df["timestamp"].iloc[i].isoformat() if hasattr(df["timestamp"].iloc[i], "isoformat") else str(df["timestamp"].iloc[i]),
                "filled": bool(filled),
            })

        low_2 = float(df["low"].iloc[i - 2])
        high_0 = float(df["high"].iloc[i])
        if low_2 > high_0:
            subsequent = df.iloc[i + 1:]
            filled = (not subsequent.empty) and (subsequent["high"].max() >= low_2)
            bearish_fvgs.append({
                "top": low_2,
                "bottom": high_0,
                "timestamp": df["timestamp"].iloc[i].isoformat() if hasattr(df["timestamp"].iloc[i], "isoformat") else str(df["timestamp"].iloc[i]),
                "filled": bool(filled),
            })

    nearest_bull = None
    below_fvgs = [f for f in bullish_fvgs if not f["filled"] and f["top"] <= current_price]
    if below_fvgs:
        nearest_bull = max(below_fvgs, key=lambda x: x["top"])

    nearest_bear = None
    above_fvgs = [f for f in bearish_fvgs if not f["filled"] and f["bottom"] >= current_price]
    if above_fvgs:
        nearest_bear = min(above_fvgs, key=lambda x: x["bottom"])

    return {
        "nearest_bullish_fvg": nearest_bull,
        "nearest_bearish_fvg": nearest_bear,
        "bullish_fvgs": bullish_fvgs[-5:],
        "bearish_fvgs": bearish_fvgs[-5:],
    }


def find_liquidity_levels(swing_highs, swing_lows, atr_pct: float = 0.002):
    """
    Cluster swing highs/lows within an ATR-scaled tolerance. Returns top 3
    buy-side (highs) and sell-side (lows) liquidity pools.
    """
    tolerance = max(0.001, min(0.01, atr_pct * 0.5))
    high_prices = sorted([s[1] for s in swing_highs])
    low_prices = sorted([s[1] for s in swing_lows])

    bsl_levels = []
    used_highs = set()
    for i, h1 in enumerate(high_prices):
        if i in used_highs:
            continue
        cluster = [h1]
        for j, h2 in enumerate(high_prices[i + 1:]):
            idx = i + 1 + j
            if idx in used_highs:
                continue
            if h1 > 0 and (h2 - h1) / h1 <= tolerance:
                cluster.append(h2)
                used_highs.add(idx)
        bsl_levels.append(max(cluster))

    ssl_levels = []
    used_lows = set()
    for i, l1 in enumerate(low_prices):
        if i in used_lows:
            continue
        cluster = [l1]
        for j, l2 in enumerate(low_prices[i + 1:]):
            idx = i + 1 + j
            if idx in used_lows:
                continue
            if l1 > 0 and (l2 - l1) / l1 <= tolerance:
                cluster.append(l2)
                used_lows.add(idx)
        ssl_levels.append(min(cluster))

    return {
        "buy_side": sorted(bsl_levels, reverse=True)[:3],
        "sell_side": sorted(ssl_levels)[:3],
    }


def get_premium_discount(swing_high, swing_low, current_price):
    """Premium/discount + OTE (61.8%-78.6% retracement) of the last 5 swings."""
    if swing_high <= swing_low:
        return {"zone": "neutral", "ote_zone": {"low": 0, "high": 0}, "in_ote": False}

    range_size = swing_high - swing_low
    midpoint = swing_low + 0.5 * range_size
    zone = "premium" if current_price > midpoint else "discount"
    ote_low = swing_high - 0.786 * range_size
    ote_high = swing_high - 0.618 * range_size
    in_ote = ote_low <= current_price <= ote_high
    return {
        "zone": zone,
        "ote_zone": {"low": round(ote_low, 4), "high": round(ote_high, 4)},
        "in_ote": in_ote,
    }


def build_smc_context(df_4h, df_1h, df_15m):
    """Build a multi-timeframe SMC context for the snapshot."""
    context = {}
    for tf, df in (("4h", df_4h), ("1h", df_1h), ("15m", df_15m)):
        if df is None or df.empty:
            continue

        current_price = float(df["close"].iloc[-1])
        if tf == "4h":
            left_p, right_p = 5, 5
        elif tf == "1h":
            left_p, right_p = 4, 3
        else:
            left_p, right_p = 3, 2

        sh, sl = find_swings(df, left=left_p, right=right_p)
        struct, bos, choch = get_market_structure(sh, sl, df)
        obs = find_order_blocks(df, sh, sl)
        fvgs = find_fvg(df)
        liq = find_liquidity_levels(sh, sl, atr_pct=_atr_pct(df))

        if sh and sl:
            max_h = max(x[1] for x in sh)
            min_l = min(x[1] for x in sl)
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
            "premium_discount": pd_zone,
        }

    return context
