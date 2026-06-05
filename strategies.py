"""
Cryptera v3.1 - Confluence Scoring Engine

Eight-component score (C1-C8) mirroring the LLM rubric exactly so the model
can verify rather than override. Volume gate is reported as a separate gate
state (Component 9).
"""

import pandas as pd
import numpy as np


def _atr_pct(df: pd.DataFrame) -> float:
    """ATR as a fraction of current price. Falls back to 0.5% when ATR is missing."""
    if df is None or df.empty or "ATR" not in df.columns:
        return 0.005
    atr = df["ATR"].iloc[-1]
    price = float(df["close"].iloc[-1])
    if pd.isna(atr) or price <= 0:
        return 0.005
    return float(atr) / price


def _structure_bias(structure: str) -> str:
    """Map a structure string ('HH/HL', 'LH/LL', ...) to bull/bear/mixed."""
    if not structure:
        return "mixed"
    has_hh = "HH" in structure
    has_hl = "HL" in structure
    has_lh = "LH" in structure
    has_ll = "LL" in structure
    if has_hh and has_hl:
        return "bull"
    if has_lh and has_ll:
        return "bear"
    if has_hh or has_hl:
        return "bull"
    if has_lh or has_ll:
        return "bear"
    return "mixed"


def derive_trend_bias(smc_context: dict) -> str:
    """Single source of truth for trend bias: 4H structure first, EMA fallback."""
    struct_4h = smc_context.get("4h", {}).get("structure", "NEUTRAL") if smc_context else "NEUTRAL"
    bias = _structure_bias(struct_4h)
    if bias == "bull":
        return "bullish"
    if bias == "bear":
        return "bearish"
    return "neutral"


# ---------------- Component scoring ---------------- #

def _c1_trend_alignment(smc_context: dict, windowed: dict) -> tuple[int, str]:
    """ADX-tiered trend alignment. Max +25, +10 on 4H/1H conflict."""
    struct_4h = smc_context.get("4h", {}).get("structure", "NEUTRAL")
    struct_1h = smc_context.get("1h", {}).get("structure", "NEUTRAL")
    adx_4h = float(windowed.get("4h", {}).get("adx", 0.0) or 0.0)

    b4 = _structure_bias(struct_4h)
    b1 = _structure_bias(struct_1h)
    agree = b4 == b1 and b4 in ("bull", "bear")

    if not agree:
        return 10, f"4H/1H conflict ({struct_4h} vs {struct_1h})"
    if adx_4h < 25:
        return 10, f"weak ADX {adx_4h:.1f}"
    if adx_4h < 41:
        return 15, f"medium ADX {adx_4h:.1f}"
    if adx_4h < 56:
        return 20, f"strong ADX {adx_4h:.1f}"
    return 25, f"very strong ADX {adx_4h:.1f}"


def _c2_ob_proximity(current_price: float, smc_context: dict, trend_bias: str) -> tuple[int, str]:
    """Distance-tiered OB / FVG proximity on 15m or 1H. Max +15."""
    if trend_bias not in ("bullish", "bearish"):
        return 0, "neutral bias"
    direction = "bullish" if trend_bias == "bullish" else "bearish"
    best_dist = None
    best_label = "none"

    for tf in ("15m", "1h"):
        ctx = smc_context.get(tf, {}) or {}
        for ob in ctx.get("order_blocks", {}).get(direction, []) or []:
            top = ob.get("top")
            bottom = ob.get("bottom")
            if top is None or bottom is None:
                continue
            edge = bottom if direction == "bullish" else top
            dist = abs(current_price - edge) / current_price
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_label = f"{tf.upper()} {direction} OB"
        fvgs = ctx.get("fvg", {}) or {}
        target_fvg = fvgs.get(f"nearest_{direction}_fvg")
        if target_fvg and not target_fvg.get("filled", True):
            top = target_fvg.get("top")
            bottom = target_fvg.get("bottom")
            if top is not None and bottom is not None:
                edge = top if direction == "bullish" else bottom
                dist = abs(current_price - edge) / current_price
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_label = f"{tf.upper()} {direction} FVG"

    if best_dist is None:
        return 0, "no nearby OB/FVG"
    if best_dist <= 0.003:
        return 15, f"{best_label} {best_dist*100:.2f}% away"
    if best_dist <= 0.005:
        return 10, f"{best_label} {best_dist*100:.2f}% away"
    if best_dist <= 0.01:
        return 5, f"{best_label} {best_dist*100:.2f}% away"
    return 0, f"nearest {best_label} {best_dist*100:.2f}% away (>1%)"


def detect_liquidity_sweep(df_15m: pd.DataFrame, trend_bias: str) -> bool:
    """Sweep + reclaim in last 5 candles relative to the prior 30 swing pool."""
    if df_15m is None or df_15m.empty or len(df_15m) < 10:
        return False
    if trend_bias not in ("bullish", "bearish"):
        return False
    n = len(df_15m)
    for offset in range(5, 0, -1):
        i = n - offset
        if i <= 0:
            continue
        candle = df_15m.iloc[i]
        scan_before = df_15m.iloc[:i].tail(30)
        if scan_before.empty or "swing_low" not in scan_before.columns:
            continue
        if trend_bias == "bullish":
            lows = scan_before[scan_before["swing_low"]]["low"]
            if not lows.empty:
                key = lows.min()
                if candle["low"] < key and candle["close"] > key:
                    return True
        else:
            highs = scan_before[scan_before["swing_high"]]["high"]
            if not highs.empty:
                key = highs.max()
                if candle["high"] > key and candle["close"] < key:
                    return True
    return False


def _c3_sweep(df_15m: pd.DataFrame, trend_bias: str) -> tuple[int, str]:
    swept = detect_liquidity_sweep(df_15m, trend_bias)
    return (10, "sweep + reclaim") if swept else (0, "no recent sweep")


def _c4_momentum(df_1h: pd.DataFrame, windowed: dict, trend_bias: str) -> tuple[int, str]:
    """Base +10 if 1H RSI/MACD align with bias, ±5 RSI slope adjustment. Cap [0, 15]."""
    if df_1h is None or df_1h.empty:
        return 0, "no 1H data"
    latest = df_1h.iloc[-1]
    rsi = float(latest.get("RSI", 50.0) or 50.0)
    macd = latest.get("MACD")
    macd_sig = latest.get("MACD_SIGNAL")
    macd_ok = pd.notna(macd) and pd.notna(macd_sig)

    base = 0
    if trend_bias == "bullish":
        if rsi < 65 and macd_ok and macd > macd_sig:
            base = 10
    elif trend_bias == "bearish":
        if rsi > 35 and macd_ok and macd < macd_sig:
            base = 10

    rsi_series = windowed.get("1h", {}).get("rsi_series_last_5", []) or []
    slope = 0.0
    slope_adj = 0
    label = "no RSI slope"
    if len(rsi_series) >= 3:
        slope = float(rsi_series[-1]) - float(rsi_series[-3])
        if slope > 5:
            slope_adj = 5
            label = f"slope +{slope:.1f} recovering"
        elif slope < -5:
            slope_adj = -5
            label = f"slope {slope:.1f} deteriorating"
        else:
            label = f"slope {slope:+.1f} neutral"
    final = max(0, min(15, base + slope_adj))
    return final, f"RSI {rsi:.1f}, {label}"


def _c5_fvg_magnet(current_price: float, smc_context: dict, trend_bias: str) -> tuple[int, str]:
    """Unfilled draw-on-liquidity FVG on 1H or 4H within 3% of price. Max +15."""
    if trend_bias not in ("bullish", "bearish"):
        return 0, "neutral bias"
    direction = "bullish" if trend_bias == "bullish" else "bearish"
    for tf in ("1h", "4h"):
        fvgs = smc_context.get(tf, {}).get("fvg", {}) or {}
        target = fvgs.get(f"nearest_{direction}_fvg")
        if target and not target.get("filled", True):
            top = target.get("top")
            bottom = target.get("bottom")
            if top is None or bottom is None:
                continue
            mid = (top + bottom) / 2.0
            if mid <= 0:
                continue
            dist = abs(current_price - mid) / current_price
            if dist <= 0.03:
                return 15, f"{tf.upper()} {direction} FVG {dist*100:.2f}% away"
    return 0, "no qualifying FVG within 3%"


def _c6_ote(smc_context: dict) -> tuple[int, str]:
    """OTE on 1H or 4H. Max +10."""
    for tf in ("1h", "4h"):
        if smc_context.get(tf, {}).get("premium_discount", {}).get("in_ote", False):
            return 10, f"in OTE on {tf.upper()}"
    return 0, "not in OTE"


def _c7_cvd_alignment(windowed: dict, trend_bias: str) -> tuple[int, str, bool]:
    """CVD alignment across TFs. Max +10. Absorption warning when 15m diverges from 4H."""
    if trend_bias not in ("bullish", "bearish"):
        return 0, "neutral bias", False
    def sign(x):
        try:
            v = float(x)
        except Exception:
            return 0
        if v > 0:
            return 1
        if v < 0:
            return -1
        return 0
    target = 1 if trend_bias == "bullish" else -1
    cvd_4h = sign(windowed.get("4h", {}).get("cvd_window_delta", 0))
    cvd_1h = sign(windowed.get("1h", {}).get("cvd_window_delta", 0))
    cvd_15m = sign(windowed.get("15m", {}).get("cvd_window_delta", 0))
    absorption = cvd_15m != 0 and cvd_4h != 0 and cvd_15m == -cvd_4h
    if absorption:
        return 0, "15m CVD opposes 4H — absorption warning", True
    matches = sum(1 for s in (cvd_4h, cvd_1h, cvd_15m) if s == target)
    if matches >= 2:
        label = "strong CVD alignment" if matches == 3 else "CVD aligned on 2 TFs"
        return 10, label, False
    return 0, f"CVD aligned on {matches} TF(s)", False


def _c8_stochrsi(windowed: dict, trend_bias: str) -> tuple[int, str]:
    """1H StochRSI overbought (sell) / oversold (buy) confirmation. Max +5."""
    k = float(windowed.get("1h", {}).get("stochrsi_k", 50.0) or 50.0)
    if trend_bias == "bearish" and k > 80:
        return 5, f"overbought K {k:.1f}"
    if trend_bias == "bullish" and k < 20:
        return 5, f"oversold K {k:.1f}"
    return 0, f"K {k:.1f} neutral"


def _volume_gate(windowed: dict) -> dict:
    rel_vol = float(windowed.get("15m", {}).get("relative_volume", 1.0) or 1.0)
    if rel_vol < 0.1:
        state = "HARD_GATE"
    elif rel_vol < 0.3:
        state = "LOW_VOL_WARNING"
    else:
        state = "CLEAR"
    return {"state": state, "rel_vol_15m": rel_vol}


def evaluate_strategies(df_4h, df_1h, df_15m, orderbook, funding, sentiment,
                        smc_context: dict, windowed_indicators: dict) -> dict:
    """
    Build the C1-C8 score matrix aligned with the LLM rubric. Returns a flat dict
    containing trend bias, volume gate state, total score and per-component
    breakdown with one-line justifications for each component.
    """
    if df_15m is None or df_15m.empty:
        raise ValueError("df_15m is required for strategy evaluation")

    current_price = float(df_15m["close"].iloc[-1])
    trend_bias = derive_trend_bias(smc_context)

    c1, n1 = _c1_trend_alignment(smc_context, windowed_indicators)
    c2, n2 = _c2_ob_proximity(current_price, smc_context, trend_bias)
    c3, n3 = _c3_sweep(df_15m, trend_bias)
    c4, n4 = _c4_momentum(df_1h, windowed_indicators, trend_bias)
    c5, n5 = _c5_fvg_magnet(current_price, smc_context, trend_bias)
    c6, n6 = _c6_ote(smc_context)
    c7, n7, absorption = _c7_cvd_alignment(windowed_indicators, trend_bias)
    c8, n8 = _c8_stochrsi(windowed_indicators, trend_bias)

    breakdown = {
        "c1_trend_alignment": c1,
        "c2_ob_proximity": c2,
        "c3_liquidity_sweep": c3,
        "c4_momentum": c4,
        "c5_fvg_magnet": c5,
        "c6_ote_bonus": c6,
        "c7_cvd_alignment": c7,
        "c8_stochrsi": c8,
    }
    notes = {
        "c1_trend_alignment": n1,
        "c2_ob_proximity": n2,
        "c3_liquidity_sweep": n3,
        "c4_momentum": n4,
        "c5_fvg_magnet": n5,
        "c6_ote_bonus": n6,
        "c7_cvd_alignment": n7,
        "c8_stochrsi": n8,
    }
    total = min(100, sum(breakdown.values()))
    gate = _volume_gate(windowed_indicators)

    return {
        "trend_bias": trend_bias,
        "liquidity_sweep": c3 > 0,
        "cvd_absorption_warning": absorption,
        "volume_gate": gate,
        "confluence_score": int(total),
        "confluence_breakdown": breakdown,
        "confluence_notes": notes,
        "current_price": current_price,
    }
