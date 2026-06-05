"""
Indicator Engine - SMC + Momentum + Volume + Volatility
"""

import pandas as pd
import pandas_ta as ta
import numpy as np


def calculate_indicators(df: pd.DataFrame, timeframe: str = None) -> pd.DataFrame:
    df = df.sort_values("timestamp").reset_index(drop=True)

    # === TREND ===
    df["EMA_50"] = ta.ema(df["close"], length=50)
    df["EMA_200"] = ta.ema(df["close"], length=200)

    # === VOLATILITY ===
    df["ATR"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    atr_mean = df["ATR"].rolling(50).mean()
    atr_std = df["ATR"].rolling(50).std()
    df["ATR_Z"] = (df["ATR"] - atr_mean) / atr_std
    # ATR percentile rank over a 100-bar rolling window (0..1)
    df["ATR_Pct"] = df["ATR"].rolling(100, min_periods=20).rank(pct=True)

    # === MOMENTUM ===
    df["RSI"] = ta.rsi(df["close"], length=14)
    macd = ta.macd(df["close"])
    if macd is not None:
        df["MACD"] = macd["MACD_12_26_9"]
        df["MACD_SIGNAL"] = macd["MACDs_12_26_9"]
    else:
        df["MACD"] = np.nan
        df["MACD_SIGNAL"] = np.nan

    # === VOLUME ===
    vol_mean = df["volume"].rolling(20).mean()
    vol_std = df["volume"].rolling(20).std()
    df["VOL_Z"] = (df["volume"] - vol_mean) / vol_std

    # === SUPERTREND ===
    try:
        st = ta.supertrend(df["high"], df["low"], df["close"], length=10, multiplier=3.0)
        if st is not None:
            df["SUPERT_10_3.0"] = st["SUPERT_10_3.0"]
            df["SUPERTd_10_3.0"] = st["SUPERTd_10_3.0"]
        else:
            df["SUPERT_10_3.0"] = np.nan
            df["SUPERTd_10_3.0"] = np.nan
    except Exception:
        df["SUPERT_10_3.0"] = np.nan
        df["SUPERTd_10_3.0"] = np.nan

    # === STOCHASTIC RSI ===
    try:
        stochrsi = ta.stochrsi(df["close"], length=14, rsi_length=14, k=3, d=3)
        if stochrsi is not None:
            df["STOCHRSIk_14_14_3_3"] = stochrsi["STOCHRSIk_14_14_3_3"]
            df["STOCHRSId_14_14_3_3"] = stochrsi["STOCHRSId_14_14_3_3"]
        else:
            df["STOCHRSIk_14_14_3_3"] = np.nan
            df["STOCHRSId_14_14_3_3"] = np.nan
    except Exception:
        df["STOCHRSIk_14_14_3_3"] = np.nan
        df["STOCHRSId_14_14_3_3"] = np.nan

    # === ADX ===
    try:
        adx = ta.adx(df["high"], df["low"], df["close"], length=14)
        if adx is not None:
            df["ADX"] = adx["ADX_14"]
        else:
            df["ADX"] = np.nan
    except Exception:
        df["ADX"] = np.nan

    # === BOLLINGER BANDS (for real squeeze detection) ===
    try:
        bb = ta.bbands(df["close"], length=20, std=2.0)
        if bb is not None:
            cols = list(bb.columns)
            lower = next((c for c in cols if c.startswith("BBL_")), None)
            middle = next((c for c in cols if c.startswith("BBM_")), None)
            upper = next((c for c in cols if c.startswith("BBU_")), None)
            if lower and middle and upper:
                df["BB_LOWER"] = bb[lower]
                df["BB_MIDDLE"] = bb[middle]
                df["BB_UPPER"] = bb[upper]
                df["BB_WIDTH"] = (df["BB_UPPER"] - df["BB_LOWER"]) / df["BB_MIDDLE"].replace(0, np.nan)
            else:
                df["BB_WIDTH"] = np.nan
        else:
            df["BB_WIDTH"] = np.nan
    except Exception:
        df["BB_WIDTH"] = np.nan

    # === CVD PROXY (candle-position) ===
    range_val = df["high"] - df["low"]
    multiplier = np.where(range_val > 0, ((df["close"] - df["low"]) - (df["high"] - df["close"])) / range_val, 0.0)
    df["cvd"] = (df["volume"] * multiplier).cumsum()

    # === SWING POINTS ===
    df["swing_high"] = df["high"] == df["high"].rolling(5, center=True).max()
    df["swing_low"] = df["low"] == df["low"].rolling(5, center=True).min()

    # === FAIR VALUE GAPS (per-candle flags) ===
    high_2 = df["high"].shift(2)
    low_2 = df["low"].shift(2)
    df["fvg_bull"] = df["low"] > high_2
    df["fvg_bear"] = df["high"] < low_2
    df["fvg_bull_top"] = np.where(df["fvg_bull"], df["low"], np.nan)
    df["fvg_bull_bottom"] = np.where(df["fvg_bull"], high_2, np.nan)
    df["fvg_bear_top"] = np.where(df["fvg_bear"], low_2, np.nan)
    df["fvg_bear_bottom"] = np.where(df["fvg_bear"], df["high"], np.nan)

    return df


def detect_volatility_regime(df: pd.DataFrame) -> dict:
    """
    Categorise the current volatility regime using ATR percentile and BB width.
    Returns {"regime": "compressed|normal|expanded", "atr_pct": float, "bb_width": float}.
    """
    if df is None or df.empty:
        return {"regime": "unknown", "atr_percentile": None, "bb_width": None}
    atr_pct = df["ATR_Pct"].iloc[-1] if "ATR_Pct" in df.columns else None
    bb_width = df["BB_WIDTH"].iloc[-1] if "BB_WIDTH" in df.columns else None

    if atr_pct is None or pd.isna(atr_pct):
        regime = "unknown"
    elif atr_pct < 0.25:
        regime = "compressed"
    elif atr_pct > 0.75:
        regime = "expanded"
    else:
        regime = "normal"

    return {
        "regime": regime,
        "atr_percentile": float(atr_pct) if atr_pct is not None and not pd.isna(atr_pct) else None,
        "bb_width": float(bb_width) if bb_width is not None and not pd.isna(bb_width) else None,
    }


def detect_regime(df_4h: pd.DataFrame, df_1h: pd.DataFrame, df_15m: pd.DataFrame,
                  smc_context: dict = None) -> str:
    """
    Global market regime. Prefers the SMC structure read when available;
    falls back to weighted EMA/RSI scoring across timeframes when not.
    """
    if smc_context:
        struct_4h = smc_context.get("4h", {}).get("structure", "NEUTRAL")
        struct_1h = smc_context.get("1h", {}).get("structure", "NEUTRAL")
        bos_1h = smc_context.get("1h", {}).get("bos", {}) or {}
        choch_1h = smc_context.get("1h", {}).get("choch", {}) or {}

        def is_bull(s):
            return "HH" in s and "HL" in s
        def is_bear(s):
            return "LH" in s and "LL" in s

        if is_bull(struct_4h):
            return "Trending Bullish"
        if is_bear(struct_4h):
            return "Trending Bearish"

        if bos_1h.get("level") is None and not choch_1h.get("detected") and not (is_bull(struct_1h) or is_bear(struct_1h)):
            return "Ranging / Sideways"

    def tf_score(df):
        if df is None or df.empty or "EMA_50" not in df.columns or "EMA_200" not in df.columns:
            return 0
        latest = df.iloc[-1]
        score = 0
        if pd.notna(latest["EMA_50"]) and pd.notna(latest["EMA_200"]):
            if latest["EMA_50"] > latest["EMA_200"]:
                score += 2
            else:
                score -= 2
        if "RSI" in df.columns and pd.notna(latest["RSI"]):
            rsi = latest["RSI"]
            if rsi > 60:
                score += 1
            elif rsi < 40:
                score -= 1
        return score

    weighted = 0.5 * tf_score(df_4h) + 0.3 * tf_score(df_1h) + 0.2 * tf_score(df_15m)
    if weighted > 0.8:
        return "Trending Bullish"
    if weighted < -0.8:
        return "Trending Bearish"
    return "Ranging / Sideways"


def get_market_structure(df: pd.DataFrame) -> dict:
    """Legacy single-TF structure view retained for callers that still use it."""
    latest = df.iloc[-1]
    swing_highs = df[df["swing_high"]].tail(3) if "swing_high" in df.columns else df.iloc[0:0]
    swing_lows = df[df["swing_low"]].tail(3) if "swing_low" in df.columns else df.iloc[0:0]
    recent = df.tail(20)
    bull_fvg = recent[recent["fvg_bull"]] if "fvg_bull" in recent.columns else recent.iloc[0:0]
    bear_fvg = recent[recent["fvg_bear"]] if "fvg_bear" in recent.columns else recent.iloc[0:0]

    return {
        "price": float(latest["close"]),
        "trend": "bullish" if latest["EMA_50"] > latest["EMA_200"] else "bearish",
        "rsi": float(latest["RSI"]) if pd.notna(latest.get("RSI")) else None,
        "atr_z": float(latest["ATR_Z"]) if pd.notna(latest.get("ATR_Z")) else None,
        "buy_side_liquidity": swing_highs["high"].tolist(),
        "sell_side_liquidity": swing_lows["low"].tolist(),
        "bull_fvg": None if bull_fvg.empty else {
            "top": float(bull_fvg.iloc[-1]["fvg_bull_top"]),
            "bottom": float(bull_fvg.iloc[-1]["fvg_bull_bottom"]),
        },
        "bear_fvg": None if bear_fvg.empty else {
            "top": float(bear_fvg.iloc[-1]["fvg_bear_top"]),
            "bottom": float(bear_fvg.iloc[-1]["fvg_bear_bottom"]),
        },
    }
