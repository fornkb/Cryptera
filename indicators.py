"""
Indicator Engine
SMC + Momentum + Volume + Volatility + Advanced Indicators
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
            # columns in pandas_ta supertrend: SUPERT_10_3.0, SUPERTd_10_3.0, SUPERTl_10_3.0, SUPERTs_10_3.0
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

    # === CVD PROXY ===
    range_val = df["high"] - df["low"]
    multiplier = np.where(range_val > 0, ((df["close"] - df["low"]) - (df["high"] - df["close"])) / range_val, 0.0)
    df["cvd"] = (df["volume"] * multiplier).cumsum()

    # === SWING POINTS (Historical / Liquidity Pools) ===
    df["swing_high"] = df["high"] == df["high"].rolling(5, center=True).max()
    df["swing_low"] = df["low"] == df["low"].rolling(5, center=True).min()

    # === FAIR VALUE GAPS ===
    high_2 = df["high"].shift(2)
    low_2 = df["low"].shift(2)

    df["fvg_bull"] = df["low"] > high_2
    df["fvg_bear"] = df["high"] < low_2

    df["fvg_bull_top"] = np.where(df["fvg_bull"], df["low"], np.nan)
    df["fvg_bull_bottom"] = np.where(df["fvg_bull"], high_2, np.nan)
    df["fvg_bear_top"] = np.where(df["fvg_bear"], low_2, np.nan)
    df["fvg_bear_bottom"] = np.where(df["fvg_bear"], df["high"], np.nan)

    return df


def get_market_structure(df: pd.DataFrame) -> dict:
    latest = df.iloc[-1]

    swing_highs = df[df["swing_high"]].tail(3)
    swing_lows = df[df["swing_low"]].tail(3)

    recent = df.tail(20)
    bull_fvg = recent[recent["fvg_bull"]]
    bear_fvg = recent[recent["fvg_bear"]]

    return {
        "price": float(latest["close"]),
        "trend": "bullish" if latest["EMA_50"] > latest["EMA_200"] else "bearish",
        "rsi": float(latest["RSI"]),
        "atr_z": float(latest["ATR_Z"]),
        "buy_side_liquidity": swing_highs["high"].tolist(),
        "sell_side_liquidity": swing_lows["low"].tolist(),
        "bull_fvg": None if bull_fvg.empty else {
            "top": float(bull_fvg.iloc[-1]["fvg_bull_top"]),
            "bottom": float(bull_fvg.iloc[-1]["fvg_bull_bottom"])
        },
        "bear_fvg": None if bear_fvg.empty else {
            "top": float(bear_fvg.iloc[-1]["fvg_bear_top"]),
            "bottom": float(bear_fvg.iloc[-1]["fvg_bear_bottom"])
        }
    }


def detect_regime(df_4h: pd.DataFrame, df_1h: pd.DataFrame, df_15m: pd.DataFrame) -> str:
    """
    Determine the global market regime using a weighted score across multiple timeframes.
    - 4h (HTF): Weight 0.5
    - 1h (MTF): Weight 0.3
    - 15m (LTF): Weight 0.2
    """
    from price_action import get_value_area
    from smc import find_swings, get_market_structure as get_smc_structure

    def get_tf_score(df):
        if df is None or df.empty or 'EMA_50' not in df.columns or 'EMA_200' not in df.columns:
            return 0
        latest = df.iloc[-1]
        score = 0
        
        # 1. EMA Trend
        if latest['EMA_50'] > latest['EMA_200']:
            score += 2
        else:
            score -= 2
            
        # 2. RSI Overbought/Oversold
        if 'RSI' in df.columns and not pd.isna(latest['RSI']):
            rsi = latest['RSI']
            if rsi > 60:
                score += 1
            elif rsi < 40:
                score -= 1
                
        return score

    # Check for Ranging Regime purely via Value Area + Structure on 1H
    if df_1h is not None and not df_1h.empty:
        current_price = float(df_1h['close'].iloc[-1])
        va = get_value_area(df_1h)
        if va['val'] <= current_price <= va['vah']:
            # If price is trapped inside Value Area High and Low, check if it's lacking structure
            sh, sl = find_swings(df_1h, left=4, right=3)
            struct, bos, choch = get_smc_structure(sh, sl, df_1h)
            if bos["level"] is None and not choch["detected"]:
                return "Ranging / Sideways"

    score_4h = get_tf_score(df_4h)
    score_1h = get_tf_score(df_1h)
    score_15m = get_tf_score(df_15m)
    
    weighted_score = (0.5 * score_4h) + (0.3 * score_1h) + (0.2 * score_15m)
    
    if weighted_score > 0.8:
        return "Trending Bullish"
    elif weighted_score < -0.8:
        return "Trending Bearish"
    else:
        return "Ranging / Sideways"
