import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Ultra-short momentum with volatility adjustment (2-day % change / 5-day rolling std)
    momentum = (df['close'] - df['close'].shift(2)) / df['close'].shift(2)
    vol_adjusted_momentum = momentum / df['close'].rolling(5).std()
    
    # Volume confirmation (log-transform with sign preservation)
    signed_log_volume = np.sign(df['volume']) * np.log1p(np.abs(df['volume']))
    
    # Intraday strength weighted by recent volatility
    intraday_strength = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-7)
    vol_weight = df['high'].rolling(3).std() / df['low'].rolling(3).std()
    weighted_strength = intraday_strength * vol_weight
    
    # Multiplicative combination core
    core_factor = vol_adjusted_momentum * signed_log_volume * weighted_strength
    
    # Micro-volatility adjustment (range/close ratio)
    micro_vol = (df['high'] - df['low']) / df['close']
    adjusted_factor = core_factor / (micro_vol + 0.01)  # Add small constant to avoid div/0
    
    # Light Hamming smoothing (3-period)
    window_size = 3
    hamming_window = np.hamming(window_size)
    smoothed_factor = adjusted_factor.rolling(window=window_size, center=True).apply(
        lambda x: np.sum(x * hamming_window))
    
    return smoothed_factor
