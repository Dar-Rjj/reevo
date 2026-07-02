import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Ultra-short momentum (1-day vs 2-day close)
    momentum = (df['close'] - df['close'].shift(2)) / df['close'].shift(2)
    
    # Volume spike detection (current vs 5-day median)
    vol_median = df['volume'].rolling(5).median()
    vol_spike = df['volume'] / (vol_median + 1e-7)
    log_vol_spike = np.log1p(vol_spike)
    
    # Intraday strength (close position within daily range)
    daily_range = df['high'] - df['low']
    close_position = (df['close'] - df['low']) / (daily_range + 1e-7)
    
    # Combined factor: momentum * volume_spike * intraday_strength
    raw_factor = momentum * log_vol_spike * close_position
    
    # Volatility scaling (5-day rolling std of returns)
    returns = df['close'].pct_change()
    vol_scaling = returns.rolling(5).std()
    scaled_factor = raw_factor / (vol_scaling + 1e-7)
    
    # Light smoothing with 5-day Hamming window
    window = np.hamming(5)
    smoothed_factor = scaled_factor.rolling(5).apply(lambda x: np.sum(window * x))
    
    return smoothed_factor
