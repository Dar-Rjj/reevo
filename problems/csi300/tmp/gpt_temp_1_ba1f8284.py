import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Ultra-short momentum: 2-day log return
    ultra_short_momentum = np.log(df['close'] / df['close'].shift(2))
    
    # Volume normalization: current volume divided by 10-day rolling median volume
    normalized_volume_spikes = df['volume'] / df['volume'].rolling(10).median()
    
    # Intraday strength: position of close within daily range, smoothed with 3-day rolling mean
    intraday_strength = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-7)
    intraday_strength = intraday_strength.rolling(3).mean()
    
    # Combined momentum and volume factor, weighted by recent intraday strength
    momentum_volume_combined = ultra_short_momentum * normalized_volume_spikes * intraday_strength
    
    # Logarithmic volume transformation with dampening
    log_volume = np.log1p(df['volume'])
    
    # Rolling volatility normalization: 10-day rolling standard deviation of returns
    rolling_volatility = df['close'].pct_change().rolling(10).std()
    
    # Normalize combined factor by rolling volatility
    normalized_factor = momentum_volume_combined / (rolling_volatility + 1e-7)
    
    # Final factor: normalized factor multiplied by log volume
    factor = normalized_factor * log_volume
    
    # Smoothing with Hamming window of size 5
    window_size = 5
    hamming_window = np.hamming(window_size)
    smoothed_factor = factor.rolling(window=window_size, center=True).apply(lambda x: np.sum(x * hamming_window))
    
    return smoothed_factor
