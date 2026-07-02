import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Ultra-short momentum: 3-day % change in close price
    ultra_short_momentum = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    
    # Volume normalization: current volume divided by 15-day rolling average volume
    normalized_volume_spikes = df['volume'] / df['volume'].rolling(15).mean()
    
    # Intraday strength: position of close within daily range
    intraday_strength = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-7)
    
    # Combined momentum and volume factor
    momentum_volume_combined = ultra_short_momentum * normalized_volume_spikes
    
    # Combine intraday strength with momentum-volume factor
    combined_signal = momentum_volume_combined * intraday_strength
    
    # Logarithmic volume transformation
    log_volume = np.log(df['volume'] + 1)
    
    # Final factor: combined signal multiplied by log volume
    factor = combined_signal * log_volume
    
    # Smoothing with Hamming window
    window_size = 7
    hamming_window = np.hamming(window_size)
    smoothed_factor = factor.rolling(window=window_size, center=True).apply(lambda x: np.sum(x * hamming_window))
    
    return smoothed_factor
