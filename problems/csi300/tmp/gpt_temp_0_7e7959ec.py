import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Ultra-short momentum (2-day % change)
    ultra_short_momentum = (df['close'] - df['close'].shift(2)) / df['close'].shift(2)
    
    # Volume spikes normalized to 10-day rolling average
    avg_volume = df['volume'].rolling(10).mean()
    normalized_volume_spikes = df['volume'] / (avg_volume + 1e-7)
    
    # Combined ultra-short momentum with volume spikes
    momentum_volume_combined = ultra_short_momentum * normalized_volume_spikes
    
    # Intraday strength (close position within daily range)
    intraday_strength = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-7)
    
    # Scale combined signal by intraday strength
    scaled_signal = momentum_volume_combined * intraday_strength
    
    # Logarithmic volume transforms
    log_volume = np.log(df['volume'] + 1)
    
    # Multiply complementary factors (scaled signal × log volume)
    factor = scaled_signal * log_volume
    
    # Robust smoothing with Hamming window
    window_size = 5
    hamming_window = np.hamming(window_size)
    factor = factor.rolling(window=window_size, center=True).apply(lambda x: np.sum(x * hamming_window))
    
    return factor
