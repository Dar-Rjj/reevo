import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Ultra-short momentum (1-day % change)
    ultra_short_momentum = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Volume spikes (current volume / 5-day rolling average volume)
    avg_volume_5d = df['volume'].rolling(5).mean()
    volume_spikes = df['volume'] / (avg_volume_5d + 1e-7)
    
    # Intraday strength (close position within daily range: (close - low) / (high - low))
    intraday_strength = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-7)
    
    # Logarithmic volume transforms (log of volume + 1)
    log_volume = np.log(df['volume'] + 1)
    
    # Combine ultra-short momentum, volume spikes, intraday strength, and log volume
    factor = ultra_short_momentum * volume_spikes * intraday_strength * log_volume
    
    # Robust smoothing (Hamming window smoothing)
    window_size = 3
    hamming_window = np.hamming(window_size)
    factor = factor.rolling(window=window_size, center=True).apply(lambda x: np.sum(x * hamming_window))
    
    return factor
