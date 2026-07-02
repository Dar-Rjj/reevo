import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Ultra-short momentum (1-day % change)
    ultra_short_momentum = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Volume confirmation (log-transform volume)
    log_volume = np.log(df['volume'] + 1)
    
    # Intraday strength (close position within daily range: (close - low) / (high - low))
    intraday_strength = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-7)
    
    # Multiplicative combination: momentum × log-volume × intraday strength
    momentum_volume_strength = ultra_short_momentum * log_volume * intraday_strength
    
    # Volatility adjustment: normalize by rolling median absolute deviation (MAD)
    rolling_mad = momentum_volume_strength.rolling(10).apply(lambda x: np.median(np.abs(x - np.median(x))))
    normalized_factor = momentum_volume_strength / (rolling_mad + 1e-7)
    
    # Robust smoothing (light Hamming window filtering)
    window_size = 3
    hamming_window = np.hamming(window_size)
    smoothed_factor = normalized_factor.rolling(window=window_size, center=True).apply(lambda x: np.sum(x * hamming_window))
    
    return smoothed_factor
