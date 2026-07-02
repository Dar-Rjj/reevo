import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Medium-term momentum (5-day % change)
    medium_momentum = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    
    # Volume normalization (current volume / 20-day rolling average volume)
    avg_volume = df['volume'].rolling(20).mean()
    normalized_volume = df['volume'] / (avg_volume + 1e-7)
    
    # Intraday strength (close position within daily range: (close - low) / (high - low))
    intraday_strength = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-7)
    
    # Logarithmic transforms (log of volume + 1)
    log_volume = np.log(df['volume'] + 1)
    
    # Price compression (current day range vs 10-day rolling average range)
    daily_range = df['high'] - df['low']
    range_rolling_mean = daily_range.rolling(10).mean()
    range_compression = daily_range / (range_rolling_mean + 1e-7)
    
    # Combine factors: medium momentum × normalized volume × intraday strength × log volume × range compression
    factor = medium_momentum * normalized_volume * intraday_strength * log_volume * range_compression
    
    # Robust smoothing (Gaussian window smoothing)
    window_size = 7
    gaussian_window = np.exp(-0.5 * (np.arange(window_size) - window_size // 2) ** 2)
    factor = factor.rolling(window=window_size, center=True).apply(lambda x: np.sum(x * gaussian_window))
    
    return factor
