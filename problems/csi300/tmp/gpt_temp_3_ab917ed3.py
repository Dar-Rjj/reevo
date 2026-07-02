import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Ultra-short momentum (1-day % change for increased reactivity)
    ultra_short_momentum = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Volume confirmation (log-transform of current volume / 10-day rolling median volume)
    median_volume = df['volume'].rolling(10).median()
    log_normalized_volume = np.log(df['volume'] / (median_volume + 1e-7))
    
    # Multiplicative combination: momentum × volume confirmation
    momentum_volume_combined = ultra_short_momentum * log_normalized_volume
    
    # Intraday strength (close position within daily range: (close - low) / (high - low))
    intraday_strength = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-7)
    
    # Scale combined signal by intraday strength
    scaled_signal = momentum_volume_combined * intraday_strength
    
    # Volatility adjustment (normalize by rolling 10-day median absolute deviation)
    rolling_mad = df['close'].rolling(10).apply(lambda x: np.median(np.abs(x - np.median(x))))
    volatility_adjusted_signal = scaled_signal / (rolling_mad + 1e-7)
    
    # Robust smoothing (Hamming window smoothing)
    window_size = 3
    hamming_window = np.hamming(window_size)
    factor = volatility_adjusted_signal.rolling(window=window_size, center=True).apply(lambda x: np.sum(x * hamming_window))
    
    return factor
