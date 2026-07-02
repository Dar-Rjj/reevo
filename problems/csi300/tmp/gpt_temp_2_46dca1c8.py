import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Ultra-short momentum (1-day % change for higher reactivity)
    ultra_short_momentum = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Log-transform volume for cleaner signals
    log_volume = np.log(df['volume'] + 1)
    
    # Combine momentum and volume
    momentum_volume_combined = ultra_short_momentum * log_volume
    
    # Intraday strength (close position within daily range: (close - low) / (high - low))
    intraday_strength = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-7)
    
    # Scale combined signal by intraday strength
    scaled_signal = momentum_volume_combined * intraday_strength
    
    # Volatility adjustment (normalize by rolling 3-day range of close prices for precision)
    rolling_range = df['close'].rolling(3).max() - df['close'].rolling(3).min()
    volatility_adjusted_signal = scaled_signal / (rolling_range + 1e-7)
    
    # Smart smoothing using a Hamming window for minimal distortion
    window_size = 3
    hamming_window = np.hamming(window_size)
    factor = volatility_adjusted_signal.rolling(window=window_size, center=True).apply(lambda x: np.sum(x * hamming_window))
    
    return factor
