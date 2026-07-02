import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Ultra-short momentum (1-day % change)
    ultra_short_momentum = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Intraday strength (close position within daily range: (close - low) / (high - low))
    intraday_strength = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-7)
    
    # Multiplicative combination of ultra-short momentum and intraday strength
    momentum_strength_combined = ultra_short_momentum * intraday_strength
    
    # Volatility-adjusted normalization using rolling range (5-day high - low)
    rolling_range = (df['high'] - df['low']).rolling(5).mean()
    volatility_adjusted = momentum_strength_combined / (rolling_range + 1e-7)
    
    # Volume confirmation using log-transformed volume (log of volume + 1)
    log_volume = np.log(df['volume'] + 1)
    
    # Combine volatility-adjusted signal with log-transformed volume
    factor = volatility_adjusted * log_volume
    
    # Light smoothing using Hamming window (window size 3)
    window_size = 3
    hamming_window = np.hamming(window_size)
    factor = factor.rolling(window=window_size, center=True).apply(lambda x: np.sum(x * hamming_window))
    
    return factor
