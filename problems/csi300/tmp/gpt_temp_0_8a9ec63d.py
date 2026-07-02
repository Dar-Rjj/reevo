import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Ultra-short momentum (2-day % change)
    momentum_2d = (df['close'] - df['close'].shift(2)) / df['close'].shift(2)
    
    # Volume spike detection (current volume vs 20-day median)
    volume_ratio = df['volume'] / (df['volume'].rolling(20).median() + 1e-7)
    log_volume_spike = np.log1p(volume_ratio)
    
    # Combined momentum and volume signal
    raw_signal = momentum_2d * log_volume_spike
    
    # Intraday strength: close position within daily range (0=low, 1=high)
    intraday_strength = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-7)
    
    # Volatility scaling (5-day ATR)
    atr = (df['high'] - df['low']).rolling(5).mean()
    
    # Final factor construction
    factor = raw_signal * intraday_strength / (atr + 1e-7)
    
    # Light smoothing with 5-day Hamming window
    hamming_window = np.hamming(5)
    factor = factor.rolling(5).apply(lambda x: np.sum(x * hamming_window))
    
    return factor
