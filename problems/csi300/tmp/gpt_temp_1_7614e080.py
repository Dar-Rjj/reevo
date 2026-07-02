import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # 1-day momentum with intraday confirmation
    momentum = (df['close'] - df['open']) / df['open']
    
    # Volume-pressure indicator (log transform with sign preservation)
    signed_log_volume = np.sign(df['volume']) * np.log(np.abs(df['volume']) + 1)
    
    # Price-range efficiency (close position in daily range)
    range_efficiency = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-7)
    
    # Volatility-scaled momentum (1-day price change / 3-day range)
    volatility_scaling = (df['high'].rolling(3).max() - df['low'].rolling(3).min()) + 1e-7
    normalized_momentum = momentum / volatility_scaling
    
    # Composite factor (momentum × volume × efficiency)
    factor = normalized_momentum * signed_log_volume * range_efficiency
    
    # Short-window smoothing (3-day Hamming window)
    window = 3
    hamming_weights = np.hamming(window)
    smoothed_factor = factor.rolling(window).apply(lambda x: np.sum(x * hamming_weights))
    
    # Cross-sectional robustness (rank normalization)
    ranked_factor = smoothed_factor.groupby(smoothed_factor.index).rank(pct=True)
    
    return ranked_factor
