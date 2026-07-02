import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Ultra-short momentum with volatility dampening
    momentum = (df['close'] - df['close'].shift(2)) / df['close'].shift(2)
    vol_adjusted_momentum = momentum / (df['close'].pct_change().rolling(5).std() + 1e-7)
    
    # Volume spike detection using log ratios
    volume_ratio = np.log1p(df['volume']) - np.log1p(df['volume'].rolling(10).mean())
    
    # Intraday strength with range normalization
    normalized_range = (df['high'] - df['low']).rolling(5).mean() + 1e-7
    intraday_strength = (df['close'] - df['low']) / normalized_range
    
    # Combined core factor
    core_factor = vol_adjusted_momentum * volume_ratio * intraday_strength
    
    # Hamming window smoothing (5-period)
    window = np.hamming(5)
    smoothed_factor = core_factor.rolling(5).apply(
        lambda x: np.sum(x * window / window.sum())
    )
    
    # Recent weighting (linear decay)
    weights = np.linspace(1, 0.5, len(smoothed_factor))
    weighted_factor = smoothed_factor * weights
    
    return weighted_factor
