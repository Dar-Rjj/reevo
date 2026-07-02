import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Price momentum with 3-day lookback (simpler, more responsive)
    momentum = df['close'].pct_change(3)
    
    # Volume breakout using current vs 3-day median (more stable than mean)
    vol_breakout = df['volume'] / df['volume'].rolling(3).median()
    
    # Normalized close position (robust to zero-range days)
    price_position = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-7)
    
    # Combine components with equal weighting
    factor = momentum * vol_breakout * price_position
    
    # Symmetric smoothing with 3-day triangular window
    window_weights = np.array([0.25, 0.5, 0.25])
    factor = factor.rolling(3).apply(lambda x: np.sum(x * window_weights))
    
    return factor
