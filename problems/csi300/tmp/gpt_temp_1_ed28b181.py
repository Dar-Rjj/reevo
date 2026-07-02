import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """Adaptive momentum factor incorporating true range scaling, volume acceleration, and price-volume alignment"""
    # Price momentum component (normalized by true range)
    true_range = (df['high'] - df['low']).rolling(5).mean() + 1e-7
    momentum = (df['close'] - df['close'].shift(5)) / true_range
    
    # Volume acceleration (log change in volume)
    volume_acceleration = df['volume'].rolling(10).apply(lambda x: np.log(x[-1] / x[0]) if x[0] != 0 else 0)
    
    # Price-volume alignment (correlation between price change and volume)
    price_change = df['close'].pct_change().rolling(5)
    volume_change = df['volume'].pct_change().rolling(5)
    price_volume_alignment = price_change.corr(volume_change)
    
    # Intraday range efficiency (closing position within day's range)
    range_efficiency = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-7)
    
    # Combine components adaptively
    alpha = momentum * volume_acceleration * price_volume_alignment * (range_efficiency - 0.5)
    
    # Smooth and cap extremes
    alpha = alpha.rolling(10).mean()
    alpha = alpha.clip(lower=-3, upper=3)
    
    return alpha
