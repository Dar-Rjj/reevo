import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate daily range
    daily_range = df['high'] - df['low']
    
    # Calculate 3-day SMA of daily ranges
    sma_range_3d = daily_range.rolling(window=3, min_periods=1).mean()
    
    # Normalized Range
    normalized_range = daily_range / sma_range_3d
    
    # Calculate 5-day SMA of volume
    sma_volume_5d = df['volume'].rolling(window=5, min_periods=1).mean()
    
    # Volume Ratio
    volume_ratio = df['volume'] / sma_volume_5d
    
    # Transformed Volume Ratio
    transformed_volume_ratio = np.where(volume_ratio > 1, np.sqrt(volume_ratio), volume_ratio - 0.5)
    
    # Intraday Return
    intraday_return = df['close'] - df['open']
    
    # Normalized Return
    normalized_return = intraday_return / daily_range
    
    # Combined Efficiency Factor
    combined_efficiency = normalized_range * transformed_volume_ratio * normalized_return
    
    # Apply 3-day Rolling Z-Score
    z_score = combined_efficiency.rolling(window=3, min_periods=1).apply(lambda x: (x.iloc[-1] - x.mean()) / x.std())
    
    return z_score
