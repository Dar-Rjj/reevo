import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate High-to-Low Range
    high_low_range = df['high'] - df['low']
    
    # Normalize Range Efficiency
    normalized_range_efficiency = high_low_range / df['close']
    
    # Compute 10-day rolling percentile for Volume
    volume_percentile = df['volume'].rolling(window=10).apply(lambda x: (x.rank(pct=True).iloc[-1]), raw=False)
    
    # Raise Normalized Range Efficiency to Exponential Power of Volume Rank
    range_efficiency = normalized_range_efficiency * np.exp(volume_percentile)
    
    # Calculate Volume Deviation
    rolling_avg_volume = df['volume'].rolling(window=10).mean()
    volume_deviation = df['volume'] - rolling_avg_volume
    
    # Calculate Weighted Range Efficiency Score
    weighted_range_efficiency = range_efficiency * volume_deviation
    
    # Use Sign to Confirm Efficiency Direction
    efficiency_score = np.sign(weighted_range_efficiency) * weighted_range_efficiency.abs()
    
    return efficiency_score
