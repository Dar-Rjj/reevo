import pandas as pd
import pandas as pd
from scipy.stats import skew

def heuristics_v2(df):
    # Calculate Intraday Range
    intraday_range = df['high'] - df['low']
    
    # Normalize by Close
    normalized_momentum = intraday_range / df['close']
    
    # Calculate Volume Ratio
    rolling_volume_mean = df['volume'].rolling(window=5).mean()
    volume_ratio = df['volume'] / rolling_volume_mean
    
    # Adjust by Volume Strength
    volume_weighted_momentum = normalized_momentum * volume_ratio
    
    # Compute Price Skewness
    price_skewness = df['close'].rolling(window=10).apply(skew)
    
    # Adjust by Price Skewness
    final_factor = volume_weighted_momentum * price_skewness
    
    return final_factor
