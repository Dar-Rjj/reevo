import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Calculate Price Efficiency Ratio
    price_efficiency_ratio = (df['close'] - df['open']) / (df['high'] - df['low'])
    
    # Apply Directional Weighting
    directional_weighting = price_efficiency_ratio * np.sign(df['close'] - df['open'])
    abs_return = np.abs(df['close'] - df['open']) / df['open']
    momentum_factor = directional_weighting * abs_return
    
    # Confirm with Volume Trends
    # Short-Term Volume Surge
    rolling_volume_5 = df['volume'].rolling(window=5, min_periods=1).mean()
    rolling_volume_20 = df['volume'].rolling(window=20, min_periods=1).mean()
    volume_surge = (df['volume'] / rolling_volume_5) * np.sign(rolling_volume_5 - rolling_volume_20)
    
    # Long-Term Volume Alignment
    def calculate_slope(series):
        return linregress(np.arange(len(series)), series).slope
    
    volume_slope = df['volume'].rolling(window=20, min_periods=1).apply(calculate_slope)
    volume_slope_normalized = volume_slope / rolling_volume_20
    volume_alignment_factor = volume_slope_normalized * price_efficiency_ratio
    
    # Combine factors
    factor = momentum_factor * volume_surge * volume_alignment_factor
    
    return factor
