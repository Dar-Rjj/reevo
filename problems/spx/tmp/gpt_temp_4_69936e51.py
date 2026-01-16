import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(data):
    # Compute Volume-Adjusted Momentum
    # Calculate Price Momentum (5-day)
    momentum = data['close'] / data['close'].shift(5) - 1
    
    # Normalize by Volume (current volume / 20-day rolling mean volume)
    rolling_mean_volume = data['volume'].rolling(window=20, min_periods=1).mean()
    volume_adjustment = data['volume'] / rolling_mean_volume
    volume_adjusted_momentum = momentum * volume_adjustment
    
    # Confirm with Price Range
    # Calculate Daily Range (High - Low)
    daily_range = data['high'] - data['low']
    
    # Scale Momentum by Range (avoid division by zero)
    range_adjusted_momentum = volume_adjusted_momentum / (daily_range + 1e-6)
    
    # Filter by Trend Strength
    # Calculate Trend Slope (10-day linear regression slope of close prices)
    def calculate_slope(series):
        if len(series) < 2:
            return np.nan
        x = np.arange(len(series))
        slope, _, _, _, _ = linregress(x, series)
        return slope
    
    trend_slope = data['close'].rolling(window=10, min_periods=2).apply(calculate_slope, raw=False)
    
    # Adjust Momentum by Slope
    final_factor = range_adjusted_momentum * trend_slope
    
    return final_factor
