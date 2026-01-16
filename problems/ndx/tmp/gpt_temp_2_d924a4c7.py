import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Price Momentum Component
    intraday_momentum = (df['high'] - df['low']) / df['open']
    rolling_close_mean = df['close'].rolling(window=7, min_periods=1).mean()
    rolling_momentum = (df['close'] - rolling_close_mean) / df['close']
    price_momentum = intraday_momentum + rolling_momentum
    
    # Volume Trend Confirmation
    def linear_slope(series):
        return linregress(np.arange(len(series)), series).slope
    
    volume_slope = df['volume'].rolling(window=5, min_periods=1).apply(linear_slope, raw=True)
    volume_slope_recent = df['volume'].rolling(window=3, min_periods=1).apply(linear_slope, raw=True)
    volume_acceleration = volume_slope_recent - volume_slope
    
    # Combined Factor
    combined_factor = price_momentum * volume_acceleration
    
    # Cross-Sectional Normalization
    mean_combined_factor = combined_factor.mean()
    std_combined_factor = combined_factor.std()
    normalized_factor = (combined_factor - mean_combined_factor) / std_combined_factor
    
    return normalized_factor
