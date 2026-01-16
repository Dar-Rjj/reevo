import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Price Momentum Component
    # Intraday Price Momentum
    intraday_momentum = (df['high'] - df['low']) / df['open']
    
    # Rolling Price Momentum
    rolling_mean_close = df['close'].rolling(window=7, min_periods=1).mean()
    rolling_momentum = (df['close'] - rolling_mean_close) / df['close']
    
    # Combine Intraday and Rolling Momentum
    price_momentum = intraday_momentum + rolling_momentum
    
    # Volume Trend Confirmation
    # Volume Slope
    def calculate_slope(series):
        return linregress(np.arange(len(series)), series.values).slope
    
    volume_slope = df['volume'].rolling(window=5, min_periods=1).apply(calculate_slope, raw=False)
    
    # Normalized Volume Slope
    volume_std = df['volume'].rolling(window=5, min_periods=1).std()
    normalized_volume_slope = volume_slope / volume_std
    
    # Combined Factor
    combined_factor = price_momentum * normalized_volume_slope
    
    # Cross-Sectional Normalization
    cross_sectional_mean = combined_factor.mean()
    cross_sectional_std = combined_factor.std()
    normalized_factor = (combined_factor - cross_sectional_mean) / cross_sectional_std
    
    return normalized_factor
