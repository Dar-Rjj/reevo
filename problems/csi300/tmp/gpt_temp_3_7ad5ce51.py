import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Price Trend Component
    # Rolling 5-day price slope
    def rolling_slope(x):
        return linregress(np.arange(len(x)), x)[0]
    
    price_slope = df['close'].rolling(window=5).apply(rolling_slope, raw=True)
    
    # Normalize price slope to range [-1, 1]
    min_slope = price_slope.rolling(window=len(df), min_periods=1).min()
    max_slope = price_slope.rolling(window=len(df), min_periods=1).max()
    normalized_price_trend = 2 * ((price_slope - min_slope) / (max_slope - min_slope)) - 1
    
    # Liquidity Momentum
    # Volume change rate
    volume_change_rate = df['volume'] / df['volume'].rolling(window=10, min_periods=1).mean()
    
    # Normalize volume change rate to range [0, 1]
    min_rate = volume_change_rate.rolling(window=len(df), min_periods=1).min()
    max_rate = volume_change_rate.rolling(window=len(df), min_periods=1).max()
    normalized_liquidity_momentum = (volume_change_rate - min_rate) / (max_rate - min_rate)
    
    # Adjusted Momentum
    adjusted_momentum = normalized_price_trend * normalized_liquidity_momentum
    
    # Rolling 5-day smoothing
    smoothed_momentum = adjusted_momentum.rolling(window=5, min_periods=1).mean()
    
    return smoothed_momentum
