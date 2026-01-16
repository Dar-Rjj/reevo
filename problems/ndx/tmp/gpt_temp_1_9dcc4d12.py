import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Price Trend Component
    # Calculate 5-day Close Price Slope using linear regression
    def calculate_slope(series):
        return linregress(np.arange(len(series)), series).slope
    
    close_slope = df['close'].rolling(window=5).apply(calculate_slope, raw=True)
    
    # Normalize by Volatility: 20-day rolling std of returns
    returns = df['close'].pct_change()
    volatility = returns.rolling(window=20).std()
    price_trend = close_slope / volatility
    
    # Volume Trend Component
    # Calculate 5-day Volume Slope using linear regression
    volume_slope = df['volume'].rolling(window=5).apply(calculate_slope, raw=True)
    
    # Measure Divergence: Multiply Price Trend by Volume Trend and take the negative
    divergence_factor = -(price_trend * volume_slope)
    
    return divergence_factor
