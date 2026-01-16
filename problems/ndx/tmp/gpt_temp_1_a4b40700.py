import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Price Trend Component
    # Calculate 5-day Close Price Slope
    def calculate_slope(series, window=5):
        slopes = pd.Series(index=series.index, dtype=float)
        for i in range(window-1, len(series)):
            y = series.iloc[i-window+1:i+1].values
            x = np.arange(window)
            slope = linregress(x, y).slope
            slopes.iloc[i] = slope
        return slopes
    
    close_slopes = calculate_slope(df['close'])
    
    # Normalize by Volatility (20-day rolling std of returns)
    returns = df['close'].pct_change()
    vol = returns.rolling(20).std()
    normalized_price_trend = close_slopes / vol
    
    # Volume Trend Component
    # Calculate 5-day EMA slope of Volume
    volume_ema = df['volume'].ewm(span=5, adjust=False).mean()
    volume_slopes = calculate_slope(volume_ema)
    
    # Measure Divergence
    divergence = - (normalized_price_trend * volume_slopes)
    
    return divergence
