import pandas as pd
import pandas as pd
from scipy.stats import skew

def heuristics_v2(df):
    # Calculate Midpoint Deviation
    midpoint_deviation = (df['high'] + df['low']) / 2 - df['close']
    
    # Smooth Momentum with 5-day Rolling Average
    momentum_smooth = midpoint_deviation.rolling(window=5, min_periods=1).mean()
    
    # Calculate Volume Skewness with 10-day Rolling Window
    volume_skewness = df['volume'].rolling(window=10, min_periods=1).apply(lambda x: skew(x), raw=True)
    
    # Multiply Momentum Stability by Volume Skewness
    factor = momentum_smooth * volume_skewness
    
    return factor
