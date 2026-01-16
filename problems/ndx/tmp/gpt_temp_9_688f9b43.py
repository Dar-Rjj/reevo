import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Detect Breakout Strength
    rolling_max_high = df['high'].rolling(window=10, min_periods=1).max()
    normalized_breakout = (rolling_max_high - df['close']) / rolling_max_high
    
    # Enhance with Volatility
    intraday_range = (df['high'] - df['low']) / df['open']
    volatility_breakout = normalized_breakout * intraday_range * 100
    
    # Adjust with Volume Confirmation
    def volume_slope(volume_series):
        if len(volume_series) < 5:
            return np.nan
        x = np.arange(len(volume_series))
        slope = linregress(x, volume_series)[0]
        return slope
    
    volume_slopes = df['volume'].rolling(window=5, min_periods=5).apply(volume_slope, raw=False)
    factor = volatility_breakout * volume_slopes
    
    # Apply Tanh normalization
    factor = np.tanh(factor)
    
    return factor
