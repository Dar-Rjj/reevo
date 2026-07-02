import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Compute Intraday Momentum
    df['intraday_momentum'] = df['close'] - df['open']
    
    # Normalize by Price Range
    df['price_range'] = df['high'] - df['low']
    df['momentum_strength'] = df['intraday_momentum'] / df['price_range'].replace(0, np.nan)
    
    # Compute Volume Trend (5-day Volume Slope)
    def volume_slope(volume):
        return linregress(np.arange(5), volume)[0]
    
    df['volume_trend'] = df['volume'].rolling(window=5).apply(volume_slope, raw=True)
    
    # Combine Signals
    df['combined_signal'] = df['momentum_strength'] * df['volume_trend']
    
    # Apply 3-day Rolling Median
    factor = df['combined_signal'].rolling(window=3, min_periods=1).median()
    
    return factor
