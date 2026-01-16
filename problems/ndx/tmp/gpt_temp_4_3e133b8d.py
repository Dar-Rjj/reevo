import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Calculate the Normalized Range
    normalized_range = (df['high'] - df['low']) / df['close']
    
    # Calculate the 10-day ATR (Average True Range)
    df['tr'] = np.maximum.reduce([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    ])
    atr_10 = df['tr'].rolling(window=10).mean()
    
    # Adjust the Intraday Momentum Component by dividing by the 10-day ATR
    intraday_momentum = normalized_range / atr_10
    
    # Compute Volume Surge as Volume divided by the 10-day Average Volume
    avg_volume_10 = df['volume'].rolling(window=10).mean()
    volume_surge = df['volume'] / avg_volume_10
    
    # Validate Volume Trend by calculating the slope of Volume over the last 5 days
    volume_trend_slope = df['volume'].rolling(window=5).apply(lambda x: linregress(np.arange(len(x)), x)[0])
    
    # Final Signal Calculation
    factor = intraday_momentum * volume_surge * volume_trend_slope
    
    return factor
