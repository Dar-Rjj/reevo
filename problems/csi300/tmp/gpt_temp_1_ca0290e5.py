import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Calculate Close-to-Open ratio
    close_open_ratio = (df['close'] - df['open']) / df['open']
    
    # Calculate High-Low range
    high_low_range = df['high'] - df['low']
    
    # Adjust by intraday volatility (avoid division by zero)
    adjusted_momentum = close_open_ratio / (high_low_range.replace(0, np.nan))
    
    # Calculate 5-day volume slope
    volume_slope = pd.Series(index=df.index, dtype=float)
    for i in range(4, len(df)):
        window = df['volume'].iloc[i-4:i+1]
        slope = linregress(np.arange(5), window.values).slope
        volume_slope.iloc[i] = slope
    
    # Combine adjusted momentum with volume trend
    factor = adjusted_momentum * volume_slope
    
    return factor
