import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Typical Price
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    
    # Calculate Rolling Volume Trend (Linear Slope over past 5 days)
    def rolling_slope(series, window=5):
        slopes = np.zeros(len(series))
        for i in range(window, len(series)):
            x = np.arange(window)
            y = series[i-window:i].values
            slope = np.polyfit(x, y, 1)[0]
            slopes[i] = slope
        return pd.Series(slopes, index=series.index)
    
    volume_trend = rolling_slope(df['volume'])
    
    # Calculate Modified Volume Ratio
    modified_volume_ratio = typical_price / volume_trend
    
    return modified_volume_ratio
