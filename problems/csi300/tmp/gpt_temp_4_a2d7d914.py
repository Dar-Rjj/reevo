import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate 5-day Price Slope
    def calculate_slope(series, window=5):
        x = np.arange(window)
        slopes = []
        for i in range(len(series) - window + 1):
            y = series[i:i+window]
            slope = np.polyfit(x, y, 1)[0]
            slopes.append(slope)
        # Pad the beginning with NaN since we lose the first (window-1) values
        return pd.Series([np.nan] * (window - 1) + slopes, index=series.index)
    
    price_slope = calculate_slope(df['close'])
    
    # Calculate 5-day Volume Slope
    volume_slope = calculate_slope(df['volume'])
    
    # Formulate Divergence Signal
    divergence = price_slope - volume_slope
    
    # Normalize by Price Level
    factor = divergence / df['close']
    
    return factor
