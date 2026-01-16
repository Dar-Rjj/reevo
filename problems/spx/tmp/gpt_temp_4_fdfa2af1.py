import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Compute Open-to-Close Return
    df['open_to_close_return'] = (df['close'] - df['open']) / df['open']
    
    # Compute High-Low Range
    df['high_low_range'] = df['high'] - df['low']
    
    # Normalize by Intraday Range (avoid division by zero)
    df['price_efficiency'] = np.where(
        df['high_low_range'] != 0,
        df['open_to_close_return'] / df['high_low_range'],
        0
    )
    
    # Calculate Volume Trend Strength
    def get_slope(series, window):
        slopes = np.zeros(len(series))
        for i in range(window-1, len(series)):
            y = series[i-window+1:i+1].values
            x = np.arange(window)
            slope, _, _, _, _ = linregress(x, y)
            slopes[i] = slope
        return slopes
    
    df['volume_5d_slope'] = get_slope(df['volume'], 5)
    df['volume_20d_slope'] = get_slope(df['volume'], 20)
    
    # Combine Signals (avoid division by zero)
    df['volume_ratio'] = np.where(
        df['volume_20d_slope'] != 0,
        df['volume_5d_slope'] / df['volume_20d_slope'],
        1
    )
    
    # Final factor calculation
    factor = df['price_efficiency'] * df['volume_ratio']
    
    return factor
