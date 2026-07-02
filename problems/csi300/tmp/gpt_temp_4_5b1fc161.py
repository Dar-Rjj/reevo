import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import zscore

def heuristics_v2(df):
    # Calculate mid price
    df['mid'] = (df['high'] + df['low']) / 2
    
    # Calculate daily range
    df['range'] = df['high'] - df['low']
    
    # Avoid division by zero by replacing zeros with NaN (will be filled later)
    df['range'] = df['range'].replace(0, np.nan)
    
    # Calculate price efficiency component
    df['price_efficiency'] = (df['close'] - df['mid']) / df['range']
    
    # Forward fill any NaN values in price efficiency (caused by zero ranges)
    df['price_efficiency'] = df['price_efficiency'].ffill()
    
    # Calculate volume ratio (current volume / 10-day rolling median volume)
    rolling_median_volume = df['volume'].rolling(window=10, min_periods=1).median()
    df['volume_ratio'] = df['volume'] / rolling_median_volume
    
    # Calculate weighted efficiency
    df['weighted_efficiency'] = df['price_efficiency'] * df['volume_ratio']
    
    # Calculate z-score of weighted efficiency (using expanding window to avoid lookahead)
    df['factor'] = df['weighted_efficiency'].expanding().apply(
        lambda x: zscore(x, nan_policy='omit')[-1] if len(x) > 1 else 0
    )
    
    return df['factor']
