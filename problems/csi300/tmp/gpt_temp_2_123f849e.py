import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, window=20):
    # Calculate Rolling Mean Price
    rolling_mean = df['close'].rolling(window=window, min_periods=1).mean()
    
    # Normalize by Close Price
    normalized_price = df['close'] / rolling_mean - 1
    
    # Calculate MAD (Median Absolute Deviation)
    mad = normalized_price.rolling(window=window, min_periods=1).apply(lambda x: np.median(np.abs(x - np.median(x))))
    
    # Normalize Deviation
    relative_strength_deviation = (normalized_price / rolling_mean) * 100
    
    return relative_strength_deviation
